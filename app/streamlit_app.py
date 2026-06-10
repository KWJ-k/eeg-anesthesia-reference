from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import dsa_reference as dsa_reference_module
import plotting as plotting_module
import raw_eeg_reference as raw_eeg_reference_module
import reference_lookup as reference_lookup_module


dsa_reference_module = importlib.reload(dsa_reference_module)
plotting_module = importlib.reload(plotting_module)
raw_eeg_reference_module = importlib.reload(raw_eeg_reference_module)
reference_lookup_module = importlib.reload(reference_lookup_module)

find_template_index = dsa_reference_module.find_template_index
find_template_row = dsa_reference_module.find_template_row
load_dsa_template = dsa_reference_module.load_dsa_template
load_template_index = dsa_reference_module.load_template_index
plot_dsa_fourier_spectrum = dsa_reference_module.plot_dsa_fourier_spectrum
plot_dsa_template = dsa_reference_module.plot_dsa_template
plot_reference_summary = plotting_module.plot_reference_summary
load_representative_raw_eeg = raw_eeg_reference_module.load_representative_raw_eeg
plot_raw_eeg_trace = raw_eeg_reference_module.plot_raw_eeg_trace
find_reference_file = reference_lookup_module.find_reference_file
get_agent_age_trend = reference_lookup_module.get_agent_age_trend
get_age_bin_trend = reference_lookup_module.get_age_bin_trend
get_mac_age_trend = reference_lookup_module.get_mac_age_trend
load_reference_table = reference_lookup_module.load_reference_table
lookup_reference = reference_lookup_module.lookup_reference
parse_age_bin = reference_lookup_module.parse_age_bin
row_to_display_dict = reference_lookup_module.row_to_display_dict
sorted_age_bins = reference_lookup_module.sorted_age_bins


AGENT_CONFIGS = {
    "Sevoflurane": {
        "title": "Sevoflurane EEG Reference",
        "table": "reference_table_sevo_all_v2_no_n2o_by_ET.csv",
        "row_table": "reference_rows_sevo_all_v2_no_n2o_qc.csv",
        "fallback_table": "reference_table_sevo_all_v1_by_ET.csv",
        "dsa_dir": "dsa_templates_v1",
        "agent_label": "ET sevo",
        "metric_label": "Matched ET Sevo",
        "data_note": None,
        "mac_tolerance": 0.10,
    },
    "Desflurane": {
        "title": "Desflurane EEG Reference",
        "table": "des_reference_v4_all_0p1/reference_table_des_all_v4_0p1_no_n2o_by_ET.csv",
        "row_table": "des_reference_v4_all_0p1/reference_rows_des_all_v4_0p1_no_n2o_qc.csv",
        "fallback_table": None,
        "dsa_dir": "des_dsa_templates_v4_all_0p1",
        "agent_label": "ET des",
        "metric_label": "Matched ET Des",
        "data_note": "DES workspace is an all-eligible VitalDB build with 0.1% ET des bins.",
        "mac_tolerance": 0.10,
    },
}


st.set_page_config(
    page_title="EEG Reference App",
    layout="wide",
)

st.markdown(
    """
    <style>
    #MainMenu,
    div[data-testid="stToolbar"],
    div[data-testid="stStatusWidget"],
    div[data-testid="stAppDeployButton"] {
        visibility: hidden;
        height: 0;
    }
    div[data-testid="stDecoration"] {
        display: none;
    }
    header[data-testid="stHeader"] {
        height: 0;
        background: transparent;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return load_reference_table(path)


@st.cache_data
def load_reference_rows(path: str) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows["age_bin"] = rows["age_bin"].astype(str)
    rows["target_agent"] = rows["target_agent"].astype(float).round(1)
    return rows


def age_bin_midpoint(age_bin: str) -> float:
    left, right = parse_age_bin(age_bin)
    return (left + right) / 2


data_dir = ROOT / "data"

with st.sidebar:
    st.header("Reference Data")

    available_agents = []
    for agent_name, config in AGENT_CONFIGS.items():
        table_path = data_dir / config["table"]
        fallback = config.get("fallback_table")
        fallback_path = data_dir / fallback if fallback else None
        if table_path.exists() or (fallback_path is not None and fallback_path.exists()):
            available_agents.append(agent_name)

    agent_name = st.selectbox("Agent", available_agents)
    agent_config = AGENT_CONFIGS[agent_name]
    default_path = data_dir / agent_config["table"]
    if not default_path.exists() and agent_config.get("fallback_table"):
        default_path = data_dir / agent_config["fallback_table"]

    dsa_index_path = find_template_index(data_dir, agent_config["dsa_dir"])
    dsa_index = (
        load_template_index(dsa_index_path)
        if dsa_index_path is not None
        else pd.DataFrame()
    )
    dsa_ok = (
        dsa_index[dsa_index["status"].astype(str).str.lower() == "ok"]
        if not dsa_index.empty
        else pd.DataFrame()
    )
    row_table_path = data_dir / agent_config["row_table"]
    reference_rows = (
        load_reference_rows(str(row_table_path))
        if row_table_path.exists()
        else pd.DataFrame()
    )

    uploaded = st.file_uploader("Upload reference CSV", type=["csv"])

    if uploaded is not None:
        ref_df = pd.read_csv(uploaded)
        ref_df["age_bin"] = ref_df["age_bin"].astype(str)
        ref_df["target_agent"] = ref_df["target_agent"].astype(float).round(1)
        if ref_df["usable_cell"].dtype != bool:
            ref_df["usable_cell"] = ref_df["usable_cell"].astype(str).str.lower().eq("true")
        data_label = uploaded.name
    elif default_path.exists():
        ref_df = load_data(str(default_path))
        data_label = default_path.name
    else:
        st.error("No reference CSV found. Add a file to data/ or upload one.")
        st.stop()

    st.caption(f"Using: `{data_label}`")
    if agent_config["data_note"]:
        st.caption(agent_config["data_note"])
    if not dsa_ok.empty:
        st.caption(f"Raw DSA templates: `{len(dsa_ok)}` cells")
    if not reference_rows.empty:
        st.caption(f"Raw EEG windows: `{len(reference_rows)}`")

    st.header("Input")
    age_bins = sorted_age_bins(ref_df)
    default_age_bin = "45-49" if "45-49" in age_bins else age_bins[0]
    selected_age_bin = st.selectbox(
        "Age bin",
        age_bins,
        index=age_bins.index(default_age_bin),
    )
    age = age_bin_midpoint(selected_age_bin)
    mode = st.radio("Mode", [agent_config["agent_label"], "MAC"], horizontal=True)

    target_values = sorted(ref_df["target_agent"].astype(float).round(1).unique())
    target_min = float(min(target_values))
    target_max = float(max(target_values))
    diffs = [
        round(target_values[idx + 1] - target_values[idx], 3)
        for idx in range(len(target_values) - 1)
        if target_values[idx + 1] > target_values[idx]
    ]
    target_step = float(min(diffs)) if diffs else 0.1
    target_default = (
        2.0
        if agent_name == "Sevoflurane" and 2.0 in target_values
        else target_values[len(target_values) // 2]
    )

    if mode == agent_config["agent_label"]:
        agent_value = st.slider(
            f"{agent_config['agent_label']} (%)",
            target_min,
            target_max,
            float(target_default),
            target_step,
        )
        mac = None
    else:
        agent_value = None
        mac = st.slider("MAC", 0.7, 1.3, 1.0, 0.1)

    require_usable = st.checkbox("Require usable cell", value=True)


st.title(agent_config["title"])

row, err = lookup_reference(
    ref_df,
    age=age,
    agent_value=agent_value,
    mac=mac,
    mac_tolerance=agent_config["mac_tolerance"] if mac is not None else None,
    require_usable=require_usable,
)

if err:
    st.warning(err)
    st.stop()

display_row = row_to_display_dict(row, selected_age_bin)
display_row.pop("age")
if "matched_agent" in display_row:
    display_row[agent_config["metric_label"]] = display_row.pop("matched_agent")
display_row = {"input_age_bin": selected_age_bin, **display_row}
result_df = pd.DataFrame([display_row])

age_trend = get_age_bin_trend(ref_df, row["age_bin"], require_usable=require_usable)
if mode == "MAC":
    agent_age_trend = get_mac_age_trend(
        ref_df,
        mac,
        mac_tolerance=agent_config["mac_tolerance"],
        require_usable=require_usable,
    )
    age_trend_title = f"Age Trend At MAC {mac:.2f}"
    concentration_x_col = "mean_mac"
    concentration_x_label = "MAC"
    selected_concentration_x = mac
else:
    agent_age_trend = get_agent_age_trend(
        ref_df,
        row["target_agent"],
        require_usable=require_usable,
    )
    age_trend_title = (
        f"Age Trend At {agent_config['agent_label']} {row['target_agent']:.1f}%"
    )
    concentration_x_col = "target_agent"
    concentration_x_label = f"{agent_config['agent_label']} (%)"
    selected_concentration_x = float(row["target_agent"])

template_row = (
    find_template_row(dsa_index, row["age_bin"], row["target_agent"])
    if not dsa_ok.empty
    else None
)
dsa_template = load_dsa_template(template_row["template_path"]) if template_row is not None else None

representative = None
representative_error = None
if not reference_rows.empty:
    try:
        representative = load_representative_raw_eeg(
            reference_rows,
            age_bin=row["age_bin"],
            target_agent=row["target_agent"],
            require_usable=require_usable,
            raw_cache_dir=data_dir / "raw_vitaldb_cache",
            display_sec=10.0,
        )
    except Exception as exc:
        representative_error = str(exc)

if dsa_template is not None or representative is not None or representative_error:
    if representative is not None:
        st.subheader("Representative Raw EEG")
        st.pyplot(
            plot_raw_eeg_trace(
                representative,
                agent_label=agent_config["agent_label"],
            ),
            width="stretch",
        )
        rep_row = representative.row
        display_start = (
            float(rep_row["window_start_sec"])
            + representative.display_start_offset_sec
        )
        display_end = (
            float(rep_row["window_start_sec"])
            + representative.display_end_offset_sec
        )
        st.caption(
            "Selected from the same cell by median-feature distance. "
            f"Case {int(rep_row['case_id'])}, {display_start:.1f}-{display_end:.1f}s."
        )
    elif representative_error:
        st.info(f"No cached representative raw EEG is available: {representative_error}")

    dsa_col, spectrum_col = st.columns(2)
    with dsa_col:
        if dsa_template is not None:
            st.subheader("Raw VitalDB DSA")
            st.pyplot(
                plot_dsa_template(dsa_template, agent_label=agent_config["agent_label"]),
                width="content",
            )
            st.caption(
                "This DSA is computed from VitalDB raw EEG with FFT-based 1 Hz frequency bins."
            )
        else:
            st.info(
                "No raw DSA template has been built yet for this age / concentration cell."
            )

    with spectrum_col:
        if dsa_template is not None:
            st.subheader("Fourier Analysis")
            st.pyplot(
                plot_dsa_fourier_spectrum(
                    dsa_template,
                    reference_sef_hz=float(row["mean_sef"]),
                ),
                width="content",
            )
            st.caption(
                "Power spectrum collapsed from the DSA template. "
                "The red dashed line is the matched SEF from the reference table. "
                "The shaded band shows time-axis IQR within the DSA, not subject-level CI."
            )
        else:
            st.info("No Fourier graph is available without a DSA template.")

st.subheader("Matched Reference")
st.dataframe(result_df, width="stretch", hide_index=True)

metrics = st.columns(5)
metrics[0].metric("BIS", f"{row['mean_bis']:.1f}")
metrics[1].metric("SEF", f"{row['mean_sef']:.1f} Hz")
metrics[2].metric("Alpha Power", f"{row['alpha_power_db']:.1f} dB")
if mode == "MAC":
    metrics[3].metric(agent_config["metric_label"], f"{row['target_agent']:.1f}%")
else:
    metrics[3].metric("MAC", f"{row['mean_mac']:.2f}")
metrics[4].metric("N Cases", f"{int(row['n_cases_clean'])}")

fig = plot_reference_summary(
    row,
    age_trend,
    agent_age_trend,
    agent_label=agent_config["agent_label"],
    age_trend_title=age_trend_title,
    concentration_x_col=concentration_x_col,
    concentration_x_label=concentration_x_label,
    selected_concentration_x=selected_concentration_x,
)
st.pyplot(fig, width="stretch")

if not dsa_ok.empty:
    with st.expander("Raw DSA template coverage"):
        st.dataframe(
            dsa_ok[
                [
                    "age_bin",
                    "target_agent",
                    "n_windows_requested",
                    "template_path",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

with st.expander("Age-bin trend table"):
    st.dataframe(
        age_trend[
            [
                "target_agent",
                "n_cases_clean",
                "mean_mac",
                "mean_bis",
                "mean_sef",
                "delta_power_db",
                "theta_power_db",
                "alpha_power_db",
                "beta_power_db",
                "usable_cell",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
