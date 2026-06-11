from __future__ import annotations

import importlib
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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
get_mac_bin_trend = reference_lookup_module.get_mac_bin_trend
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
        "et_trend_smooth_window": None,
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
        "et_trend_smooth_window": 0.20,
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
    @media (max-width: 768px) {
        header[data-testid="stHeader"] {
            height: 3rem;
            background: #0e1117;
        }
        .block-container {
            padding-top: 3.5rem;
        }
    }
    div.st-key-mobile_top_controls {
        display: none;
    }
    @media (max-width: 768px) {
        div.st-key-mobile_top_controls {
            display: block;
            margin-bottom: 1rem;
        }
        section[data-testid="stSidebar"],
        button[data-testid="stExpandSidebarButton"],
        button[data-testid="stCollapseSidebarButton"] {
            display: none !important;
        }
        div.st-key-mobile_top_controls [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(255, 255, 255, 0.18);
            background: #171b24;
        }
        div.st-key-mobile_top_controls [data-testid="stMarkdownContainer"] p {
            margin-bottom: 0;
        }
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


def weighted_age_slope(trend: pd.DataFrame, value_col: str) -> dict | None:
    fit_df = trend[["age_midpoint", value_col, "n_cases_clean"]].dropna().copy()
    if len(fit_df) < 2:
        return None

    x = fit_df["age_midpoint"].astype(float).to_numpy()
    y = fit_df[value_col].astype(float).to_numpy()
    weights = np.sqrt(fit_df["n_cases_clean"].clip(lower=1).astype(float).to_numpy())
    slope, intercept = np.polyfit(x, y, deg=1, w=weights)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "slope_per_year": float(slope),
        "slope_per_10_years": float(slope * 10.0),
        "r_squared": r_squared,
        "n_age_bins": int(len(fit_df)),
        "n_cases_sum": int(fit_df["n_cases_clean"].sum()),
    }


def sync_control(source_key: str, canonical_key: str, mirror_keys: tuple[str, ...] = ()) -> None:
    value = st.session_state[source_key]
    st.session_state[canonical_key] = value
    for mirror_key in mirror_keys:
        st.session_state[mirror_key] = value


def set_control_value(key: str, value: object) -> None:
    if st.session_state.get(key) != value:
        st.session_state[key] = value


def render_input_controls(
    prefix: str,
    agent_config: dict,
    age_bins: list[str],
    target_min: float,
    target_max: float,
    target_step: float,
) -> None:
    other_prefix = "mobile" if prefix == "sidebar" else "sidebar"
    st.selectbox(
        "Age bin",
        age_bins,
        key=f"{prefix}_age_bin",
        on_change=sync_control,
        args=(f"{prefix}_age_bin", "selected_age_bin", (f"{other_prefix}_age_bin",)),
    )
    mode = st.radio(
        "Mode",
        [agent_config["agent_label"], "MAC"],
        horizontal=True,
        key=f"{prefix}_mode",
        on_change=sync_control,
        args=(f"{prefix}_mode", "mode", (f"{other_prefix}_mode",)),
    )
    if mode == agent_config["agent_label"]:
        st.slider(
            f"{agent_config['agent_label']} (%)",
            target_min,
            target_max,
            key=f"{prefix}_agent_value",
            step=target_step,
            on_change=sync_control,
            args=(
                f"{prefix}_agent_value",
                "agent_value",
                (f"{other_prefix}_agent_value",),
            ),
        )
    else:
        st.slider(
            "MAC",
            0.7,
            1.3,
            key=f"{prefix}_mac",
            step=0.1,
            on_change=sync_control,
            args=(f"{prefix}_mac", "mac", (f"{other_prefix}_mac",)),
        )
    st.checkbox(
        "Require usable cell",
        key=f"{prefix}_require_usable",
        on_change=sync_control,
        args=(
            f"{prefix}_require_usable",
            "require_usable",
            (f"{other_prefix}_require_usable",),
        ),
    )


data_dir = ROOT / "data"

available_agents = []
for agent_name, config in AGENT_CONFIGS.items():
    table_path = data_dir / config["table"]
    fallback = config.get("fallback_table")
    fallback_path = data_dir / fallback if fallback else None
    if table_path.exists() or (fallback_path is not None and fallback_path.exists()):
        available_agents.append(agent_name)

if not available_agents:
    st.error("No reference CSV found. Add a file to data/.")
    st.stop()

if (
    "agent_name" not in st.session_state
    or st.session_state["agent_name"] not in available_agents
):
    st.session_state["agent_name"] = available_agents[0]

set_control_value("sidebar_agent", st.session_state["agent_name"])
set_control_value("mobile_agent", st.session_state["agent_name"])

with st.sidebar:
    st.header("Reference Data")

    st.selectbox(
        "Agent",
        available_agents,
        key="sidebar_agent",
        on_change=sync_control,
        args=("sidebar_agent", "agent_name", ("mobile_agent",)),
    )
    agent_name = st.session_state["agent_name"]
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

    if default_path.exists():
        ref_df = load_data(str(default_path))
        data_label = default_path.name
    else:
        st.error("No reference CSV found. Add a file to data/.")
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

    if (
        "selected_age_bin" not in st.session_state
        or st.session_state["selected_age_bin"] not in age_bins
    ):
        st.session_state["selected_age_bin"] = default_age_bin
    if (
        "mode" not in st.session_state
        or st.session_state["mode"] not in (agent_config["agent_label"], "MAC")
    ):
        st.session_state["mode"] = agent_config["agent_label"]
    if (
        "agent_value" not in st.session_state
        or not target_min <= float(st.session_state["agent_value"]) <= target_max
    ):
        st.session_state["agent_value"] = float(target_default)
    if "mac" not in st.session_state:
        st.session_state["mac"] = 1.0
    if "require_usable" not in st.session_state:
        st.session_state["require_usable"] = True

    for control_prefix in ("sidebar", "mobile"):
        set_control_value(
            f"{control_prefix}_agent",
            st.session_state["agent_name"],
        )
        set_control_value(
            f"{control_prefix}_age_bin",
            st.session_state["selected_age_bin"],
        )
        set_control_value(f"{control_prefix}_mode", st.session_state["mode"])
        set_control_value(
            f"{control_prefix}_agent_value",
            float(st.session_state["agent_value"]),
        )
        set_control_value(f"{control_prefix}_mac", float(st.session_state["mac"]))
        set_control_value(
            f"{control_prefix}_require_usable",
            bool(st.session_state["require_usable"]),
        )

    render_input_controls(
        "sidebar",
        agent_config,
        age_bins,
        target_min,
        target_max,
        target_step,
    )

with st.container(border=True, key="mobile_top_controls"):
    st.markdown("**Input**")
    mobile_agent_col, mobile_age_col = st.columns(2)
    with mobile_agent_col:
        st.selectbox(
            "Agent",
            available_agents,
            key="mobile_agent",
            on_change=sync_control,
            args=("mobile_agent", "agent_name", ("sidebar_agent",)),
        )
    with mobile_age_col:
        st.selectbox(
            "Age bin",
            age_bins,
            key="mobile_age_bin",
            on_change=sync_control,
            args=("mobile_age_bin", "selected_age_bin", ("sidebar_age_bin",)),
        )
    mobile_mode_col, mobile_dose_col = st.columns(2)
    with mobile_mode_col:
        mobile_mode = st.radio(
            "Mode",
            [agent_config["agent_label"], "MAC"],
            horizontal=True,
            key="mobile_mode",
            on_change=sync_control,
            args=("mobile_mode", "mode", ("sidebar_mode",)),
        )
    with mobile_dose_col:
        if mobile_mode == agent_config["agent_label"]:
            st.slider(
                f"{agent_config['agent_label']} (%)",
                target_min,
                target_max,
                key="mobile_agent_value",
                step=target_step,
                on_change=sync_control,
                args=("mobile_agent_value", "agent_value", ("sidebar_agent_value",)),
            )
        else:
            st.slider(
                "MAC",
                0.7,
                1.3,
                key="mobile_mac",
                step=0.1,
                on_change=sync_control,
                args=("mobile_mac", "mac", ("sidebar_mac",)),
            )
    st.checkbox(
        "Require usable cell",
        key="mobile_require_usable",
        on_change=sync_control,
        args=(
            "mobile_require_usable",
            "require_usable",
            ("sidebar_require_usable",),
        ),
    )

st.title(agent_config["title"])

selected_age_bin = st.session_state["selected_age_bin"]
age = age_bin_midpoint(selected_age_bin)
mode = st.session_state["mode"]
require_usable = bool(st.session_state["require_usable"])
if mode == agent_config["agent_label"]:
    agent_value = float(st.session_state["agent_value"])
    mac = None
else:
    agent_value = None
    mac = float(st.session_state["mac"])

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

if mode == "MAC":
    age_trend = get_mac_bin_trend(ref_df, row["age_bin"], require_usable=require_usable)
    agent_age_trend = get_mac_age_trend(
        ref_df,
        mac,
        mac_tolerance=agent_config["mac_tolerance"],
        require_usable=require_usable,
    )
    age_trend_title = f"Age Trend At MAC {mac:.2f}"
    concentration_x_col = "mac_bin"
    concentration_x_label = "MAC"
    selected_concentration_x = mac
else:
    age_trend = get_age_bin_trend(ref_df, row["age_bin"], require_usable=require_usable)
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

concentration_smooth_window = (
    agent_config["et_trend_smooth_window"] if mode != "MAC" else None
)

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
metrics[4].metric("Cell N cases", f"{int(row['n_cases_clean'])}")

bis_slope = weighted_age_slope(agent_age_trend, "mean_bis")
sef_slope = weighted_age_slope(agent_age_trend, "mean_sef")
if bis_slope is not None and sef_slope is not None:
    st.subheader("Age Trend Slope")
    if mode == "MAC":
        trend_mode_note = (
            "Trend age bins counts age bins with usable cells in the selected 0.1-MAC bin. "
            "MAC bins use midpoint boundaries between adjacent MAC levels, so values from "
            "0.95 to <1.05 MAC are grouped as MAC 1.0. It can differ from ET mode because "
            "each age bin is matched by age-adjusted MAC, not by a fixed "
            f"{agent_config['agent_label']} concentration."
        )
    else:
        trend_mode_note = (
            f"Trend age bins counts usable age bins at the selected fixed "
            f"{agent_config['agent_label']} concentration. "
            "It can differ from MAC mode because MAC mode matches cells by age-adjusted MAC, "
            "not by a fixed end-tidal concentration."
        )
    slope_cols = st.columns(4)
    slope_cols[0].metric(
        "BIS slope",
        f"{bis_slope['slope_per_10_years']:+.2f} / 10 yr",
    )
    slope_cols[1].metric(
        "SEF slope",
        f"{sef_slope['slope_per_10_years']:+.2f} Hz / 10 yr",
    )
    slope_cols[2].metric("Trend age bins", f"{bis_slope['n_age_bins']}")
    slope_cols[3].metric("Trend total cases", f"{bis_slope['n_cases_sum']}")
    st.caption(
        "Weighted linear trend across matched age-bin means. "
        "Weights use n_cases_clean, so larger reference cells contribute more. "
        "Cell N cases is the selected matched cell only; trend counts summarize the cells used for the age-trend fit. "
        + trend_mode_note
    )

fig = plot_reference_summary(
    row,
    age_trend,
    agent_age_trend,
    agent_label=agent_config["agent_label"],
    age_trend_title=age_trend_title,
    concentration_x_col=concentration_x_col,
    concentration_x_label=concentration_x_label,
    selected_concentration_x=selected_concentration_x,
    concentration_smooth_window=concentration_smooth_window,
)
st.pyplot(fig, width="stretch")
if concentration_smooth_window is not None:
    st.caption(
        f"In ET des mode, faint points show raw 0.1% cells; thick lines show "
        f"n-weighted overlapping smoothing within +/-{concentration_smooth_window:.1f}% ET des."
    )

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
