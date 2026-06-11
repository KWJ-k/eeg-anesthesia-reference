from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")

import matplotlib.pyplot as plt


def find_template_index(
    data_dir: str | Path,
    template_dir: str = "dsa_templates_v1",
) -> Path | None:
    path = Path(data_dir) / template_dir / "template_index.csv"
    if path.exists():
        return path
    return None


def load_template_index(path: str | Path) -> pd.DataFrame:
    index = pd.read_csv(path)
    if index.empty:
        return index
    index["age_bin"] = index["age_bin"].astype(str)
    index["target_agent"] = index["target_agent"].astype(float).round(1)
    if "template_path" in index.columns:
        index_path = Path(path).resolve()
        repo_root = index_path.parents[2]
        index_dir = index_path.parent

        def normalize_template_path(template_path: str | Path) -> str:
            normalized = Path(str(template_path).replace("\\", "/"))
            if normalized.is_absolute():
                return str(normalized)
            if normalized.parts and normalized.parts[0] == "data":
                return str(repo_root / normalized)
            return str(index_dir / normalized)

        index["template_path"] = index["template_path"].apply(normalize_template_path)
    return index


def find_template_row(
    index: pd.DataFrame,
    age_bin: str,
    target_agent: float,
) -> pd.Series | None:
    if index.empty:
        return None
    matches = index[
        (index["age_bin"].astype(str) == str(age_bin))
        & (index["target_agent"].round(1) == round(float(target_agent), 1))
        & (index["status"].astype(str).str.lower() == "ok")
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def load_dsa_template(template_path: str | Path) -> dict[str, np.ndarray]:
    with np.load(template_path) as data:
        return {key: data[key] for key in data.files}


def plot_dsa_template(template: dict[str, np.ndarray], agent_label: str = "ET Agent"):
    time_sec = template["time_sec"]
    freq_center_hz = template["freq_center_hz"]
    power_db = template["power_db"]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    image = ax.imshow(
        power_db,
        aspect="auto",
        origin="lower",
        extent=[
            float(time_sec[0]),
            float(time_sec[-1]),
            float(freq_center_hz[0]),
            float(freq_center_hz[-1]),
        ],
        cmap="jet",
        vmin=-30,
        vmax=15,
    )
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(
        f"Raw VitalDB DSA: Age {str(template['age_bin'])}, {agent_label} {float(template['target_agent']):.1f}%"
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("Power (dB)")
    fig.tight_layout()
    return fig


def plot_dsa_fourier_spectrum(
    template: dict[str, np.ndarray],
    reference_sef_hz: float | None = None,
):
    freq_center_hz = np.asarray(template["freq_center_hz"], dtype=float)
    power_db = np.asarray(template["power_db"], dtype=float)

    center_power = np.nanmedian(power_db, axis=1)
    lower_power = np.nanpercentile(power_db, 25, axis=1)
    upper_power = np.nanpercentile(power_db, 75, axis=1)
    keep = (freq_center_hz >= 0.5) & (freq_center_hz <= 45.0)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    band_regions = [
        ("Delta", 0.5, 4.0, "#4C78A8"),
        ("Theta", 4.0, 8.0, "#F58518"),
        ("Alpha", 8.0, 13.0, "#E45756"),
        ("Beta", 13.0, 30.0, "#72B7B2"),
    ]
    for label, left, right, color in band_regions:
        ax.axvspan(left, right, color=color, alpha=0.08, linewidth=0)
        ax.text(
            (left + right) / 2,
            -27,
            label,
            color=color,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            alpha=0.8,
        )

    ax.fill_between(
        freq_center_hz[keep],
        lower_power[keep],
        upper_power[keep],
        color="#9CA3AF",
        alpha=0.36,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        freq_center_hz[keep],
        center_power[keep],
        color="#0F3B63",
        linewidth=2.4,
        zorder=3,
    )
    if reference_sef_hz is not None and np.isfinite(reference_sef_hz):
        ax.axvline(
            reference_sef_hz,
            color="#D62728",
            linestyle="--",
            linewidth=1.5,
        )
        ax.text(
            reference_sef_hz + 0.6,
            23,
            f"Reference SEF\n{reference_sef_hz:.1f} Hz",
            color="#D62728",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_xlim(0, 45)
    ax.set_ylim(-30, 30)
    ax.set_xlabel("Frequency [Hz]", fontweight="bold")
    ax.set_ylabel("Power [dB]", fontweight="bold")
    ax.set_title("DSA-Derived Fourier Spectrum")
    fig.tight_layout()
    return fig
