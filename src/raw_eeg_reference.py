from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")

import matplotlib.pyplot as plt

from vitaldb_raw import has_cached_case_track, load_eeg_window


REPRESENTATIVE_FEATURES = [
    "mean_bis",
    "mean_sef",
    "mean_mac",
    "delta_power_db",
    "theta_power_db",
    "alpha_power_db",
    "beta_power_db",
    "alpha_peak_freq",
    "alpha_peak_power_db",
]


@dataclass(frozen=True)
class RepresentativeRawEEG:
    row: pd.Series
    signal: np.ndarray
    display_signal: np.ndarray
    display_time_sec: np.ndarray
    sample_rate_hz: float
    display_start_offset_sec: float
    display_end_offset_sec: float


@dataclass(frozen=True)
class PowerSpectrum:
    freq_hz: np.ndarray
    mean_power_db: np.ndarray
    lower_power_db: np.ndarray
    upper_power_db: np.ndarray


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def filter_cell_rows(
    rows: pd.DataFrame,
    age_bin: str,
    target_agent: float,
    require_usable: bool = True,
) -> pd.DataFrame:
    cell_rows = rows[
        (rows["age_bin"].astype(str) == str(age_bin))
        & (rows["target_agent"].astype(float).round(1) == round(float(target_agent), 1))
    ].copy()

    if require_usable and "usable_for_reference" in cell_rows.columns:
        cell_rows = cell_rows[_as_bool(cell_rows["usable_for_reference"])]

    return cell_rows.reset_index(drop=True)


def rank_representative_windows(
    rows: pd.DataFrame,
    age_bin: str,
    target_agent: float,
    require_usable: bool = True,
) -> pd.DataFrame:
    cell_rows = filter_cell_rows(
        rows,
        age_bin=age_bin,
        target_agent=target_agent,
        require_usable=require_usable,
    )
    if cell_rows.empty:
        return cell_rows

    feature_cols = [
        column
        for column in REPRESENTATIVE_FEATURES
        if column in cell_rows.columns and cell_rows[column].notna().any()
    ]
    if not feature_cols:
        cell_rows["representative_score"] = 0.0
    else:
        features = cell_rows[feature_cols].apply(pd.to_numeric, errors="coerce")
        medians = features.median(axis=0)
        mad = (features - medians).abs().median(axis=0)
        fallback_scale = features.std(axis=0).replace(0, np.nan)
        scale = mad.replace(0, np.nan).fillna(fallback_scale).fillna(1.0)
        cell_rows["representative_score"] = (
            ((features - medians).abs() / scale).mean(axis=1, skipna=True).fillna(np.inf)
        )

    if "window_quality_score" not in cell_rows.columns:
        cell_rows["window_quality_score"] = 0.0
    if "agent_bin_error" not in cell_rows.columns:
        cell_rows["agent_bin_error"] = 0.0

    return cell_rows.sort_values(
        [
            "representative_score",
            "window_quality_score",
            "agent_bin_error",
            "case_id",
            "window_start_sec",
        ],
        ascending=[True, False, True, True, True],
    ).reset_index(drop=True)


def load_representative_raw_eeg(
    rows: pd.DataFrame,
    age_bin: str,
    target_agent: float,
    require_usable: bool = True,
    track_name: str = "BIS/EEG1_WAV",
    sample_rate_hz: float = 128.0,
    raw_cache_dir: str | Path | None = None,
    display_sec: float = 10.0,
) -> RepresentativeRawEEG:
    ranked = rank_representative_windows(
        rows,
        age_bin=age_bin,
        target_agent=target_agent,
        require_usable=require_usable,
    )
    if ranked.empty:
        raise RuntimeError("No reference rows are available for this cell.")

    errors = []
    for candidate in ranked.itertuples(index=True):
        if raw_cache_dir is not None and not has_cached_case_track(
            case_id=int(candidate.case_id),
            track_name=track_name,
            sample_rate_hz=sample_rate_hz,
            cache_dir=raw_cache_dir,
        ):
            errors.append(f"case {candidate.case_id}: raw EEG cache missing")
            continue

        try:
            signal = load_eeg_window(
                case_id=int(candidate.case_id),
                start_sec=float(candidate.window_start_sec),
                end_sec=float(candidate.window_end_sec),
                track_name=track_name,
                sample_rate_hz=sample_rate_hz,
                cache_dir=raw_cache_dir,
            )
        except Exception as exc:
            errors.append(f"case {candidate.case_id}: {exc}")
            continue

        display_samples = min(signal.size, int(round(display_sec * sample_rate_hz)))
        start_idx = max(0, (signal.size - display_samples) // 2)
        end_idx = start_idx + display_samples
        display_signal = signal[start_idx:end_idx].astype(float)
        display_time_sec = np.arange(display_signal.size) / sample_rate_hz
        selected = ranked.loc[int(candidate.Index)]

        return RepresentativeRawEEG(
            row=selected,
            signal=np.asarray(signal, dtype=np.float32),
            display_signal=np.asarray(display_signal, dtype=np.float32),
            display_time_sec=display_time_sec.astype(np.float32),
            sample_rate_hz=float(sample_rate_hz),
            display_start_offset_sec=float(start_idx / sample_rate_hz),
            display_end_offset_sec=float(end_idx / sample_rate_hz),
        )

    raise RuntimeError("Unable to load representative raw EEG. " + " | ".join(errors[:3]))


def plot_raw_eeg_trace(
    representative: RepresentativeRawEEG,
    agent_label: str = "ET Agent",
) -> plt.Figure:
    row = representative.row
    signal = representative.display_signal.astype(float)
    time_sec = representative.display_time_sec

    fig, ax = plt.subplots(figsize=(11.4, 2.6))
    ax.plot(time_sec, signal, color="#2f6fbb", linewidth=0.8)
    ax.axhline(0, color="0.65", linewidth=0.6, alpha=0.8)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("EEG amplitude")
    ax.set_title(
        "Representative Raw EEG (10 sec): "
        f"Case {int(row['case_id'])}, {agent_label} {float(row['target_agent']):.1f}%"
    )

    finite = signal[np.isfinite(signal)]
    if finite.size:
        lower, upper = np.nanpercentile(finite, [0.5, 99.5])
        if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
            padding = (upper - lower) * 0.15
            ax.set_ylim(lower - padding, upper + padding)

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _interpolate_for_fft(signal: np.ndarray, max_missing_fraction: float = 0.2) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    finite = np.isfinite(values)
    if finite.mean() < 1.0 - max_missing_fraction:
        raise ValueError("Too much missing raw EEG for Fourier analysis.")
    if finite.all():
        return values

    x = np.arange(values.size)
    values = values.copy()
    values[~finite] = np.interp(x[~finite], x[finite], values[finite])
    return values


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def compute_power_spectrum(
    representative: RepresentativeRawEEG,
    epoch_sec: float = 2.0,
    step_sec: float = 0.5,
    freq_max_hz: float = 45.0,
) -> PowerSpectrum:
    values = _interpolate_for_fft(representative.display_signal)
    values = values - np.nanmedian(values)

    sample_rate_hz = representative.sample_rate_hz
    epoch_samples = int(round(epoch_sec * sample_rate_hz))
    step_samples = int(round(step_sec * sample_rate_hz))
    if values.size < epoch_samples:
        raise ValueError("Representative raw EEG segment is too short for FFT.")

    starts = np.arange(0, values.size - epoch_samples + 1, step_samples)
    epochs = np.stack([values[start : start + epoch_samples] for start in starts])
    epochs = epochs - np.nanmean(epochs, axis=1, keepdims=True)
    window = np.hanning(epoch_samples)
    tapered = epochs * window

    fft_values = np.fft.rfft(tapered, axis=1)
    freq_hz = np.fft.rfftfreq(epoch_samples, d=1.0 / sample_rate_hz)
    scale = sample_rate_hz * np.sum(window**2)
    psd = (np.abs(fft_values) ** 2) / scale
    if psd.shape[1] > 2:
        psd[:, 1:-1] *= 2.0

    power_db = 10.0 * np.log10(psd + np.finfo(float).eps)
    keep = (freq_hz >= 0.5) & (freq_hz <= freq_max_hz)
    freq_hz = freq_hz[keep]
    power_db = power_db[:, keep]

    mean_power = np.nanmean(power_db, axis=0)
    spread = np.nanstd(power_db, axis=0)
    lower = _smooth(mean_power - spread)
    upper = _smooth(mean_power + spread)
    mean_power = _smooth(mean_power)

    return PowerSpectrum(
        freq_hz=freq_hz.astype(np.float32),
        mean_power_db=mean_power.astype(np.float32),
        lower_power_db=lower.astype(np.float32),
        upper_power_db=upper.astype(np.float32),
    )


def plot_fourier_power_spectrum(
    representative: RepresentativeRawEEG,
    freq_max_hz: float = 45.0,
) -> plt.Figure:
    spectrum = compute_power_spectrum(
        representative,
        freq_max_hz=freq_max_hz,
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.fill_between(
        spectrum.freq_hz,
        spectrum.lower_power_db,
        spectrum.upper_power_db,
        color="#4c78a8",
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(
        spectrum.freq_hz,
        spectrum.mean_power_db,
        color="#4c78a8",
        linewidth=2.0,
    )
    ax.set_xlim(0, freq_max_hz)
    ax.set_ylim(-30, 30)
    ax.set_xlabel("Frequency [Hz]", fontweight="bold")
    ax.set_ylabel("Power [dB]", fontweight="bold")
    ax.set_title("Fourier Power Spectrum")
    ax.grid(False)
    fig.tight_layout()
    return fig
