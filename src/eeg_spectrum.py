from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpectrogramResult:
    time_sec: np.ndarray
    freq_bin_start_hz: np.ndarray
    freq_bin_end_hz: np.ndarray
    freq_center_hz: np.ndarray
    power_db: np.ndarray
    valid_epoch_fraction: float


def interpolate_missing(signal: np.ndarray, max_missing_fraction: float = 0.2) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    finite = np.isfinite(values)
    if finite.mean() < 1.0 - max_missing_fraction:
        raise ValueError(
            f"Too much missing EEG data: {(1.0 - finite.mean()):.1%} missing."
        )
    if finite.all():
        return values

    x = np.arange(values.size)
    values = values.copy()
    values[~finite] = np.interp(x[~finite], x[finite], values[finite])
    return values


def robust_detrend(signal: np.ndarray) -> np.ndarray:
    values = interpolate_missing(signal)
    return values - np.nanmedian(values)


def epoch_signal(
    signal: np.ndarray,
    sample_rate_hz: float,
    epoch_sec: float,
    step_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = robust_detrend(signal)
    epoch_samples = int(round(epoch_sec * sample_rate_hz))
    step_samples = int(round(step_sec * sample_rate_hz))
    if epoch_samples <= 0 or step_samples <= 0:
        raise ValueError("epoch_sec and step_sec must produce at least one sample.")
    if values.size < epoch_samples:
        raise ValueError(
            f"EEG window is too short: {values.size} samples for {epoch_samples}."
        )

    starts = np.arange(0, values.size - epoch_samples + 1, step_samples)
    epochs = np.stack([values[start : start + epoch_samples] for start in starts])
    centers = (starts + epoch_samples / 2) / sample_rate_hz
    return centers, epochs


def _power_spectrum_db(
    epochs: np.ndarray,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(epochs.shape[1])
    demeaned = epochs - np.nanmean(epochs, axis=1, keepdims=True)
    tapered = demeaned * window
    fft_values = np.fft.rfft(tapered, axis=1)
    freqs = np.fft.rfftfreq(epochs.shape[1], d=1.0 / sample_rate_hz)

    scale = sample_rate_hz * np.sum(window**2)
    psd = (np.abs(fft_values) ** 2) / scale
    if psd.shape[1] > 2:
        psd[:, 1:-1] *= 2.0
    power_db = 10.0 * np.log10(psd + np.finfo(float).eps)
    return freqs, power_db


def bin_power_1hz(
    freqs: np.ndarray,
    power_db: np.ndarray,
    freq_min_hz: float = 0.0,
    freq_max_hz: float = 40.0,
    bin_width_hz: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(freq_min_hz, freq_max_hz, bin_width_hz)
    ends = starts + bin_width_hz
    centers = starts + bin_width_hz / 2.0

    linear_power = np.power(10.0, power_db / 10.0)
    binned = np.full((starts.size, power_db.shape[0]), np.nan)

    for idx, (left, right) in enumerate(zip(starts, ends)):
        mask = (freqs >= left) & (freqs < right)
        if not mask.any():
            continue
        binned[idx] = 10.0 * np.log10(
            np.nanmean(linear_power[:, mask], axis=1) + np.finfo(float).eps
        )

    return starts, ends, centers, binned


def compute_dsa_1hz(
    signal: np.ndarray,
    sample_rate_hz: float = 128.0,
    epoch_sec: float = 4.0,
    step_sec: float = 1.0,
    freq_max_hz: float = 40.0,
) -> SpectrogramResult:
    time_sec, epochs = epoch_signal(signal, sample_rate_hz, epoch_sec, step_sec)

    finite_epoch_fraction = np.isfinite(epochs).all(axis=1).mean()
    freqs, power_db = _power_spectrum_db(epochs, sample_rate_hz)
    starts, ends, centers, binned = bin_power_1hz(
        freqs,
        power_db,
        freq_min_hz=0.0,
        freq_max_hz=freq_max_hz,
        bin_width_hz=1.0,
    )
    return SpectrogramResult(
        time_sec=time_sec,
        freq_bin_start_hz=starts,
        freq_bin_end_hz=ends,
        freq_center_hz=centers,
        power_db=binned,
        valid_epoch_fraction=float(finite_epoch_fraction),
    )


def average_dsa(results: list[SpectrogramResult]) -> SpectrogramResult:
    if not results:
        raise ValueError("At least one spectrogram result is required.")

    first = results[0]
    min_time = min(result.power_db.shape[1] for result in results)
    stack = np.stack([result.power_db[:, :min_time] for result in results])
    return SpectrogramResult(
        time_sec=first.time_sec[:min_time],
        freq_bin_start_hz=first.freq_bin_start_hz,
        freq_bin_end_hz=first.freq_bin_end_hz,
        freq_center_hz=first.freq_center_hz,
        power_db=np.nanmedian(stack, axis=0),
        valid_epoch_fraction=float(
            np.nanmean([result.valid_epoch_fraction for result in results])
        ),
    )


def spectrogram_to_long_table(
    result: SpectrogramResult,
    case_id: int | None = None,
    age_bin: str | None = None,
    target_agent: float | None = None,
) -> pd.DataFrame:
    rows = []
    for freq_idx, center in enumerate(result.freq_center_hz):
        for time_idx, time_sec in enumerate(result.time_sec):
            rows.append(
                {
                    "case_id": case_id,
                    "age_bin": age_bin,
                    "target_agent": target_agent,
                    "time_sec": time_sec,
                    "freq_center_hz": center,
                    "freq_bin_start_hz": result.freq_bin_start_hz[freq_idx],
                    "freq_bin_end_hz": result.freq_bin_end_hz[freq_idx],
                    "power_db": result.power_db[freq_idx, time_idx],
                }
            )
    return pd.DataFrame(rows)


def band_power_from_spectrogram(
    result: SpectrogramResult,
    freq_left_hz: float,
    freq_right_hz: float,
) -> float:
    mask = (result.freq_center_hz >= freq_left_hz) & (
        result.freq_center_hz < freq_right_hz
    )
    if not mask.any():
        return float("nan")
    linear_power = np.power(10.0, result.power_db[mask] / 10.0)
    return float(10.0 * np.log10(np.nanmean(linear_power) + np.finfo(float).eps))


def alpha_peak_from_spectrogram(
    result: SpectrogramResult,
    freq_left_hz: float = 8.0,
    freq_right_hz: float = 13.0,
) -> tuple[float, float]:
    mask = (result.freq_center_hz >= freq_left_hz) & (
        result.freq_center_hz < freq_right_hz
    )
    if not mask.any():
        return float("nan"), float("nan")

    mean_power = np.nanmean(result.power_db[mask], axis=1)
    peak_idx = int(np.nanargmax(mean_power))
    freqs = result.freq_center_hz[mask]
    return float(freqs[peak_idx]), float(mean_power[peak_idx])
