from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_EEG_TRACKS = ("BIS/EEG1_WAV", "BIS/EEG2_WAV")
DEFAULT_NUMERIC_TRACKS = ("BIS/BIS", "BIS/SEF", "BIS/SQI")


def _import_vitaldb():
    try:
        import vitaldb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The `vitaldb` package is required to download raw VitalDB EEG. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return vitaldb


def load_case_tracks(
    case_id: int,
    track_names: Sequence[str],
    interval_sec: float,
) -> np.ndarray:
    vitaldb = _import_vitaldb()
    values = vitaldb.load_case(int(case_id), list(track_names), interval=float(interval_sec))
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return values


def _safe_track_name(track_name: str) -> str:
    return track_name.replace("/", "_").replace("\\", "_")


def case_track_cache_path(
    case_id: int,
    track_name: str,
    sample_rate_hz: float,
    cache_dir: str | Path,
) -> Path:
    return (
        Path(cache_dir)
        / f"case_{int(case_id)}_{_safe_track_name(track_name)}_{sample_rate_hz:g}hz.npz"
    )


def has_cached_case_track(
    case_id: int,
    track_name: str,
    sample_rate_hz: float,
    cache_dir: str | Path | None,
) -> bool:
    if cache_dir is None:
        return False
    return case_track_cache_path(case_id, track_name, sample_rate_hz, cache_dir).exists()


def load_case_track_cached(
    case_id: int,
    track_name: str,
    sample_rate_hz: float,
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    if cache_dir is None:
        values = load_case_tracks(
            case_id=case_id,
            track_names=[track_name],
            interval_sec=1.0 / sample_rate_hz,
        )
        return values[:, 0]

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = case_track_cache_path(case_id, track_name, sample_rate_hz, cache_dir)
    if cache_path.exists():
        with np.load(cache_path) as cached:
            return np.asarray(cached["signal"], dtype=float)

    values = load_case_tracks(
        case_id=case_id,
        track_names=[track_name],
        interval_sec=1.0 / sample_rate_hz,
    )[:, 0]
    np.savez_compressed(
        cache_path,
        signal=np.asarray(values, dtype=np.float32),
        case_id=int(case_id),
        track_name=str(track_name),
        sample_rate_hz=float(sample_rate_hz),
    )
    return values


def load_eeg_window(
    case_id: int,
    start_sec: float,
    end_sec: float,
    track_name: str = "BIS/EEG1_WAV",
    sample_rate_hz: float = 128.0,
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec.")

    values = load_case_track_cached(
        case_id=case_id,
        track_name=track_name,
        sample_rate_hz=sample_rate_hz,
        cache_dir=cache_dir,
    )

    start_idx = int(round(start_sec * sample_rate_hz))
    end_idx = int(round(end_sec * sample_rate_hz))
    if start_idx < 0 or end_idx > values.size:
        raise ValueError(
            f"Requested window {start_sec:.1f}-{end_sec:.1f}s is outside case {case_id} "
            f"with {values.size / sample_rate_hz:.1f}s of EEG."
        )
    return values[start_idx:end_idx]


def save_raw_eeg_window(
    output_path: str | Path,
    signal: np.ndarray,
    case_id: int,
    start_sec: float,
    end_sec: float,
    track_name: str,
    sample_rate_hz: float,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        signal=np.asarray(signal, dtype=np.float32),
        case_id=int(case_id),
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        track_name=str(track_name),
        sample_rate_hz=float(sample_rate_hz),
    )
