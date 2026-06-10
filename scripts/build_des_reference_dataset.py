from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_spectrum import (
    alpha_peak_from_spectrogram,
    band_power_from_spectrogram,
    compute_dsa_1hz,
)
from vitaldb_raw import load_eeg_window


NUMERIC_TRACKS = [
    "Primus/EXP_DES",
    "Primus/MAC",
    "BIS/BIS",
    "BIS/SEF",
    "BIS/SQI",
    "Primus/FEN2O",
    "Primus/FIN2O",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a DES reference row/table dataset from VitalDB raw tracks."
    )
    parser.add_argument("--output-dir", default="data/des_reference_v1")
    parser.add_argument("--raw-cache-dir", default="data/raw_vitaldb_cache")
    parser.add_argument("--eeg-track", default="BIS/EEG1_WAV")
    parser.add_argument("--sample-rate", type=float, default=128.0)
    parser.add_argument("--window-sec", type=int, default=120)
    parser.add_argument("--target-start", type=float, default=3.0)
    parser.add_argument("--target-stop", type=float, default=9.0)
    parser.add_argument("--target-step", type=float, default=0.5)
    parser.add_argument("--target-tolerance", type=float, default=0.25)
    parser.add_argument("--age-min", type=int, default=20)
    parser.add_argument("--age-max", type=int, default=74)
    parser.add_argument("--age-bin-width", type=int, default=5)
    parser.add_argument("--sqi-min", type=float, default=80.0)
    parser.add_argument("--min-agent-fraction", type=float, default=0.9)
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    parser.add_argument("--n2o-max", type=float, default=1.0)
    parser.add_argument("--min-cases-per-cell", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-start", type=int, default=0)
    return parser.parse_args()


def _import_vitaldb():
    try:
        import vitaldb  # type: ignore
    except ImportError as exc:
        raise ImportError("Install vitaldb with `pip install -r requirements.txt`.") from exc
    return vitaldb


def target_grid(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step)) + 1
    return [round(start + idx * step, 1) for idx in range(count)]


def age_to_bin(age: float, age_min: int, age_max: int, width: int) -> str | None:
    if pd.isna(age) or age < age_min or age > age_max:
        return None
    left = age_min + int((age - age_min) // width) * width
    right = left + width - 1
    return f"{left}-{right}"


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=window).mean().to_numpy()


def rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=window).max().to_numpy()


def best_window_for_target(
    tracks: pd.DataFrame,
    target: float,
    window_sec: int,
    tolerance: float,
    sqi_min: float,
    min_agent_fraction: float,
    min_valid_fraction: float,
    n2o_max: float,
) -> dict | None:
    exp_des = tracks["exp_des"].to_numpy(dtype=float)
    mac = tracks["mac"].to_numpy(dtype=float)
    bis = tracks["bis"].to_numpy(dtype=float)
    sef = tracks["sef"].to_numpy(dtype=float)
    sqi = tracks["sqi"].to_numpy(dtype=float)
    fen2o = tracks["fen2o"].fillna(0.0).to_numpy(dtype=float)
    fin2o = tracks["fin2o"].fillna(0.0).to_numpy(dtype=float)

    finite = (
        np.isfinite(exp_des)
        & np.isfinite(mac)
        & np.isfinite(bis)
        & np.isfinite(sef)
        & np.isfinite(sqi)
    )
    near_agent = np.isfinite(exp_des) & (np.abs(exp_des - target) <= tolerance)
    good_sqi = np.isfinite(sqi) & (sqi >= sqi_min)

    valid_fraction = rolling_mean(finite.astype(float), window_sec)
    agent_fraction = rolling_mean(near_agent.astype(float), window_sec)
    sqi_fraction = rolling_mean(good_sqi.astype(float), window_sec)
    max_n2o = np.maximum(rolling_max(fen2o, window_sec), rolling_max(fin2o, window_sec))
    mean_agent = rolling_mean(exp_des, window_sec)
    mean_mac = rolling_mean(mac, window_sec)
    mean_bis = rolling_mean(bis, window_sec)
    mean_sef = rolling_mean(sef, window_sec)
    mean_sqi = rolling_mean(sqi, window_sec)
    agent_error = np.abs(mean_agent - target)

    candidate = (
        (valid_fraction >= min_valid_fraction)
        & (agent_fraction >= min_agent_fraction)
        & (sqi_fraction >= min_valid_fraction)
        & (max_n2o <= n2o_max)
        & np.isfinite(agent_error)
    )
    if not candidate.any():
        return None

    score = (
        agent_fraction * 4
        + valid_fraction
        + sqi_fraction
        + mean_sqi / 100
        - agent_error
    )
    score[~candidate] = -np.inf
    end_idx = int(np.nanargmax(score))
    start_idx = end_idx - window_sec + 1
    if start_idx < 0:
        return None

    return {
        "window_start_sec": float(start_idx),
        "window_end_sec": float(end_idx + 1),
        "mean_agent": float(mean_agent[end_idx]),
        "mean_mac": float(mean_mac[end_idx]),
        "mean_bis": float(mean_bis[end_idx]),
        "mean_sef": float(mean_sef[end_idx]),
        "mean_sqi": float(mean_sqi[end_idx]),
        "window_quality_score": float(score[end_idx]),
        "window_agent_fraction": float(agent_fraction[end_idx]),
        "window_sqi_fraction": float(sqi_fraction[end_idx]),
        "window_valid_fraction": float(valid_fraction[end_idx]),
        "agent_bin_error": float(agent_error[end_idx]),
        "max_n2o": float(max_n2o[end_idx]),
    }


def load_numeric_tracks(case_id: int) -> pd.DataFrame:
    vitaldb = _import_vitaldb()
    values = np.asarray(vitaldb.load_case(case_id, NUMERIC_TRACKS, interval=1), dtype=float)
    if values.ndim != 2 or values.shape[1] != len(NUMERIC_TRACKS):
        raise ValueError(f"Unexpected track matrix shape: {values.shape}")
    tracks = pd.DataFrame(
        values,
        columns=["exp_des", "mac", "bis", "sef", "sqi", "fen2o", "fin2o"],
    )
    tracks[["exp_des", "mac", "bis", "sef", "sqi"]] = tracks[
        ["exp_des", "mac", "bis", "sef", "sqi"]
    ].ffill()
    tracks[["fen2o", "fin2o"]] = tracks[["fen2o", "fin2o"]].ffill().fillna(0.0)
    return tracks


def add_eeg_features(
    row: dict,
    case_id: int,
    eeg_track: str,
    sample_rate_hz: float,
    raw_cache_dir: str,
) -> dict:
    signal = load_eeg_window(
        case_id=case_id,
        start_sec=row["window_start_sec"],
        end_sec=row["window_end_sec"],
        track_name=eeg_track,
        sample_rate_hz=sample_rate_hz,
        cache_dir=raw_cache_dir,
    )
    dsa = compute_dsa_1hz(signal, sample_rate_hz=sample_rate_hz, freq_max_hz=40.0)
    alpha_freq, alpha_power = alpha_peak_from_spectrogram(dsa)

    row.update(
        {
            "delta_power_db": band_power_from_spectrogram(dsa, 0.5, 4.0),
            "theta_power_db": band_power_from_spectrogram(dsa, 4.0, 8.0),
            "alpha_power_db": band_power_from_spectrogram(dsa, 8.0, 13.0),
            "beta_power_db": band_power_from_spectrogram(dsa, 13.0, 30.0),
            "alpha_peak_freq": alpha_freq,
            "alpha_peak_power_db": alpha_power,
            "valid_epoch_fraction": dsa.valid_epoch_fraction,
        }
    )
    row["alpha_delta_diff_db"] = row["alpha_power_db"] - row["delta_power_db"]
    return row


def build_reference_table(rows: pd.DataFrame, min_cases_per_cell: int) -> pd.DataFrame:
    grouped = rows.groupby(["agent", "reference_sex", "age_bin", "target_agent"])
    records = []
    for keys, group in grouped:
        agent, reference_sex, age_bin, target_agent = keys
        record = {
            "agent": agent,
            "reference_sex": reference_sex,
            "age_bin": age_bin,
            "target_agent": target_agent,
            "n_cases_clean": group["case_id"].nunique(),
            "n_windows_clean": len(group),
            "n_female": int((group["sex"] == "F").sum()),
            "n_male": int((group["sex"] == "M").sum()),
            "mean_age": group["age"].mean(),
            "mean_agent": group["mean_agent"].mean(),
            "sd_agent": group["mean_agent"].std(),
            "mean_mac": group["mean_mac"].mean(),
            "mean_bis": group["mean_bis"].mean(),
            "sd_bis": group["mean_bis"].std(),
            "mean_sef": group["mean_sef"].mean(),
            "sd_sef": group["mean_sef"].std(),
            "mean_sqi": group["mean_sqi"].mean(),
            "delta_power_db": group["delta_power_db"].mean(),
            "theta_power_db": group["theta_power_db"].mean(),
            "alpha_power_db": group["alpha_power_db"].mean(),
            "beta_power_db": group["beta_power_db"].mean(),
            "alpha_peak_freq": group["alpha_peak_freq"].median(),
            "alpha_peak_power_db": group["alpha_peak_power_db"].mean(),
            "alpha_delta_diff_db": group["alpha_delta_diff_db"].mean(),
            "usable_cell": group["case_id"].nunique() >= min_cases_per_cell,
        }
        records.append(record)
    return pd.DataFrame(records).sort_values(["age_bin", "target_agent"])


def main() -> None:
    args = parse_args()
    vitaldb = _import_vitaldb()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = sorted(vitaldb.caseids_des)
    clinical = vitaldb.load_clinical_data(case_ids, ["caseid", "age", "sex"])
    clinical["age_bin"] = clinical["age"].apply(
        lambda age: age_to_bin(age, args.age_min, args.age_max, args.age_bin_width)
    )
    clinical = clinical[clinical["age_bin"].notna()].copy()
    case_ids = clinical["caseid"].astype(int).tolist()
    case_ids = case_ids[args.case_start :]
    if args.max_cases is not None:
        case_ids = case_ids[: args.max_cases]

    clinical_by_case = clinical.set_index("caseid")
    targets = target_grid(args.target_start, args.target_stop, args.target_step)
    rows = []
    errors = []

    for case_idx, case_id in enumerate(case_ids, start=1):
        info = clinical_by_case.loc[case_id]
        print(f"[{case_idx}/{len(case_ids)}] case={case_id} age={info['age']}")
        try:
            tracks = load_numeric_tracks(case_id)
        except Exception as exc:
            errors.append({"case_id": case_id, "stage": "numeric", "error": str(exc)})
            continue

        for target in targets:
            window = best_window_for_target(
                tracks=tracks,
                target=target,
                window_sec=args.window_sec,
                tolerance=args.target_tolerance,
                sqi_min=args.sqi_min,
                min_agent_fraction=args.min_agent_fraction,
                min_valid_fraction=args.min_valid_fraction,
                n2o_max=args.n2o_max,
            )
            if window is None:
                continue

            row = {
                "case_id": case_id,
                "agent": "des",
                "target_agent": target,
                "age": float(info["age"]),
                "sex": info["sex"],
                "age_bin": info["age_bin"],
                "reference_sex": "all",
                **window,
            }
            try:
                row = add_eeg_features(
                    row=row,
                    case_id=case_id,
                    eeg_track=args.eeg_track,
                    sample_rate_hz=args.sample_rate,
                    raw_cache_dir=args.raw_cache_dir,
                )
                rows.append(row)
            except Exception as exc:
                errors.append({"case_id": case_id, "stage": "eeg", "error": str(exc)})

    rows_df = pd.DataFrame(rows)
    rows_path = output_dir / "reference_rows_des_all_v1_no_n2o_qc.csv"
    rows_df.to_csv(rows_path, index=False)

    if not rows_df.empty:
        table = build_reference_table(rows_df, args.min_cases_per_cell)
    else:
        table = pd.DataFrame()
    table_path = output_dir / "reference_table_des_all_v1_no_n2o_by_ET.csv"
    table.to_csv(table_path, index=False)

    pd.DataFrame(errors).to_csv(output_dir / "build_errors.csv", index=False)
    print(f"Wrote {rows_path}")
    print(f"Wrote {table_path}")
    print(f"Rows: {len(rows_df)}; table cells: {len(table)}; errors: {len(errors)}")


if __name__ == "__main__":
    main()
