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

from eeg_spectrum import SpectrogramResult, average_dsa, compute_dsa_1hz
from vitaldb_raw import load_eeg_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build 1-Hz DSA reference templates from VitalDB raw EEG windows "
            "listed in a reference row table."
        )
    )
    parser.add_argument(
        "--row-table",
        default="data/reference_rows_sevo_all_v2_no_n2o_qc.csv",
        help="CSV with case_id, age_bin, target_agent, window_start_sec, window_end_sec.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/dsa_templates_v1",
        help="Directory for per-cell npz templates and template_index.csv.",
    )
    parser.add_argument(
        "--raw-cache-dir",
        default="data/raw_vitaldb_cache",
        help="Directory for cached case/track EEG downloads.",
    )
    parser.add_argument("--track", default="BIS/EEG1_WAV", help="VitalDB EEG waveform track.")
    parser.add_argument("--sample-rate", type=float, default=128.0)
    parser.add_argument("--epoch-sec", type=float, default=4.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--freq-max", type=float, default=40.0)
    parser.add_argument(
        "--max-windows-per-cell",
        type=int,
        default=20,
        help="Limit raw downloads per age/concentration cell. Ignored with --all-windows.",
    )
    parser.add_argument(
        "--all-windows",
        action="store_true",
        help="Use every matching usable window in each age/concentration cell.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Optional development limit for the number of cells to build.",
    )
    parser.add_argument(
        "--age-bin",
        default=None,
        help="Optional single age bin to build, such as 45-49.",
    )
    parser.add_argument(
        "--target-agent",
        type=float,
        default=None,
        help="Optional single ET sevo target to build, such as 2.4.",
    )
    parser.add_argument(
        "--require-usable",
        action="store_true",
        help="Use only rows where usable_for_reference is true.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild templates even when an ok template already exists.",
    )
    return parser.parse_args()


def clean_rows(
    df: pd.DataFrame,
    require_usable: bool,
    age_bin: str | None,
    target_agent: float | None,
) -> pd.DataFrame:
    required = {
        "case_id",
        "age_bin",
        "target_agent",
        "window_start_sec",
        "window_end_sec",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    rows = df.copy()
    rows["age_bin"] = rows["age_bin"].astype(str)
    rows["target_agent"] = rows["target_agent"].astype(float).round(1)
    if require_usable and "usable_for_reference" in rows.columns:
        usable = rows["usable_for_reference"]
        if usable.dtype != bool:
            usable = usable.astype(str).str.lower().eq("true")
        rows = rows[usable]
    if age_bin is not None:
        rows = rows[rows["age_bin"].astype(str) == str(age_bin)]
    if target_agent is not None:
        rows = rows[rows["target_agent"].round(1) == round(float(target_agent), 1)]
    sort_defaults = {
        "window_quality_score": 1.0,
        "mean_sqi": 0.0,
        "artifact_ratio": 0.0,
        "agent_bin_error": 0.0,
    }
    for column, default in sort_defaults.items():
        if column not in rows.columns:
            rows[column] = default
    rows = rows.sort_values(
        [
            "age_bin",
            "target_agent",
            "window_quality_score",
            "mean_sqi",
            "artifact_ratio",
            "agent_bin_error",
            "case_id",
        ],
        ascending=[True, True, False, False, True, True, True],
    )
    return rows


def build_cell_template(
    rows: pd.DataFrame,
    track: str,
    sample_rate_hz: float,
    epoch_sec: float,
    step_sec: float,
    freq_max_hz: float,
    raw_cache_dir: str | Path | None,
) -> tuple[SpectrogramResult, list[str]]:
    results = []
    errors = []

    for row in rows.itertuples(index=False):
        try:
            signal = load_eeg_window(
                case_id=int(row.case_id),
                start_sec=float(row.window_start_sec),
                end_sec=float(row.window_end_sec),
                track_name=track,
                sample_rate_hz=sample_rate_hz,
                cache_dir=raw_cache_dir,
            )
            results.append(
                compute_dsa_1hz(
                    signal,
                    sample_rate_hz=sample_rate_hz,
                    epoch_sec=epoch_sec,
                    step_sec=step_sec,
                    freq_max_hz=freq_max_hz,
                )
            )
        except Exception as exc:
            errors.append(f"case {row.case_id}: {exc}")

    if not results:
        raise RuntimeError("No usable raw EEG windows in this cell. " + " | ".join(errors[:3]))

    return average_dsa(results), errors


def save_template(
    output_path: Path,
    result: SpectrogramResult,
    age_bin: str,
    target_agent: float,
    n_windows_requested: int,
    n_errors: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        time_sec=result.time_sec.astype(np.float32),
        freq_bin_start_hz=result.freq_bin_start_hz.astype(np.float32),
        freq_bin_end_hz=result.freq_bin_end_hz.astype(np.float32),
        freq_center_hz=result.freq_center_hz.astype(np.float32),
        power_db=result.power_db.astype(np.float32),
        age_bin=str(age_bin),
        target_agent=float(target_agent),
        n_windows_requested=int(n_windows_requested),
        n_errors=int(n_errors),
        valid_epoch_fraction=float(result.valid_epoch_fraction),
    )


def merge_template_index(index_path: Path, new_rows: list[dict]) -> pd.DataFrame:
    new_index = pd.DataFrame(new_rows)
    if index_path.exists():
        existing = pd.read_csv(index_path)
        existing["age_bin"] = existing["age_bin"].astype(str)
        existing["target_agent"] = existing["target_agent"].astype(float).round(1)
        new_index["age_bin"] = new_index["age_bin"].astype(str)
        new_index["target_agent"] = new_index["target_agent"].astype(float).round(1)

        new_keys = set(zip(new_index["age_bin"], new_index["target_agent"]))
        existing = existing[
            ~existing.apply(
                lambda row: (str(row["age_bin"]), round(float(row["target_agent"]), 1))
                in new_keys,
                axis=1,
            )
        ]
        merged = pd.concat([existing, new_index], ignore_index=True)
    else:
        merged = new_index

    return merged.sort_values(["age_bin", "target_agent"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    rows = clean_rows(
        pd.read_csv(args.row_table),
        require_usable=args.require_usable,
        age_bin=args.age_bin,
        target_agent=args.target_agent,
    )
    if rows.empty:
        raise ValueError("No rows matched the requested filters.")
    output_dir = Path(args.output_dir)
    index_rows = []
    index_path = output_dir / "template_index.csv"
    existing_ok = set()
    if index_path.exists() and not args.overwrite:
        existing_index = pd.read_csv(index_path)
        if not existing_index.empty:
            existing_index["age_bin"] = existing_index["age_bin"].astype(str)
            existing_index["target_agent"] = (
                existing_index["target_agent"].astype(float).round(1)
            )
            existing_ok = set(
                zip(
                    existing_index[
                        existing_index["status"].astype(str).str.lower() == "ok"
                    ]["age_bin"],
                    existing_index[
                        existing_index["status"].astype(str).str.lower() == "ok"
                    ]["target_agent"],
                )
            )

    grouped = rows.groupby(["age_bin", "target_agent"], sort=True)
    for cell_idx, ((age_bin, target_agent), cell_rows) in enumerate(grouped, start=1):
        if args.max_cells is not None and cell_idx > args.max_cells:
            break

        selected_rows = (
            cell_rows
            if args.all_windows
            else cell_rows.head(args.max_windows_per_cell)
        )
        safe_age_bin = str(age_bin).replace("-", "_")
        template_name = f"age_{safe_age_bin}_et_{float(target_agent):.1f}.npz"
        output_path = output_dir / template_name
        cell_key = (str(age_bin), round(float(target_agent), 1))

        if cell_key in existing_ok and output_path.exists() and not args.overwrite:
            print(
                f"[{cell_idx}/{len(grouped)}] age={age_bin} et={target_agent:.1f} "
                "already exists, skipping"
            )
            continue

        print(
            f"[{cell_idx}/{len(grouped)}] age={age_bin} et={target_agent:.1f} "
            f"windows={len(selected_rows)}"
        )
        try:
            template, errors = build_cell_template(
                selected_rows,
                track=args.track,
                sample_rate_hz=args.sample_rate,
                epoch_sec=args.epoch_sec,
                step_sec=args.step_sec,
                freq_max_hz=args.freq_max,
                raw_cache_dir=args.raw_cache_dir,
            )
            save_template(
                output_path=output_path,
                result=template,
                age_bin=age_bin,
                target_agent=target_agent,
                n_windows_requested=len(selected_rows),
                n_errors=len(errors),
            )
            status = "ok"
            error_preview = " | ".join(errors[:3])
        except Exception as exc:
            status = "failed"
            error_preview = str(exc)

        index_rows.append(
            {
                "age_bin": age_bin,
                "target_agent": target_agent,
                "template_path": str(output_path),
                "n_windows_requested": len(selected_rows),
                "status": status,
                "error_preview": error_preview,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if index_rows:
        merged_index = merge_template_index(index_path, index_rows)
        merged_index.to_csv(index_path, index=False)
        print(f"Wrote {index_path}")
    else:
        print("No new templates built.")


if __name__ == "__main__":
    main()
