from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_TRACKS = [
    "Orchestra/PPF20_CE",
    "Orchestra/PPF20_CP",
    "Orchestra/PPF20_RATE",
    "Orchestra/RFTN20_CE",
    "Orchestra/RFTN20_CP",
    "Orchestra/RFTN20_RATE",
    "BIS/BIS",
    "BIS/SEF",
    "BIS/SQI",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a lightweight TIVA propofol Ce age-BIS MVP dataset."
    )
    parser.add_argument("--output-dir", default="data/tiva_propofol_ce3_mvp_v1")
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--target-ppf-ce", type=float, default=3.0)
    parser.add_argument("--target-tolerance", type=float, default=0.25)
    parser.add_argument("--window-sec", type=int, default=120)
    parser.add_argument("--age-min", type=int, default=20)
    parser.add_argument("--age-max", type=int, default=84)
    parser.add_argument("--age-bin-width", type=int, default=5)
    parser.add_argument("--sqi-min", type=float, default=80.0)
    parser.add_argument("--min-target-fraction", type=float, default=0.9)
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-cases-per-age-bin", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=20260526)
    return parser.parse_args()


def _import_vitaldb():
    try:
        import vitaldb  # type: ignore
    except ImportError as exc:
        raise ImportError("Install vitaldb with `pip install -r requirements.txt`.") from exc
    return vitaldb


def age_to_bin(age: float, age_min: int, age_max: int, width: int) -> str | None:
    if pd.isna(age) or age < age_min or age > age_max:
        return None
    left = age_min + int((age - age_min) // width) * width
    right = left + width - 1
    return f"{left}-{right}"


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=window).mean().to_numpy()


def load_numeric_tracks(case_id: int) -> pd.DataFrame:
    vitaldb = _import_vitaldb()
    values = np.asarray(vitaldb.load_case(case_id, NUMERIC_TRACKS, interval=1), dtype=float)
    if values.ndim != 2 or values.shape[1] != len(NUMERIC_TRACKS):
        raise ValueError(f"Unexpected track matrix shape: {values.shape}")

    tracks = pd.DataFrame(
        values,
        columns=[
            "ppf_ce",
            "ppf_cp",
            "ppf_rate",
            "rft_ce",
            "rft_cp",
            "rft_rate",
            "bis",
            "sef",
            "sqi",
        ],
    )
    tracks = tracks.ffill()
    return tracks


def best_window_for_ppf_ce(
    tracks: pd.DataFrame,
    target: float,
    tolerance: float,
    window_sec: int,
    sqi_min: float,
    min_target_fraction: float,
    min_valid_fraction: float,
) -> dict | None:
    ppf_ce = tracks["ppf_ce"].to_numpy(dtype=float)
    ppf_cp = tracks["ppf_cp"].to_numpy(dtype=float)
    ppf_rate = tracks["ppf_rate"].to_numpy(dtype=float)
    rft_ce = tracks["rft_ce"].to_numpy(dtype=float)
    rft_cp = tracks["rft_cp"].to_numpy(dtype=float)
    rft_rate = tracks["rft_rate"].to_numpy(dtype=float)
    bis = tracks["bis"].to_numpy(dtype=float)
    sef = tracks["sef"].to_numpy(dtype=float)
    sqi = tracks["sqi"].to_numpy(dtype=float)

    finite = (
        np.isfinite(ppf_ce)
        & np.isfinite(bis)
        & np.isfinite(sef)
        & np.isfinite(sqi)
    )
    near_target = np.isfinite(ppf_ce) & (np.abs(ppf_ce - target) <= tolerance)
    good_sqi = np.isfinite(sqi) & (sqi >= sqi_min)

    valid_fraction = rolling_mean(finite.astype(float), window_sec)
    target_fraction = rolling_mean(near_target.astype(float), window_sec)
    sqi_fraction = rolling_mean(good_sqi.astype(float), window_sec)

    mean_ppf_ce = rolling_mean(ppf_ce, window_sec)
    mean_ppf_cp = rolling_mean(ppf_cp, window_sec)
    mean_ppf_rate = rolling_mean(ppf_rate, window_sec)
    mean_rft_ce = rolling_mean(rft_ce, window_sec)
    mean_rft_cp = rolling_mean(rft_cp, window_sec)
    mean_rft_rate = rolling_mean(rft_rate, window_sec)
    mean_bis = rolling_mean(bis, window_sec)
    mean_sef = rolling_mean(sef, window_sec)
    mean_sqi = rolling_mean(sqi, window_sec)
    ppf_ce_error = np.abs(mean_ppf_ce - target)

    candidate = (
        (valid_fraction >= min_valid_fraction)
        & (target_fraction >= min_target_fraction)
        & (sqi_fraction >= min_valid_fraction)
        & np.isfinite(ppf_ce_error)
    )
    if not candidate.any():
        return None

    score = (
        target_fraction * 4
        + valid_fraction
        + sqi_fraction
        + mean_sqi / 100
        - ppf_ce_error
    )
    score[~candidate] = -np.inf
    end_idx = int(np.nanargmax(score))
    start_idx = end_idx - window_sec + 1
    if start_idx < 0:
        return None

    return {
        "window_start_sec": float(start_idx),
        "window_end_sec": float(end_idx + 1),
        "mean_ppf_ce": float(mean_ppf_ce[end_idx]),
        "mean_ppf_cp": float(mean_ppf_cp[end_idx]),
        "mean_ppf_rate": float(mean_ppf_rate[end_idx]),
        "mean_rft_ce": float(mean_rft_ce[end_idx]),
        "mean_rft_cp": float(mean_rft_cp[end_idx]),
        "mean_rft_rate": float(mean_rft_rate[end_idx]),
        "mean_bis": float(mean_bis[end_idx]),
        "mean_sef": float(mean_sef[end_idx]),
        "mean_sqi": float(mean_sqi[end_idx]),
        "window_quality_score": float(score[end_idx]),
        "window_target_fraction": float(target_fraction[end_idx]),
        "window_sqi_fraction": float(sqi_fraction[end_idx]),
        "window_valid_fraction": float(valid_fraction[end_idx]),
        "ppf_ce_error": float(ppf_ce_error[end_idx]),
    }


def build_age_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouped = rows.groupby("age_bin", sort=False)
    for age_bin, group in grouped:
        left, right = str(age_bin).split("-")
        records.append(
            {
                "age_bin": age_bin,
                "age_midpoint": (int(left) + int(right)) / 2,
                "n_cases": group["case_id"].nunique(),
                "mean_age": group["age"].mean(),
                "mean_ppf_ce": group["mean_ppf_ce"].mean(),
                "sd_ppf_ce": group["mean_ppf_ce"].std(),
                "mean_rft_ce": group["mean_rft_ce"].mean(),
                "sd_rft_ce": group["mean_rft_ce"].std(),
                "mean_bis": group["mean_bis"].mean(),
                "sd_bis": group["mean_bis"].std(),
                "sem_bis": group["mean_bis"].std() / np.sqrt(group["case_id"].nunique()),
                "mean_sef": group["mean_sef"].mean(),
                "mean_sqi": group["mean_sqi"].mean(),
                "usable_cell": group["case_id"].nunique() >= 3,
            }
        )
    return pd.DataFrame(records).sort_values("age_midpoint")


def save_figure(rows: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit_df = rows.dropna(subset=["age", "mean_bis"]).copy()
    x = fit_df["age"].astype(float).to_numpy()
    y = fit_df["mean_bis"].astype(float).to_numpy()
    slope, intercept = np.polyfit(x, y, deg=1)
    line_x = np.linspace(x.min(), x.max(), 200)
    line_y = slope * line_x + intercept
    ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 7.2),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 2.2], "hspace": 0.12},
    )
    ax_ce, ax_bis = axes

    ax_ce.scatter(
        rows["age"],
        rows["mean_ppf_ce"],
        s=18,
        color="#4C78A8",
        alpha=0.45,
        linewidth=0,
    )
    ax_ce.axhline(3.0, color="#1F4E79", linewidth=1.8, linestyle="--")
    ax_ce.axhspan(2.75, 3.25, color="#DDEBF7", alpha=0.8, linewidth=0)
    ax_ce.set_ylabel("Propofol Ce")
    ax_ce.set_ylim(2.65, 3.35)
    ax_ce.grid(axis="y", alpha=0.25)
    ax_ce.text(
        0.01,
        0.92,
        "Target propofol Ce 3.0 +/- 0.25",
        transform=ax_ce.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#1F4E79",
    )

    scatter = ax_bis.scatter(
        rows["age"],
        rows["mean_bis"],
        c=rows["mean_rft_ce"],
        cmap="viridis",
        s=26,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
        label="Case-level window",
    )
    ax_bis.plot(line_x, line_y, color="#1F4E79", linewidth=2.6, label="Linear fit")
    ax_bis.errorbar(
        summary["age_midpoint"],
        summary["mean_bis"],
        yerr=summary["sem_bis"],
        fmt="o",
        color="#D62728",
        ecolor="#D62728",
        markersize=6,
        capsize=3,
        label="Age-bin mean +/- SEM",
    )
    ax_bis.set_xlabel("Age (years)")
    ax_bis.set_ylabel("BIS")
    ax_bis.set_ylim(15, 75)
    ax_bis.grid(True, alpha=0.24)
    ax_bis.legend(frameon=False, loc="upper left")
    ax_bis.text(
        0.02,
        0.06,
        f"BIS slope {slope * 10:+.2f} / 10 yr; R2 = {r_squared:.2f}; cases = {len(rows)}",
        transform=ax_bis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "alpha": 0.92,
        },
    )
    colorbar = fig.colorbar(scatter, ax=axes, pad=0.018, fraction=0.045)
    colorbar.set_label("Remifentanil Ce")
    fig.suptitle("TIVA: BIS by Age at Propofol Ce 3.0", fontsize=15, fontweight="bold")
    fig.text(
        0.01,
        0.012,
        "MVP analysis: one stable 120-sec window per case; high-SQI windows only; color indicates mean remifentanil Ce.",
        fontsize=8.2,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 0.94, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    return {
        "slope_per_year": float(slope),
        "slope_per_10_years": float(slope * 10.0),
        "intercept": float(intercept),
        "r_squared": r_squared,
    }


def main() -> None:
    args = parse_args()
    vitaldb = _import_vitaldb()
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    case_ids = sorted(vitaldb.caseids_tiva)

    clinical = vitaldb.load_clinical_data(case_ids, ["caseid", "age", "sex"])
    clinical["age_bin"] = clinical["age"].apply(
        lambda age: age_to_bin(age, args.age_min, args.age_max, args.age_bin_width)
    )
    clinical = clinical[clinical["age_bin"].notna()].copy()

    if args.max_cases_per_age_bin is not None:
        sampled_groups = []
        for _, group in clinical.groupby("age_bin", sort=True):
            sampled_groups.append(
                group.sample(
                    n=min(len(group), args.max_cases_per_age_bin),
                    random_state=args.random_seed,
                )
            )
        clinical = pd.concat(sampled_groups, ignore_index=True).sort_values(
            ["age_bin", "caseid"]
        )
    elif args.max_cases is not None:
        clinical = clinical.head(args.max_cases).copy()

    clinical_by_case = clinical.set_index("caseid")
    case_ids = clinical["caseid"].astype(int).tolist()

    rows = []
    errors = []
    for case_idx, case_id in enumerate(case_ids, start=1):
        info = clinical_by_case.loc[case_id]
        print(f"[{case_idx}/{len(case_ids)}] case={case_id} age={info['age']}")
        try:
            tracks = load_numeric_tracks(case_id)
            window = best_window_for_ppf_ce(
                tracks=tracks,
                target=args.target_ppf_ce,
                tolerance=args.target_tolerance,
                window_sec=args.window_sec,
                sqi_min=args.sqi_min,
                min_target_fraction=args.min_target_fraction,
                min_valid_fraction=args.min_valid_fraction,
            )
        except Exception as exc:
            errors.append({"case_id": case_id, "stage": "load_or_window", "error": str(exc)})
            continue

        if window is None:
            continue

        rows.append(
            {
                "case_id": case_id,
                "age": float(info["age"]),
                "sex": info["sex"],
                "age_bin": info["age_bin"],
                "target_ppf_ce": args.target_ppf_ce,
                **window,
            }
        )

    rows_df = pd.DataFrame(rows)
    rows_path = output_dir / "tiva_propofol_ce3_case_windows_v1.csv"
    rows_df.to_csv(rows_path, index=False)

    if rows_df.empty:
        summary = pd.DataFrame()
        fit = {}
    else:
        summary = build_age_summary(rows_df)
        fit = save_figure(
            rows_df,
            summary,
            figure_dir / "tiva_propofol_ce3_age_bis_mvp_v1.png",
        )

    summary_path = output_dir / "tiva_propofol_ce3_age_summary_v1.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(errors).to_csv(output_dir / "build_errors.csv", index=False)
    pd.DataFrame([fit]).to_csv(output_dir / "tiva_propofol_ce3_linear_fit_v1.csv", index=False)

    print(f"Wrote {rows_path}")
    print(f"Wrote {summary_path}")
    print(f"Rows: {len(rows_df)}; age bins: {len(summary)}; errors: {len(errors)}")
    if fit:
        print(
            "Fit: "
            f"slope={fit['slope_per_10_years']:+.3f} BIS/10yr, "
            f"R2={fit['r_squared']:.3f}"
        )


if __name__ == "__main__":
    main()
