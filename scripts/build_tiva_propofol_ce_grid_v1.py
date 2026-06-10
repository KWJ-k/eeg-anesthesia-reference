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
        description="Build a TIVA propofol Ce grid dataset for age-BIS analysis."
    )
    parser.add_argument("--output-dir", default="data/tiva_propofol_ce_grid_v1")
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--target-ppf-ce-list", default="2.0,3.0,4.0")
    parser.add_argument("--target-tolerance", type=float, default=0.25)
    parser.add_argument("--window-sec", type=int, default=120)
    parser.add_argument("--age-min", type=int, default=20)
    parser.add_argument("--age-max", type=int, default=84)
    parser.add_argument("--age-bin-width", type=int, default=5)
    parser.add_argument("--sqi-min", type=float, default=80.0)
    parser.add_argument("--min-target-fraction", type=float, default=0.9)
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    parser.add_argument("--exclude-edge-min", type=float, default=0.0)
    parser.add_argument("--max-ppf-ce-range", type=float, default=None)
    parser.add_argument("--min-ppf-rate", type=float, default=None)
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


def parse_targets(raw_targets: str) -> list[float]:
    targets = [float(item.strip()) for item in raw_targets.split(",") if item.strip()]
    if not targets:
        raise ValueError("At least one propofol Ce target is required.")
    return sorted(set(targets))


def age_to_bin(age: float, age_min: int, age_max: int, width: int) -> str | None:
    if pd.isna(age) or age < age_min or age > age_max:
        return None
    left = age_min + int((age - age_min) // width) * width
    right = left + width - 1
    return f"{left}-{right}"


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=window).mean().to_numpy()


def rolling_range(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values)
    rolling = series.rolling(window, min_periods=window)
    return (rolling.max() - rolling.min()).to_numpy()


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
    return tracks.ffill()


def best_window_for_ppf_ce(
    tracks: pd.DataFrame,
    target: float,
    tolerance: float,
    window_sec: int,
    sqi_min: float,
    min_target_fraction: float,
    min_valid_fraction: float,
    exclude_edge_sec: int,
    max_ppf_ce_range: float | None,
    min_ppf_rate: float | None,
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

    finite = np.isfinite(ppf_ce) & np.isfinite(bis) & np.isfinite(sef) & np.isfinite(sqi)
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
    ppf_ce_range = rolling_range(ppf_ce, window_sec)

    end_indices = np.arange(len(tracks), dtype=int)
    window_starts = end_indices - window_sec + 1
    within_case_center = (
        (window_starts >= exclude_edge_sec)
        & (end_indices <= len(tracks) - exclude_edge_sec - 1)
    )

    candidate = (
        (valid_fraction >= min_valid_fraction)
        & (target_fraction >= min_target_fraction)
        & (sqi_fraction >= min_valid_fraction)
        & np.isfinite(ppf_ce_error)
        & within_case_center
    )
    if max_ppf_ce_range is not None:
        candidate &= np.isfinite(ppf_ce_range) & (ppf_ce_range <= max_ppf_ce_range)
    if min_ppf_rate is not None:
        candidate &= np.isfinite(mean_ppf_rate) & (mean_ppf_rate >= min_ppf_rate)
    if not candidate.any():
        return None

    score = target_fraction * 4 + valid_fraction + sqi_fraction + mean_sqi / 100 - ppf_ce_error
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
        "ppf_ce_range": float(ppf_ce_range[end_idx]),
        "window_position_fraction": float((start_idx + end_idx + 1) / 2 / len(tracks)),
    }


def build_age_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (target, age_bin), group in rows.groupby(["target_ppf_ce", "age_bin"], sort=True):
        left, right = str(age_bin).split("-")
        n_cases = group["case_id"].nunique()
        sd_bis = group["mean_bis"].std()
        records.append(
            {
                "target_ppf_ce": target,
                "age_bin": age_bin,
                "age_midpoint": (int(left) + int(right)) / 2,
                "n_cases": n_cases,
                "mean_age": group["age"].mean(),
                "mean_ppf_ce": group["mean_ppf_ce"].mean(),
                "sd_ppf_ce": group["mean_ppf_ce"].std(),
                "mean_ppf_ce_range": group["ppf_ce_range"].mean(),
                "mean_rft_ce": group["mean_rft_ce"].mean(),
                "sd_rft_ce": group["mean_rft_ce"].std(),
                "mean_bis": group["mean_bis"].mean(),
                "sd_bis": sd_bis,
                "sem_bis": sd_bis / np.sqrt(n_cases) if n_cases else np.nan,
                "mean_sef": group["mean_sef"].mean(),
                "mean_sqi": group["mean_sqi"].mean(),
                "usable_cell": n_cases >= 3,
            }
        )
    return pd.DataFrame(records).sort_values(["target_ppf_ce", "age_midpoint"])


def fit_linear(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return {
            "slope_per_year": np.nan,
            "slope_per_10_years": np.nan,
            "intercept": np.nan,
            "r_squared": np.nan,
            "n": int(len(x)),
        }
    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "slope_per_year": float(slope),
        "slope_per_10_years": float(slope * 10.0),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "n": int(len(x)),
    }


def fit_multivariable(df: pd.DataFrame, columns: list[str]) -> dict:
    fit_df = df.dropna(subset=["mean_bis", *columns]).copy()
    if len(fit_df) < len(columns) + 2:
        return {"n": int(len(fit_df)), "r_squared": np.nan}
    x_cols = [np.ones(len(fit_df))]
    x_cols.extend(fit_df[col].astype(float).to_numpy() for col in columns)
    x = np.column_stack(x_cols)
    y = fit_df["mean_bis"].astype(float).to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    predicted = x @ coef
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out = {
        "n": int(len(fit_df)),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "intercept": float(coef[0]),
    }
    for idx, col in enumerate(columns, start=1):
        out[f"{col}_coef"] = float(coef[idx])
        if col == "age":
            out["age_slope_per_10_years"] = float(coef[idx] * 10.0)
    return out


def build_fit_table(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for target, group in rows.groupby("target_ppf_ce", sort=True):
        simple = fit_linear(group["age"].to_numpy(), group["mean_bis"].to_numpy())
        adjusted = fit_multivariable(group, ["age", "mean_rft_ce", "mean_ppf_ce"])
        records.append(
            {
                "target_ppf_ce": target,
                "simple_age_slope_per_10yr": simple["slope_per_10_years"],
                "simple_r_squared": simple["r_squared"],
                "simple_n": simple["n"],
                "adjusted_age_slope_per_10yr": adjusted.get("age_slope_per_10_years", np.nan),
                "rft_ce_coef": adjusted.get("mean_rft_ce_coef", np.nan),
                "ppf_ce_coef": adjusted.get("mean_ppf_ce_coef", np.nan),
                "adjusted_r_squared": adjusted.get("r_squared", np.nan),
                "adjusted_n": adjusted.get("n", 0),
            }
        )
    return pd.DataFrame(records)


def save_grid_figure(rows: pd.DataFrame, summary: pd.DataFrame, fits: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {
        2.0: "#4C78A8",
        3.0: "#D62728",
        4.0: "#7A5195",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)
    for target, group in summary.groupby("target_ppf_ce", sort=True):
        color = colors.get(float(target), None)
        ax.errorbar(
            group["age_midpoint"],
            group["mean_bis"],
            yerr=group["sem_bis"],
            marker="o",
            markersize=5,
            linewidth=2.0,
            capsize=3,
            color=color,
            label=f"Propofol Ce {target:g}",
        )
        fit_group = rows[rows["target_ppf_ce"] == target].dropna(subset=["age", "mean_bis"])
        if len(fit_group) >= 3:
            slope, intercept = np.polyfit(fit_group["age"], fit_group["mean_bis"], deg=1)
            line_x = np.linspace(fit_group["age"].min(), fit_group["age"].max(), 100)
            ax.plot(line_x, slope * line_x + intercept, color=color, alpha=0.35, linewidth=1.5)

    ax.set_title("TIVA: BIS by Age Across Propofol Ce Targets", fontsize=13, fontweight="bold")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("BIS")
    ax.set_ylim(15, 75)
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False, loc="upper left")

    if not fits.empty:
        text_lines = []
        for _, row in fits.iterrows():
            text_lines.append(
                f"Ce {row['target_ppf_ce']:g}: simple {row['simple_age_slope_per_10yr']:+.2f}, "
                f"remi-adj {row['adjusted_age_slope_per_10yr']:+.2f}/10yr, n={int(row['simple_n'])}"
            )
        ax.text(
            0.02,
            0.04,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#CCCCCC",
                "alpha": 0.92,
            },
        )

    fig.text(
        0.01,
        0.012,
        "One stable 120-sec high-SQI window per case and target. Faint lines are unadjusted age fits; "
        "remi-adj slope uses BIS ~ age + remifentanil Ce + observed propofol Ce within each target.",
        fontsize=8,
        color="#444444",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_remi_figure(rows: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = rows.dropna(subset=["target_ppf_ce", "mean_rft_ce", "mean_bis"]).copy()
    if plot_rows.empty:
        return

    targets = sorted(plot_rows["target_ppf_ce"].unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(3.6 * len(targets), 3.6), dpi=300, sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        group = plot_rows[plot_rows["target_ppf_ce"] == target]
        scatter = ax.scatter(
            group["mean_rft_ce"],
            group["mean_bis"],
            c=group["age"],
            cmap="viridis",
            s=22,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.3,
        )
        if len(group) >= 3:
            slope, intercept = np.polyfit(group["mean_rft_ce"], group["mean_bis"], deg=1)
            line_x = np.linspace(group["mean_rft_ce"].min(), group["mean_rft_ce"].max(), 100)
            ax.plot(line_x, slope * line_x + intercept, color="#1F4E79", linewidth=1.8)
            ax.text(
                0.03,
                0.06,
                f"slope {slope:+.2f} BIS/Ce",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9},
            )
        ax.set_title(f"Propofol Ce {target:g}")
        ax.set_xlabel("Remifentanil Ce")
        ax.grid(True, alpha=0.24)
    axes[0].set_ylabel("BIS")
    fig.colorbar(scatter, ax=axes, pad=0.018, fraction=0.045, label="Age")
    fig.suptitle("BIS vs Remifentanil Ce by Propofol Target", fontsize=13, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 0.96, 0.94))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    vitaldb = _import_vitaldb()
    targets = parse_targets(args.target_ppf_ce_list)
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
        clinical = pd.concat(sampled_groups, ignore_index=True).sort_values(["age_bin", "caseid"])
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
        except Exception as exc:
            errors.append({"case_id": case_id, "stage": "load", "error": str(exc)})
            continue

        for target in targets:
            try:
                window = best_window_for_ppf_ce(
                    tracks=tracks,
                    target=target,
                    tolerance=args.target_tolerance,
                    window_sec=args.window_sec,
                    sqi_min=args.sqi_min,
                    min_target_fraction=args.min_target_fraction,
                    min_valid_fraction=args.min_valid_fraction,
                    exclude_edge_sec=int(round(args.exclude_edge_min * 60)),
                    max_ppf_ce_range=args.max_ppf_ce_range,
                    min_ppf_rate=args.min_ppf_rate,
                )
            except Exception as exc:
                errors.append(
                    {
                        "case_id": case_id,
                        "target_ppf_ce": target,
                        "stage": "window",
                        "error": str(exc),
                    }
                )
                continue

            if window is None:
                continue

            rows.append(
                {
                    "case_id": case_id,
                    "age": float(info["age"]),
                    "sex": info["sex"],
                    "age_bin": info["age_bin"],
                    "target_ppf_ce": target,
                    **window,
                }
            )

    rows_df = pd.DataFrame(rows)
    rows_path = output_dir / "tiva_propofol_ce_grid_case_windows_v1.csv"
    rows_df.to_csv(rows_path, index=False)

    if rows_df.empty:
        summary = pd.DataFrame()
        fits = pd.DataFrame()
    else:
        summary = build_age_summary(rows_df)
        fits = build_fit_table(rows_df)
        save_grid_figure(
            rows_df,
            summary,
            fits,
            figure_dir / "tiva_propofol_ce_grid_age_bis_v1.png",
        )
        save_remi_figure(
            rows_df,
            figure_dir / "tiva_propofol_ce_grid_remi_bis_v1.png",
        )

    summary_path = output_dir / "tiva_propofol_ce_grid_age_summary_v1.csv"
    fit_path = output_dir / "tiva_propofol_ce_grid_fit_summary_v1.csv"
    errors_path = output_dir / "build_errors.csv"
    summary.to_csv(summary_path, index=False)
    fits.to_csv(fit_path, index=False)
    pd.DataFrame(errors).to_csv(errors_path, index=False)

    print(f"Wrote {rows_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {fit_path}")
    print(f"Rows: {len(rows_df)}; age-target cells: {len(summary)}; errors: {len(errors)}")
    if not fits.empty:
        print(fits.to_string(index=False))


if __name__ == "__main__":
    main()
