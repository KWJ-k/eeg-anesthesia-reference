from __future__ import annotations

import matplotlib
import pandas as pd


matplotlib.use("Agg")

import matplotlib.pyplot as plt


SUMMARY_DPI = 180
TITLE_SIZE = 15
LABEL_SIZE = 13
TICK_SIZE = 12
LEGEND_SIZE = 11
LINE_WIDTH = 2.1
MARKER_SIZE = 5.2


def _weighted_window_smooth(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    half_window: float,
) -> pd.DataFrame:
    smooth_df = df[[x_col, y_col, "n_cases_clean"]].dropna().copy()
    if smooth_df.empty:
        return smooth_df

    smooth_df[x_col] = smooth_df[x_col].astype(float)
    smooth_df[y_col] = smooth_df[y_col].astype(float)
    smooth_df["n_cases_clean"] = smooth_df["n_cases_clean"].clip(lower=1).astype(float)

    rows = []
    for center_x in smooth_df[x_col]:
        neighbors = smooth_df[(smooth_df[x_col] - center_x).abs() <= half_window]
        weights = neighbors["n_cases_clean"]
        weighted_y = float((neighbors[y_col] * weights).sum() / weights.sum())
        rows.append({x_col: float(center_x), y_col: weighted_y})

    return pd.DataFrame(rows).drop_duplicates(subset=[x_col]).sort_values(x_col)


def plot_reference_summary(
    row: pd.Series,
    age_trend: pd.DataFrame,
    agent_age_trend: pd.DataFrame,
    agent_label: str = "ET sevo",
    age_trend_title: str | None = None,
    concentration_x_col: str = "target_agent",
    concentration_x_label: str | None = None,
    selected_concentration_x: float | None = None,
    concentration_smooth_window: float | None = None,
    layout: str = "horizontal",
):
    labels = ["Delta", "Theta", "Alpha", "Beta"]
    values = [
        row["delta_power_db"],
        row["theta_power_db"],
        row["alpha_power_db"],
        row["beta_power_db"],
    ]

    if layout == "vertical":
        fig, axes = plt.subplots(3, 1, figsize=(5.7, 11.7), dpi=SUMMARY_DPI)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=SUMMARY_DPI)

    axes[0].bar(
        labels,
        values,
        color=["#4C78A8", "#F58518", "#E45756", "#72B7B2"],
    )
    axes[0].set_ylabel("Power (dB)", fontsize=LABEL_SIZE)
    axes[0].set_title("Expected Band Power", fontsize=TITLE_SIZE)
    axes[0].set_ylim(-10, 25)
    axes[0].axhline(0, color="black", linewidth=1.0)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="both", labelsize=TICK_SIZE)

    if not agent_age_trend.empty:
        bis_line = axes[1].plot(
            agent_age_trend["age_midpoint"],
            agent_age_trend["mean_bis"],
            marker="o",
            markersize=MARKER_SIZE,
            color="purple",
            linewidth=LINE_WIDTH,
            label="BIS",
        )[0]
        axes[1].set_ylim(0, 100)
        axes[1].set_xlabel("Age (years)", fontsize=LABEL_SIZE)
        axes[1].set_ylabel("BIS", fontsize=LABEL_SIZE)
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis="both", labelsize=TICK_SIZE)
        axes[1].axvline(
            sum(int(part) for part in str(row["age_bin"]).split("-")) / 2,
            color="black",
            linestyle="--",
            alpha=0.6,
            linewidth=1.4,
        )

        ax2 = axes[1].twinx()
        sef_line = ax2.plot(
            agent_age_trend["age_midpoint"],
            agent_age_trend["mean_sef"],
            marker="o",
            markersize=MARKER_SIZE,
            color="tab:blue",
            linewidth=LINE_WIDTH,
            label="SEF",
        )[0]
        ax2.set_ylim(0, 25)
        ax2.set_ylabel("SEF (Hz)", color="tab:blue", fontsize=LABEL_SIZE)
        ax2.tick_params(axis="y", labelcolor="tab:blue", labelsize=TICK_SIZE)

        axes[1].legend(
            [bis_line, sef_line],
            ["BIS", "SEF"],
            loc="upper left",
            fontsize=LEGEND_SIZE,
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "No age trend data",
            ha="center",
            va="center",
            fontsize=LABEL_SIZE,
            transform=axes[1].transAxes,
        )
        axes[1].set_axis_off()
    if age_trend_title is None:
        age_trend_title = f"Age Trend At {agent_label} {row['target_agent']:.1f}%"
    axes[1].set_title(age_trend_title, fontsize=TITLE_SIZE)

    plot_age_trend = age_trend.sort_values(concentration_x_col)
    if concentration_x_label is None:
        concentration_x_label = f"{agent_label} (%)"
    if selected_concentration_x is None:
        selected_concentration_x = float(row[concentration_x_col])

    use_smoothing = (
        concentration_smooth_window is not None
        and concentration_smooth_window > 0
        and "n_cases_clean" in plot_age_trend.columns
        and len(plot_age_trend) >= 3
    )

    raw_alpha = 0.25 if use_smoothing else 1.0
    raw_linewidth = 1.2 if use_smoothing else LINE_WIDTH
    raw_label = "_nolegend_" if use_smoothing else "BIS"

    bis_line = axes[2].plot(
        plot_age_trend[concentration_x_col],
        plot_age_trend["mean_bis"],
        marker="o",
        markersize=MARKER_SIZE,
        color="purple",
        alpha=raw_alpha,
        linewidth=raw_linewidth,
        label=raw_label,
    )[0]
    axes[2].set_ylim(0, 100)
    axes[2].set_xlabel(concentration_x_label, fontsize=LABEL_SIZE)
    axes[2].set_ylabel("BIS", fontsize=LABEL_SIZE)
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis="both", labelsize=TICK_SIZE)

    ax3 = axes[2].twinx()
    raw_sef_label = "_nolegend_" if use_smoothing else "SEF"
    sef_line = ax3.plot(
        plot_age_trend[concentration_x_col],
        plot_age_trend["mean_sef"],
        marker="o",
        markersize=MARKER_SIZE,
        color="tab:blue",
        alpha=raw_alpha,
        linewidth=raw_linewidth,
        label=raw_sef_label,
    )[0]
    ax3.set_ylim(0, 25)
    ax3.set_ylabel("SEF (Hz)", color="tab:blue", fontsize=LABEL_SIZE)
    ax3.tick_params(axis="y", labelcolor="tab:blue", labelsize=TICK_SIZE)

    if use_smoothing:
        smoothed_bis = _weighted_window_smooth(
            plot_age_trend,
            concentration_x_col,
            "mean_bis",
            float(concentration_smooth_window),
        )
        smoothed_sef = _weighted_window_smooth(
            plot_age_trend,
            concentration_x_col,
            "mean_sef",
            float(concentration_smooth_window),
        )
        bis_line = axes[2].plot(
            smoothed_bis[concentration_x_col],
            smoothed_bis["mean_bis"],
            marker="o",
            markersize=MARKER_SIZE,
            color="purple",
            linewidth=2.7,
            label="BIS",
        )[0]
        sef_line = ax3.plot(
            smoothed_sef[concentration_x_col],
            smoothed_sef["mean_sef"],
            marker="o",
            markersize=MARKER_SIZE,
            color="tab:blue",
            linewidth=2.7,
            label="SEF",
        )[0]

    axes[2].legend(
        [bis_line, sef_line],
        ["BIS", "SEF"],
        loc="upper right",
        fontsize=LEGEND_SIZE,
    )

    axes[2].axvline(
        selected_concentration_x,
        color="black",
        linestyle="--",
        alpha=0.6,
        linewidth=1.4,
    )
    axes[2].set_title(f"Trend In Age Bin {row['age_bin']}", fontsize=TITLE_SIZE)

    fig.tight_layout()
    return fig
