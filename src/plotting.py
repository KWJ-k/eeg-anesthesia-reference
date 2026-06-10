from __future__ import annotations

import matplotlib
import pandas as pd


matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_reference_summary(
    row: pd.Series,
    age_trend: pd.DataFrame,
    agent_age_trend: pd.DataFrame,
    agent_label: str = "ET sevo",
    age_trend_title: str | None = None,
    concentration_x_col: str = "target_agent",
    concentration_x_label: str | None = None,
    selected_concentration_x: float | None = None,
):
    labels = ["Delta", "Theta", "Alpha", "Beta"]
    values = [
        row["delta_power_db"],
        row["theta_power_db"],
        row["alpha_power_db"],
        row["beta_power_db"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))

    axes[0].bar(
        labels,
        values,
        color=["#4C78A8", "#F58518", "#E45756", "#72B7B2"],
    )
    axes[0].set_ylabel("Power (dB)")
    axes[0].set_title("Expected Band Power")
    axes[0].set_ylim(-10, 25)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].grid(axis="y", alpha=0.3)

    if not agent_age_trend.empty:
        bis_line = axes[1].plot(
            agent_age_trend["age_midpoint"],
            agent_age_trend["mean_bis"],
            marker="o",
            color="purple",
            label="BIS",
        )[0]
        axes[1].set_ylim(0, 100)
        axes[1].set_xlabel("Age (years)")
        axes[1].set_ylabel("BIS")
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(
            sum(int(part) for part in str(row["age_bin"]).split("-")) / 2,
            color="black",
            linestyle="--",
            alpha=0.6,
        )

        ax2 = axes[1].twinx()
        sef_line = ax2.plot(
            agent_age_trend["age_midpoint"],
            agent_age_trend["mean_sef"],
            marker="o",
            color="tab:blue",
            label="SEF",
        )[0]
        ax2.set_ylim(0, 25)
        ax2.set_ylabel("SEF (Hz)", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        axes[1].legend([bis_line, sef_line], ["BIS", "SEF"], loc="upper left")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No age trend data",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_axis_off()
    if age_trend_title is None:
        age_trend_title = f"Age Trend At {agent_label} {row['target_agent']:.1f}%"
    axes[1].set_title(age_trend_title)

    plot_age_trend = age_trend.sort_values(concentration_x_col)
    if concentration_x_label is None:
        concentration_x_label = f"{agent_label} (%)"
    if selected_concentration_x is None:
        selected_concentration_x = float(row[concentration_x_col])

    bis_line = axes[2].plot(
        plot_age_trend[concentration_x_col],
        plot_age_trend["mean_bis"],
        marker="o",
        color="purple",
        label="BIS",
    )[0]
    axes[2].set_ylim(0, 100)
    axes[2].set_xlabel(concentration_x_label)
    axes[2].set_ylabel("BIS")
    axes[2].grid(True, alpha=0.3)

    ax3 = axes[2].twinx()
    sef_line = ax3.plot(
        plot_age_trend[concentration_x_col],
        plot_age_trend["mean_sef"],
        marker="o",
        color="tab:blue",
        label="SEF",
    )[0]
    ax3.set_ylim(0, 25)
    ax3.set_ylabel("SEF (Hz)", color="tab:blue")
    ax3.tick_params(axis="y", labelcolor="tab:blue")

    axes[2].legend([bis_line, sef_line], ["BIS", "SEF"], loc="upper right")

    axes[2].axvline(
        selected_concentration_x,
        color="black",
        linestyle="--",
        alpha=0.6,
    )
    axes[2].set_title(f"Trend In Age Bin {row['age_bin']}")

    fig.tight_layout()
    return fig
