from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


PREFERRED_FILES = [
    "reference_table_sevo_all_v2_no_n2o_by_ET.csv",
    "reference_table_sevo_all_v1_by_ET.csv",
]


def find_reference_file(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    for name in PREFERRED_FILES:
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        "No reference table found. Expected one of: "
        + ", ".join(PREFERRED_FILES)
    )


def load_reference_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["age_bin"] = df["age_bin"].astype(str)
    df["target_agent"] = df["target_agent"].astype(float).round(1)

    if "usable_cell" in df.columns and df["usable_cell"].dtype != bool:
        df["usable_cell"] = df["usable_cell"].astype(str).str.lower().eq("true")

    return df


def parse_age_bin(age_bin: str) -> tuple[int, int]:
    left, right = str(age_bin).split("-")
    return int(left), int(right)


def sorted_age_bins(df: pd.DataFrame) -> list[str]:
    return sorted(df["age_bin"].astype(str).unique(), key=lambda x: parse_age_bin(x)[0])


def age_to_bin(age: int | float, available_bins: list[str]) -> Optional[str]:
    for age_bin in available_bins:
        left, right = parse_age_bin(age_bin)
        if left <= age <= right:
            return age_bin
    return None


def _add_mac_bin(df: pd.DataFrame, mac_step: float = 0.1) -> pd.DataFrame:
    binned = df.copy()
    scaled_mac = binned["mean_mac"].astype(float) / float(mac_step)
    binned["mac_bin"] = (np.floor(scaled_mac + 0.5) * float(mac_step)).round(1)
    return binned


def _weighted_mean(rows: pd.DataFrame, column: str) -> float:
    values = rows[column].astype(float)
    weights = rows["n_cases_clean"].clip(lower=1).astype(float)
    return float((values * weights).sum() / weights.sum())


def _aggregate_reference_rows(
    rows: pd.DataFrame,
    mac_bin: float | None = None,
) -> pd.Series:
    numeric_mean_cols = [
        "target_agent",
        "mean_age",
        "mean_agent",
        "sd_agent",
        "mean_mac",
        "mean_bis",
        "sd_bis",
        "mean_sef",
        "sd_sef",
        "mean_sqi",
        "delta_power_db",
        "theta_power_db",
        "alpha_power_db",
        "beta_power_db",
        "alpha_peak_freq",
        "alpha_peak_power_db",
        "alpha_delta_diff_db",
    ]
    count_cols = ["n_cases_clean", "n_windows_clean", "n_female", "n_male"]

    first = rows.iloc[0].copy()
    for column in numeric_mean_cols:
        if column in rows.columns:
            first[column] = _weighted_mean(rows, column)
    for column in count_cols:
        if column in rows.columns:
            first[column] = int(rows[column].sum())
    if "usable_cell" in rows.columns:
        first["usable_cell"] = bool(rows["usable_cell"].all())
    if mac_bin is not None:
        first["mac_bin"] = float(mac_bin)
    return first


def lookup_reference(
    df: pd.DataFrame,
    age: int | float,
    agent_value: Optional[float] = None,
    et_sevo: Optional[float] = None,
    mac: Optional[float] = None,
    mac_tolerance: float | None = None,
    require_usable: bool = True,
) -> Tuple[Optional[pd.Series], Optional[str]]:
    available_bins = sorted_age_bins(df)
    age_bin = age_to_bin(age, available_bins)

    if age_bin is None:
        return (
            None,
            f"Unsupported age. Available bins: {available_bins[0]} to {available_bins[-1]}",
        )

    candidates = df[df["age_bin"] == age_bin].copy()

    if require_usable and "usable_cell" in candidates.columns:
        candidates = candidates[candidates["usable_cell"]]

    if candidates.empty:
        return None, f"No usable reference cell for age bin {age_bin}."

    if agent_value is None and et_sevo is not None:
        agent_value = et_sevo

    if agent_value is not None:
        target = round(float(agent_value), 1)
        candidates["distance"] = (candidates["target_agent"] - target).abs()
        row = candidates.sort_values(
            ["distance", "n_cases_clean"], ascending=[True, False]
        ).iloc[0]
    elif mac is not None:
        target = round(float(mac), 1)
        candidates = _add_mac_bin(candidates)
        candidates = candidates[candidates["mac_bin"] == target]
        if candidates.empty:
            return None, f"No reference cell in MAC {target:.1f} bin for age bin {age_bin}."
        row = _aggregate_reference_rows(candidates, mac_bin=target)
    else:
        return None, "Either agent_value or mac is required."

    return row, None


def get_age_bin_trend(
    df: pd.DataFrame,
    age_bin: str,
    require_usable: bool = True,
) -> pd.DataFrame:
    trend = df[df["age_bin"] == age_bin].copy()
    if require_usable and "usable_cell" in trend.columns:
        trend = trend[trend["usable_cell"]]
    return trend.sort_values("target_agent")


def get_mac_bin_trend(
    df: pd.DataFrame,
    age_bin: str,
    require_usable: bool = True,
) -> pd.DataFrame:
    candidates = df[df["age_bin"] == age_bin].copy()
    if require_usable and "usable_cell" in candidates.columns:
        candidates = candidates[candidates["usable_cell"]]

    if candidates.empty:
        return candidates

    candidates = _add_mac_bin(candidates)
    rows = [
        _aggregate_reference_rows(mac_rows, mac_bin=mac_bin)
        for mac_bin, mac_rows in candidates.groupby("mac_bin", sort=True)
    ]
    return pd.DataFrame(rows).sort_values("mac_bin")


def get_agent_age_trend(
    df: pd.DataFrame,
    target_agent: float,
    require_usable: bool = True,
) -> pd.DataFrame:
    trend = df[df["target_agent"].round(1) == round(float(target_agent), 1)].copy()
    if require_usable and "usable_cell" in trend.columns:
        trend = trend[trend["usable_cell"]]

    if trend.empty:
        return trend

    trend["age_midpoint"] = trend["age_bin"].apply(
        lambda age_bin: sum(parse_age_bin(age_bin)) / 2
    )
    return trend.sort_values("age_midpoint")


def get_mac_age_trend(
    df: pd.DataFrame,
    mac: float,
    mac_tolerance: float | None = None,
    require_usable: bool = True,
) -> pd.DataFrame:
    candidates = df.copy()
    if require_usable and "usable_cell" in candidates.columns:
        candidates = candidates[candidates["usable_cell"]]

    if candidates.empty:
        return candidates

    selected_rows = []
    for _, age_rows in candidates.groupby("age_bin", sort=False):
        age_rows = age_rows.copy()
        age_rows = _add_mac_bin(age_rows)
        age_rows = age_rows[age_rows["mac_bin"] == round(float(mac), 1)]
        if age_rows.empty:
            continue
        selected_rows.append(
            _aggregate_reference_rows(age_rows, mac_bin=round(float(mac), 1))
        )

    trend = pd.DataFrame(selected_rows)
    if trend.empty:
        return trend
    trend["age_midpoint"] = trend["age_bin"].apply(
        lambda age_bin: sum(parse_age_bin(age_bin)) / 2
    )
    return trend.sort_values("age_midpoint")


def row_to_display_dict(row: pd.Series, age: int | float) -> dict:
    return {
        "age": age,
        "matched_age_bin": row["age_bin"],
        "matched_agent": row["target_agent"],
        "mean_mac": row["mean_mac"],
        "n_cases_clean": row["n_cases_clean"],
        "usable_cell": row.get("usable_cell", True),
        "expected_BIS": row["mean_bis"],
        "expected_SEF": row["mean_sef"],
        "delta_power_db": row["delta_power_db"],
        "theta_power_db": row["theta_power_db"],
        "alpha_power_db": row["alpha_power_db"],
        "beta_power_db": row["beta_power_db"],
        "alpha_peak_freq": row["alpha_peak_freq"],
        "alpha_peak_power_db": row["alpha_peak_power_db"],
    }
