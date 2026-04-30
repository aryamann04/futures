from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def add_confluence_features(
    df: pd.DataFrame,
    atr_col: Optional[str] = None,
    tolerance_points: Optional[float] = None,
    tolerance_atr: float = 0.2,
) -> pd.DataFrame:
    out = df.copy()
    level_columns = [
        c
        for c in out.columns
        if any(
            token in c
            for token in [
                "prev_day_",
                "asia_",
                "london_",
                "new_york_",
                "open_range",
                "swing",
                "vwap",
                "fvg",
                "volume_poc",
                "value_area_",
            ]
        )
        and out[c].dtype != object
    ]
    out["confluence_zone_lower"] = np.nan
    out["confluence_zone_upper"] = np.nan
    out["confluence_zone_center"] = np.nan
    out["confluence_score"] = 0
    out["confluence_direction_bias"] = "neutral"

    for i, row in out.iterrows():
        levels = []
        for col in level_columns:
            value = row[col]
            if pd.notna(value):
                levels.append((col, float(value)))
        if not levels:
            continue
        tol = tolerance_points
        if tol is None:
            atr_value = float(row[atr_col]) if atr_col and atr_col in out.columns and pd.notna(row[atr_col]) else np.nan
            tol = atr_value * tolerance_atr if np.isfinite(atr_value) else max(abs(float(row["close"])) * 0.001, 0.25)
        levels.sort(key=lambda item: item[1])
        best_cluster: list[tuple[str, float]] = []
        current_cluster: list[tuple[str, float]] = [levels[0]]
        for candidate in levels[1:]:
            if abs(candidate[1] - current_cluster[-1][1]) <= tol:
                current_cluster.append(candidate)
            else:
                if len(current_cluster) > len(best_cluster):
                    best_cluster = current_cluster[:]
                current_cluster = [candidate]
        if len(current_cluster) > len(best_cluster):
            best_cluster = current_cluster
        if not best_cluster:
            continue
        prices = [price for _, price in best_cluster]
        out.at[i, "confluence_zone_lower"] = min(prices)
        out.at[i, "confluence_zone_upper"] = max(prices)
        out.at[i, "confluence_zone_center"] = float(np.mean(prices))
        out.at[i, "confluence_score"] = len(best_cluster)
        below = sum(price <= float(row["close"]) for _, price in best_cluster)
        above = len(best_cluster) - below
        out.at[i, "confluence_direction_bias"] = "bullish" if below > above else "bearish" if above > below else "neutral"
    return out
