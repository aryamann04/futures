from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from features.intraday_utils import pick_column


def add_market_structure_features(
    df: pd.DataFrame,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
    swing_left: int = 3,
    swing_right: int = 3,
) -> pd.DataFrame:
    """Confirm swings after `swing_right` future bars, making the delay explicit."""
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    ts_col = ts_col or pick_column(out.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    symbol_col = symbol_col or pick_column(out.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"], required=False)
    out = out.sort_values(([symbol_col] if symbol_col else []) + [ts_col]).reset_index(drop=True)

    for col in [
        "swing_high_price",
        "swing_low_price",
        "last_confirmed_swing_high",
        "last_confirmed_swing_low",
        "bos_bullish",
        "bos_bearish",
        "choch_bullish",
        "choch_bearish",
        "swing_high_confirmed",
        "swing_low_confirmed",
    ]:
        out[col] = np.nan if "price" in col or "last_" in col else 0

    groups = [(None, out)] if symbol_col is None else out.groupby(symbol_col, sort=False)
    for _, group in groups:
        highs = group["high"].to_numpy(dtype=float)
        lows = group["low"].to_numpy(dtype=float)
        idx = group.index.to_list()
        swing_high_levels: list[float] = []
        swing_low_levels: list[float] = []
        last_swing_high = np.nan
        last_swing_low = np.nan
        prev_swing_high = np.nan
        prev_swing_low = np.nan

        for local_i, global_i in enumerate(idx):
            pivot_i = local_i - swing_right
            if pivot_i >= swing_left:
                start = pivot_i - swing_left
                end = pivot_i + swing_right + 1
                pivot_high = highs[pivot_i]
                pivot_low = lows[pivot_i]
                window_high = highs[start:end]
                window_low = lows[start:end]
                if np.isfinite(pivot_high) and pivot_high == np.max(window_high):
                    prev_swing_high = last_swing_high
                    last_swing_high = pivot_high
                    swing_high_levels.append(pivot_high)
                    out.at[global_i, "swing_high_confirmed"] = 1
                    out.at[global_i, "swing_high_price"] = pivot_high
                if np.isfinite(pivot_low) and pivot_low == np.min(window_low):
                    prev_swing_low = last_swing_low
                    last_swing_low = pivot_low
                    swing_low_levels.append(pivot_low)
                    out.at[global_i, "swing_low_confirmed"] = 1
                    out.at[global_i, "swing_low_price"] = pivot_low

            out.at[global_i, "last_confirmed_swing_high"] = last_swing_high
            out.at[global_i, "last_confirmed_swing_low"] = last_swing_low
            if np.isfinite(last_swing_high) and highs[local_i] > last_swing_high:
                out.at[global_i, "bos_bullish"] = 1
            if np.isfinite(last_swing_low) and lows[local_i] < last_swing_low:
                out.at[global_i, "bos_bearish"] = 1
            if np.isfinite(prev_swing_high) and highs[local_i] > prev_swing_high and np.isfinite(last_swing_low):
                out.at[global_i, "choch_bullish"] = 1
            if np.isfinite(prev_swing_low) and lows[local_i] < prev_swing_low and np.isfinite(last_swing_high):
                out.at[global_i, "choch_bearish"] = 1

    return out
