from __future__ import annotations

import pandas as pd

from strategies.common import empty_plan, finalize_plan, prepare_strategy_df


def opening_range_breakout_strategy(
    df: pd.DataFrame,
    *,
    opening_range_prefix: str = "ny_open_range",
    atr_col: str = "atr_1min_14",
    require_vwap_filter: bool = False,
    require_volume_spike: bool = False,
    target_atr_multiple: float = 2.0,
    max_hold_bars: int = 90,
) -> pd.DataFrame:
    out = prepare_strategy_df(df)
    plan = empty_plan(out)
    high_col = f"{opening_range_prefix}_high"
    low_col = f"{opening_range_prefix}_low"

    long_entry = (out["close"] > out[high_col]) & (out["close"].shift(1) <= out[high_col].shift(1))
    short_entry = (out["close"] < out[low_col]) & (out["close"].shift(1) >= out[low_col].shift(1))

    if require_vwap_filter and "vwap" in out.columns:
        long_entry &= out["close"] > out["vwap"]
        short_entry &= out["close"] < out["vwap"]
    if require_volume_spike and "volume_spike" in out.columns:
        long_entry &= out["volume_spike"].astype(bool)
        short_entry &= out["volume_spike"].astype(bool)

    plan.loc[long_entry & ~short_entry, "entry_signal"] = 1
    plan.loc[short_entry & ~long_entry, "entry_signal"] = -1
    plan.loc[plan["entry_signal"] == 1, "stop_loss"] = out[low_col]
    plan.loc[plan["entry_signal"] == 1, "take_profit"] = out["close"] + out[atr_col] * target_atr_multiple
    plan.loc[plan["entry_signal"] == 1, "setup"] = f"{opening_range_prefix}_breakout_long"
    plan.loc[plan["entry_signal"] == -1, "stop_loss"] = out[high_col]
    plan.loc[plan["entry_signal"] == -1, "take_profit"] = out["close"] - out[atr_col] * target_atr_multiple
    plan.loc[plan["entry_signal"] == -1, "setup"] = f"{opening_range_prefix}_breakout_short"
    plan.loc[plan["entry_signal"] != 0, "max_hold_bars"] = max_hold_bars
    plan.loc[plan["entry_signal"] != 0, "session_name"] = "opening_range"
    return finalize_plan(plan)

