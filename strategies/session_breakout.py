from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from strategies.common import empty_plan, finalize_plan, prepare_strategy_df


def session_breakout_strategy(
    df: pd.DataFrame,
    *,
    level_prefix: str = "london",
    trend_bias_col: Optional[str] = "confluence_direction_bias",
    atr_col: str = "atr_1min_14",
    require_close_confirmation: bool = True,
    retest_required: bool = False,
    max_extension_atr: float = 1.5,
    stop_method: str = "level",
    target_atr_multiple: float = 2.5,
    max_hold_bars: int = 120,
    volume_spike_required: bool = False,
) -> pd.DataFrame:
    out = prepare_strategy_df(df)
    plan = empty_plan(out)
    high_col = f"{level_prefix}_high"
    low_col = f"{level_prefix}_low"
    if high_col not in out.columns or low_col not in out.columns:
        raise ValueError(f"Missing session level columns for prefix '{level_prefix}'")

    bull_trend = pd.Series(True, index=out.index)
    bear_trend = pd.Series(True, index=out.index)
    if trend_bias_col in out.columns:
        bull_trend = out[trend_bias_col].isin(["bullish", "neutral"])
        bear_trend = out[trend_bias_col].isin(["bearish", "neutral"])

    volume_ok = pd.Series(True, index=out.index)
    if volume_spike_required and "volume_spike" in out.columns:
        volume_ok = out["volume_spike"].astype(bool)

    long_break = out["high"] > out[high_col]
    short_break = out["low"] < out[low_col]
    if require_close_confirmation:
        long_break &= out["close"] > out[high_col]
        short_break &= out["close"] < out[low_col]

    extension_long = ((out["close"] - out[high_col]) / out[atr_col].replace(0, np.nan)).fillna(0.0)
    extension_short = ((out[low_col] - out["close"]) / out[atr_col].replace(0, np.nan)).fillna(0.0)
    long_ready = long_break & (extension_long <= max_extension_atr) & bull_trend & volume_ok
    short_ready = short_break & (extension_short <= max_extension_atr) & bear_trend & volume_ok

    if retest_required:
        prior_break_above = out["close"].shift(1) > out[high_col].shift(1)
        prior_break_below = out["close"].shift(1) < out[low_col].shift(1)
        long_ready = prior_break_above & (out["low"] <= out[high_col]) & (out["close"] > out[high_col]) & bull_trend & volume_ok
        short_ready = prior_break_below & (out["high"] >= out[low_col]) & (out["close"] < out[low_col]) & bear_trend & volume_ok

    plan.loc[long_ready & ~short_ready, "entry_signal"] = 1
    plan.loc[short_ready & ~long_ready, "entry_signal"] = -1

    long_stop = out[high_col] - out[atr_col] * 0.5 if stop_method == "level" else out["last_confirmed_swing_low"]
    short_stop = out[low_col] + out[atr_col] * 0.5 if stop_method == "level" else out["last_confirmed_swing_high"]
    long_target = out["close"] + out[atr_col] * target_atr_multiple
    short_target = out["close"] - out[atr_col] * target_atr_multiple

    plan.loc[plan["entry_signal"] == 1, "stop_loss"] = long_stop
    plan.loc[plan["entry_signal"] == 1, "take_profit"] = long_target
    plan.loc[plan["entry_signal"] == 1, "setup"] = f"{level_prefix}_high_breakout"
    plan.loc[plan["entry_signal"] == 1, "session_name"] = level_prefix

    plan.loc[plan["entry_signal"] == -1, "stop_loss"] = short_stop
    plan.loc[plan["entry_signal"] == -1, "take_profit"] = short_target
    plan.loc[plan["entry_signal"] == -1, "setup"] = f"{level_prefix}_low_breakout"
    plan.loc[plan["entry_signal"] == -1, "session_name"] = level_prefix
    plan.loc[plan["entry_signal"] != 0, "max_hold_bars"] = max_hold_bars
    return finalize_plan(plan)
