from __future__ import annotations

import pandas as pd

from strategies.common import empty_plan, finalize_plan, prepare_strategy_df


def fvg_pullback_strategy(
    df: pd.DataFrame,
    *,
    timeframe: str = "5m",
    atr_col: str = "atr_1min_14",
    min_size_atr: float = 0.1,
    max_fvg_age_bars: int = 120,
    entry_mode: str = "midpoint",
    target_atr_multiple: float = 2.5,
    max_hold_bars: int = 120,
) -> pd.DataFrame:
    out = prepare_strategy_df(df)
    plan = empty_plan(out)
    prefix = timeframe.replace("min", "m")
    bull_col = f"nearest_bullish_fvg_below_{prefix}"
    bear_col = f"nearest_bearish_fvg_above_{prefix}"
    if bull_col not in out.columns or bear_col not in out.columns:
        raise ValueError(f"Missing FVG projection columns for timeframe '{timeframe}'")

    bullish_bias = out.get("price_above_vwap", pd.Series(1, index=out.index)).astype(bool)
    bearish_bias = out.get("price_below_vwap", pd.Series(1, index=out.index)).astype(bool)
    touch_mid = out.get(f"touches_fvg_mid_{prefix}", 0).astype(bool)

    if entry_mode == "midpoint":
        long_entry = touch_mid & bullish_bias & out[bull_col].notna()
        short_entry = touch_mid & bearish_bias & out[bear_col].notna()
    else:
        inside = out.get(f"inside_any_fvg_{prefix}", 0).astype(bool)
        long_entry = inside & bullish_bias & out[bull_col].notna()
        short_entry = inside & bearish_bias & out[bear_col].notna()

    plan.loc[long_entry & ~short_entry, "entry_signal"] = 1
    plan.loc[short_entry & ~long_entry, "entry_signal"] = -1
    plan.loc[plan["entry_signal"] == 1, "stop_loss"] = out[bull_col] - out[atr_col] * 0.35
    plan.loc[plan["entry_signal"] == 1, "take_profit"] = out["close"] + out[atr_col] * target_atr_multiple
    plan.loc[plan["entry_signal"] == 1, "setup"] = f"fvg_pullback_{timeframe}_long"
    plan.loc[plan["entry_signal"] == -1, "stop_loss"] = out[bear_col] + out[atr_col] * 0.35
    plan.loc[plan["entry_signal"] == -1, "take_profit"] = out["close"] - out[atr_col] * target_atr_multiple
    plan.loc[plan["entry_signal"] == -1, "setup"] = f"fvg_pullback_{timeframe}_short"
    plan.loc[plan["entry_signal"] != 0, "max_hold_bars"] = max_hold_bars
    plan.loc[plan["entry_signal"] != 0, "session_name"] = "continuation"
    return finalize_plan(plan)

