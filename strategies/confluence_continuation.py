from __future__ import annotations

import pandas as pd

from strategies.common import empty_plan, finalize_plan, prepare_strategy_df


def confluence_continuation_strategy(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr_1min_14",
    score_threshold: int = 3,
    target_atr_multiple: float = 2.5,
    max_hold_bars: int = 120,
) -> pd.DataFrame:
    out = prepare_strategy_df(df)
    plan = empty_plan(out)
    lower = out["confluence_zone_lower"]
    upper = out["confluence_zone_upper"]

    long_entry = (
        (out["confluence_score"] >= score_threshold)
        & (out["confluence_direction_bias"] == "bullish")
        & (out["low"] <= upper)
        & (out["close"] >= lower)
    )
    short_entry = (
        (out["confluence_score"] >= score_threshold)
        & (out["confluence_direction_bias"] == "bearish")
        & (out["high"] >= lower)
        & (out["close"] <= upper)
    )

    plan.loc[long_entry & ~short_entry, "entry_signal"] = 1
    plan.loc[short_entry & ~long_entry, "entry_signal"] = -1
    plan.loc[plan["entry_signal"] == 1, "stop_loss"] = lower - out[atr_col] * 0.35
    plan.loc[plan["entry_signal"] == 1, "take_profit"] = out["close"] + out[atr_col] * target_atr_multiple
    plan.loc[plan["entry_signal"] == 1, "setup"] = "confluence_continuation_long"
    plan.loc[plan["entry_signal"] == -1, "stop_loss"] = upper + out[atr_col] * 0.35
    plan.loc[plan["entry_signal"] == -1, "take_profit"] = out["close"] - out[atr_col] * target_atr_multiple
    plan.loc[plan["entry_signal"] == -1, "setup"] = "confluence_continuation_short"
    plan.loc[plan["entry_signal"] != 0, "max_hold_bars"] = max_hold_bars
    plan.loc[plan["entry_signal"] != 0, "session_name"] = "confluence"
    return finalize_plan(plan)

