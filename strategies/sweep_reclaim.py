from __future__ import annotations

from typing import Iterable

import pandas as pd

from strategies.common import empty_plan, finalize_plan, prepare_strategy_df


def sweep_reclaim_strategy(
    df: pd.DataFrame,
    *,
    level_columns: Iterable[str] = ("prev_day_high", "prev_day_low", "asia_high", "asia_low", "london_high", "london_low"),
    reclaim_window: int = 3,
    require_structure_confirmation: bool = False,
    require_fvg_confirmation: bool = False,
    atr_col: str = "atr_1min_14",
    target_method: str = "vwap",
    stop_buffer_atr: float = 0.35,
    max_hold_bars: int = 90,
) -> pd.DataFrame:
    out = prepare_strategy_df(df)
    plan = empty_plan(out)

    bullish = pd.Series(False, index=out.index)
    bearish = pd.Series(False, index=out.index)
    stop_long = pd.Series(index=out.index, dtype="float64")
    stop_short = pd.Series(index=out.index, dtype="float64")
    target_long = pd.Series(index=out.index, dtype="float64")
    target_short = pd.Series(index=out.index, dtype="float64")
    setup_long = pd.Series("", index=out.index, dtype="object")
    setup_short = pd.Series("", index=out.index, dtype="object")

    for level_col in level_columns:
        if level_col not in out.columns:
            continue
        level = out[level_col]
        for offset in range(0, reclaim_window + 1):
            sweep_down = (out["low"].shift(offset) < level.shift(offset)) & (out["close"] > level)
            sweep_up = (out["high"].shift(offset) > level.shift(offset)) & (out["close"] < level)
            bullish |= sweep_down.fillna(False)
            bearish |= sweep_up.fillna(False)
        stop_long = stop_long.fillna(level - out[atr_col] * stop_buffer_atr)
        stop_short = stop_short.fillna(level + out[atr_col] * stop_buffer_atr)
        setup_long = setup_long.mask(bullish & (setup_long == ""), f"sweep_reclaim_{level_col}_long")
        setup_short = setup_short.mask(bearish & (setup_short == ""), f"sweep_reclaim_{level_col}_short")

    if require_structure_confirmation:
        bullish &= out.get("choch_bullish", 0).astype(bool) | out.get("bos_bullish", 0).astype(bool)
        bearish &= out.get("choch_bearish", 0).astype(bool) | out.get("bos_bearish", 0).astype(bool)
    if require_fvg_confirmation:
        bullish &= out.get("active_bullish_fvg_count_1m", 0) > 0
        bearish &= out.get("active_bearish_fvg_count_1m", 0) > 0

    if target_method == "vwap" and "vwap" in out.columns:
        target_long = out["vwap"]
        target_short = out["vwap"]
    else:
        target_long = out["close"] + out[atr_col] * 2.0
        target_short = out["close"] - out[atr_col] * 2.0

    plan.loc[bullish & ~bearish, "entry_signal"] = 1
    plan.loc[bearish & ~bullish, "entry_signal"] = -1
    plan.loc[plan["entry_signal"] == 1, "stop_loss"] = stop_long
    plan.loc[plan["entry_signal"] == 1, "take_profit"] = target_long
    plan.loc[plan["entry_signal"] == 1, "setup"] = setup_long
    plan.loc[plan["entry_signal"] == -1, "stop_loss"] = stop_short
    plan.loc[plan["entry_signal"] == -1, "take_profit"] = target_short
    plan.loc[plan["entry_signal"] == -1, "setup"] = setup_short
    plan.loc[plan["entry_signal"] != 0, "max_hold_bars"] = max_hold_bars
    plan.loc[plan["entry_signal"] != 0, "session_name"] = "reversal"
    return finalize_plan(plan)

