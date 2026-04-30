from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.common import empty_plan, finalize_plan, prepare_strategy_df


def naive_session_breakout(df: pd.DataFrame, *, atr_col: str = "atr_1min_14", max_hold_bars: int = 60) -> pd.DataFrame:
    out = prepare_strategy_df(df)
    plan = empty_plan(out)
    long_entry = out["close"] > out["london_high"]
    short_entry = out["close"] < out["london_low"]
    plan.loc[long_entry & ~short_entry, "entry_signal"] = 1
    plan.loc[short_entry & ~long_entry, "entry_signal"] = -1
    plan.loc[plan["entry_signal"] == 1, "stop_loss"] = out["london_low"]
    plan.loc[plan["entry_signal"] == 1, "take_profit"] = out["close"] + out[atr_col] * 2.0
    plan.loc[plan["entry_signal"] == -1, "stop_loss"] = out["london_high"]
    plan.loc[plan["entry_signal"] == -1, "take_profit"] = out["close"] - out[atr_col] * 2.0
    plan.loc[plan["entry_signal"] != 0, "max_hold_bars"] = max_hold_bars
    plan.loc[plan["entry_signal"] != 0, "setup"] = "naive_session_breakout"
    return finalize_plan(plan)


def vwap_reclaim_only(df: pd.DataFrame, *, atr_col: str = "atr_1min_14", max_hold_bars: int = 45) -> pd.DataFrame:
    out = prepare_strategy_df(df)
    plan = empty_plan(out)
    plan.loc[out.get("vwap_reclaim_up", 0).astype(bool), "entry_signal"] = 1
    plan.loc[out.get("vwap_reclaim_down", 0).astype(bool), "entry_signal"] = -1
    plan.loc[plan["entry_signal"] == 1, "stop_loss"] = out["vwap"] - out[atr_col]
    plan.loc[plan["entry_signal"] == 1, "take_profit"] = out["close"] + out[atr_col] * 1.5
    plan.loc[plan["entry_signal"] == -1, "stop_loss"] = out["vwap"] + out[atr_col]
    plan.loc[plan["entry_signal"] == -1, "take_profit"] = out["close"] - out[atr_col] * 1.5
    plan.loc[plan["entry_signal"] != 0, "max_hold_bars"] = max_hold_bars
    plan.loc[plan["entry_signal"] != 0, "setup"] = "vwap_reclaim_only"
    return finalize_plan(plan)


def atr_breakout_only(df: pd.DataFrame, *, atr_col: str = "atr_1min_14", max_hold_bars: int = 45) -> pd.DataFrame:
    out = prepare_strategy_df(df)
    plan = empty_plan(out)
    upper = out["open"] + out[atr_col]
    lower = out["open"] - out[atr_col]
    plan.loc[out["close"] > upper, "entry_signal"] = 1
    plan.loc[out["close"] < lower, "entry_signal"] = -1
    plan.loc[plan["entry_signal"] == 1, "stop_loss"] = out["open"]
    plan.loc[plan["entry_signal"] == 1, "take_profit"] = out["close"] + out[atr_col] * 1.5
    plan.loc[plan["entry_signal"] == -1, "stop_loss"] = out["open"]
    plan.loc[plan["entry_signal"] == -1, "take_profit"] = out["close"] - out[atr_col] * 1.5
    plan.loc[plan["entry_signal"] != 0, "max_hold_bars"] = max_hold_bars
    plan.loc[plan["entry_signal"] != 0, "setup"] = "atr_breakout_only"
    return finalize_plan(plan)


def random_time_entry(df: pd.DataFrame, *, seed: int = 7, probability: float = 0.01, atr_col: str = "atr_1min_14") -> pd.DataFrame:
    out = prepare_strategy_df(df)
    plan = empty_plan(out)
    rng = np.random.default_rng(seed)
    draws = rng.random(len(out))
    sides = rng.choice([-1, 1], size=len(out))
    mask = draws < probability
    plan.loc[mask, "entry_signal"] = sides[mask]
    plan.loc[plan["entry_signal"] == 1, "stop_loss"] = out["close"] - out[atr_col]
    plan.loc[plan["entry_signal"] == 1, "take_profit"] = out["close"] + out[atr_col] * 2.0
    plan.loc[plan["entry_signal"] == -1, "stop_loss"] = out["close"] + out[atr_col]
    plan.loc[plan["entry_signal"] == -1, "take_profit"] = out["close"] - out[atr_col] * 2.0
    plan.loc[plan["entry_signal"] != 0, "max_hold_bars"] = 30
    plan.loc[plan["entry_signal"] != 0, "setup"] = "random_time_entry"
    return finalize_plan(plan)

