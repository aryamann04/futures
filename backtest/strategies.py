from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _pick_column(columns, candidates, required=True):
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def _prepare(
    df: pd.DataFrame,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> tuple[pd.DataFrame, str, Optional[str]]:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]

    ts_col = ts_col or _pick_column(out.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    symbol_col = symbol_col or _pick_column(out.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"], required=False)

    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    out = out.dropna(subset=[ts_col]).sort_values(([symbol_col] if symbol_col else []) + [ts_col]).reset_index(drop=True)

    return out, ts_col, symbol_col


def _empty_plan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["entry_signal"] = 0
    out["exit_signal"] = 0
    out["stop_loss_pct"] = np.nan
    out["take_profit_pct"] = np.nan
    out["max_hold_bars"] = np.nan
    out["size"] = 1.0
    return out


def ema_mean_reversion(
    df: pd.DataFrame,
    z_col: str = "price_vs_ema20",
    long_threshold: float = -0.0015,
    short_threshold: float = 0.0015,
    stop_loss_pct: float = 0.0015,
    take_profit_pct: float = 0.0025,
    max_hold_bars: int = 300,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    plan = _empty_plan(out)

    if z_col not in plan.columns:
        raise ValueError(f"Missing feature column: {z_col}")

    long_entry = plan[z_col] <= long_threshold
    short_entry = plan[z_col] >= short_threshold

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    plan.loc[entry_mask, "stop_loss_pct"] = stop_loss_pct
    plan.loc[entry_mask, "take_profit_pct"] = take_profit_pct
    plan.loc[entry_mask, "max_hold_bars"] = max_hold_bars
    plan.loc[entry_mask, "size"] = size

    return plan


def breakout_momentum(
    df: pd.DataFrame,
    breakout_col: str = "price_vs_sma20",
    vol_col: Optional[str] = "adx_14",
    long_threshold: float = 0.0015,
    short_threshold: float = -0.0015,
    vol_min: Optional[float] = None,
    stop_loss_pct: float = 0.0015,
    take_profit_pct: float = 0.0030,
    max_hold_bars: int = 300,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    plan = _empty_plan(out)

    if breakout_col not in plan.columns:
        raise ValueError(f"Missing feature column: {breakout_col}")

    long_entry = plan[breakout_col] >= long_threshold
    short_entry = plan[breakout_col] <= short_threshold

    if vol_col is not None and vol_col in plan.columns and vol_min is not None:
        vol_ok = plan[vol_col] >= vol_min
        long_entry = long_entry & vol_ok
        short_entry = short_entry & vol_ok

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    plan.loc[entry_mask, "stop_loss_pct"] = stop_loss_pct
    plan.loc[entry_mask, "take_profit_pct"] = take_profit_pct
    plan.loc[entry_mask, "max_hold_bars"] = max_hold_bars
    plan.loc[entry_mask, "size"] = size

    return plan


def rsi_reversal(
    df: pd.DataFrame,
    rsi_col: str = "rsi_14",
    long_threshold: float = 30.0,
    short_threshold: float = 70.0,
    stop_loss_pct: float = 0.0015,
    take_profit_pct: float = 0.0025,
    max_hold_bars: int = 300,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    plan = _empty_plan(out)

    if rsi_col not in plan.columns:
        raise ValueError(f"Missing feature column: {rsi_col}")

    long_entry = plan[rsi_col] <= long_threshold
    short_entry = plan[rsi_col] >= short_threshold

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    plan.loc[entry_mask, "stop_loss_pct"] = stop_loss_pct
    plan.loc[entry_mask, "take_profit_pct"] = take_profit_pct
    plan.loc[entry_mask, "max_hold_bars"] = max_hold_bars
    plan.loc[entry_mask, "size"] = size

    return plan


def combine_entry_rules(
    df: pd.DataFrame,
    long_rule: pd.Series,
    short_rule: pd.Series,
    stop_loss_pct: float,
    take_profit_pct: float,
    max_hold_bars: int,
    size: float = 1.0,
) -> pd.DataFrame:
    plan = _empty_plan(df)

    long_rule = long_rule.fillna(False)
    short_rule = short_rule.fillna(False)

    plan.loc[long_rule, "entry_signal"] = 1
    plan.loc[short_rule, "entry_signal"] = -1

    entry_mask = long_rule | short_rule
    plan.loc[entry_mask, "stop_loss_pct"] = stop_loss_pct
    plan.loc[entry_mask, "take_profit_pct"] = take_profit_pct
    plan.loc[entry_mask, "max_hold_bars"] = max_hold_bars
    plan.loc[entry_mask, "size"] = size

    return plan

def ema_bb_mean_reversion(
    df: pd.DataFrame,
    z_col: str = "price_vs_ema80",
    bb_col: str = "bb_pos",
    long_z: float = -0.0015,
    short_z: float = 0.0015,
    long_bb: float = 0.15,
    short_bb: float = 0.85,
    stop_loss_pct: float = 0.0018,
    take_profit_pct: float = 0.0055,
    max_hold_bars: int = 1200,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    plan = _empty_plan(out)

    long_entry = (plan[z_col] <= long_z) & (plan[bb_col] <= long_bb)
    short_entry = (plan[z_col] >= short_z) & (plan[bb_col] >= short_bb)

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    plan.loc[entry_mask, "stop_loss_pct"] = stop_loss_pct
    plan.loc[entry_mask, "take_profit_pct"] = take_profit_pct
    plan.loc[entry_mask, "max_hold_bars"] = max_hold_bars
    plan.loc[entry_mask, "size"] = size

    return plan

def ema_adx_mean_reversion(
    df: pd.DataFrame,
    z_col: str = "price_vs_ema80",
    adx_col: str = "adx_14",
    long_threshold: float = -0.0015,
    short_threshold: float = 0.0015,
    adx_max: float = 22.0,
    stop_loss_pct: float = 0.0018,
    take_profit_pct: float = 0.0055,
    max_hold_bars: int = 1200,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    plan = _empty_plan(out)

    long_entry = (plan[z_col] <= long_threshold) & (plan[adx_col] <= adx_max)
    short_entry = (plan[z_col] >= short_threshold) & (plan[adx_col] <= adx_max)

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    plan.loc[entry_mask, "stop_loss_pct"] = stop_loss_pct
    plan.loc[entry_mask, "take_profit_pct"] = take_profit_pct
    plan.loc[entry_mask, "max_hold_bars"] = max_hold_bars
    plan.loc[entry_mask, "size"] = size

    return plan

def ema_atr_mean_reversion(
    df: pd.DataFrame,
    z_col: str = "ema80_atr_dist",
    long_threshold: float = -1.25,
    short_threshold: float = 1.25,
    stop_loss_pct: float = 0.0018,
    take_profit_pct: float = 0.0055,
    max_hold_bars: int = 1200,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    plan = _empty_plan(out)

    long_entry = plan[z_col] <= long_threshold
    short_entry = plan[z_col] >= short_threshold

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    plan.loc[entry_mask, "stop_loss_pct"] = stop_loss_pct
    plan.loc[entry_mask, "take_profit_pct"] = take_profit_pct
    plan.loc[entry_mask, "max_hold_bars"] = max_hold_bars
    plan.loc[entry_mask, "size"] = size

    return plan

def ema_adx_session_mean_reversion(
    df: pd.DataFrame,
    z_col: str = "price_vs_ema100",
    adx_col: str = "adx_14",
    session_col: str = "is_london_ny",
    long_threshold: float = -0.0018,
    short_threshold: float = 0.0018,
    adx_max: float = 20.0,
    stop_loss_pct: float = 0.0018,
    take_profit_pct: float = 0.0058,
    max_hold_bars: int = 1500,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    plan = _empty_plan(out)

    long_entry = (
        (plan[z_col] <= long_threshold)
        & (plan[adx_col] <= adx_max)
        & (plan[session_col] == 1)
    )
    short_entry = (
        (plan[z_col] >= short_threshold)
        & (plan[adx_col] <= adx_max)
        & (plan[session_col] == 1)
    )

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    plan.loc[entry_mask, "stop_loss_pct"] = stop_loss_pct
    plan.loc[entry_mask, "take_profit_pct"] = take_profit_pct
    plan.loc[entry_mask, "max_hold_bars"] = max_hold_bars
    plan.loc[entry_mask, "size"] = size

    return plan