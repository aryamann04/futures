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
    symbol_col = symbol_col or _pick_column(
        out.columns,
        ["symbol", "raw_symbol", "instrument_id", "ticker"],
        required=False,
    )

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
    out["max_hold_seconds"] = np.nan
    out["size"] = 1.0
    return out


def _set_trade_params(
    plan: pd.DataFrame,
    entry_mask: pd.Series,
    stop_loss_pct: float,
    take_profit_pct: float,
    max_hold_bars: Optional[int],
    max_hold_seconds: Optional[float],
    size: float,
) -> None:
    plan.loc[entry_mask, "stop_loss_pct"] = stop_loss_pct
    plan.loc[entry_mask, "take_profit_pct"] = take_profit_pct

    if max_hold_bars is not None:
        plan.loc[entry_mask, "max_hold_bars"] = max_hold_bars
    if max_hold_seconds is not None:
        plan.loc[entry_mask, "max_hold_seconds"] = max_hold_seconds

    plan.loc[entry_mask, "size"] = size


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


def ema_mean_reversion(
    df: pd.DataFrame,
    z_col: str = "price_vs_ema20",
    long_threshold: float = -0.0015,
    short_threshold: float = 0.0015,
    stop_loss_pct: float = 0.0015,
    take_profit_pct: float = 0.0025,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [z_col])

    plan = _empty_plan(out)

    long_entry = out[z_col] <= long_threshold
    short_entry = out[z_col] >= short_threshold

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

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
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [breakout_col])

    if vol_col is not None and vol_min is not None:
        _require_columns(out, [vol_col])

    plan = _empty_plan(out)

    long_entry = out[breakout_col] >= long_threshold
    short_entry = out[breakout_col] <= short_threshold

    if vol_col is not None and vol_min is not None:
        vol_ok = out[vol_col] >= vol_min
        long_entry = long_entry & vol_ok
        short_entry = short_entry & vol_ok

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def rsi_reversal(
    df: pd.DataFrame,
    rsi_col: str = "rsi_14",
    long_threshold: float = 30.0,
    short_threshold: float = 70.0,
    stop_loss_pct: float = 0.0015,
    take_profit_pct: float = 0.0025,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [rsi_col])

    plan = _empty_plan(out)

    long_entry = out[rsi_col] <= long_threshold
    short_entry = out[rsi_col] >= short_threshold

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def combine_entry_rules(
    df: pd.DataFrame,
    long_rule: pd.Series,
    short_rule: pd.Series,
    stop_loss_pct: float,
    take_profit_pct: float,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    size: float = 1.0,
) -> pd.DataFrame:
    plan = _empty_plan(df)

    long_rule = long_rule.fillna(False)
    short_rule = short_rule.fillna(False)

    plan.loc[long_rule, "entry_signal"] = 1
    plan.loc[short_rule, "entry_signal"] = -1

    entry_mask = long_rule | short_rule
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

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
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [z_col, bb_col])

    plan = _empty_plan(out)

    long_entry = (out[z_col] <= long_z) & (out[bb_col] <= long_bb)
    short_entry = (out[z_col] >= short_z) & (out[bb_col] >= short_bb)

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

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
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [z_col, adx_col])

    plan = _empty_plan(out)

    long_entry = (out[z_col] <= long_threshold) & (out[adx_col] <= adx_max)
    short_entry = (out[z_col] >= short_threshold) & (out[adx_col] <= adx_max)

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def ema_atr_mean_reversion(
    df: pd.DataFrame,
    z_col: str = "ema80_atr_dist",
    long_threshold: float = -1.25,
    short_threshold: float = 1.25,
    stop_loss_pct: float = 0.0018,
    take_profit_pct: float = 0.0055,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [z_col])

    plan = _empty_plan(out)

    long_entry = out[z_col] <= long_threshold
    short_entry = out[z_col] >= short_threshold

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

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
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [z_col, adx_col, session_col])

    plan = _empty_plan(out)

    long_entry = (
        (out[z_col] <= long_threshold)
        & (out[adx_col] <= adx_max)
        & (out[session_col] == 1)
    )
    short_entry = (
        (out[z_col] >= short_threshold)
        & (out[adx_col] <= adx_max)
        & (out[session_col] == 1)
    )

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def ema_adx_scaled_mean_reversion(
    df: pd.DataFrame,
    z_col: str = "price_vs_ema90",
    adx_col: str = "adx_14",
    long_threshold: float = -0.0015,
    short_threshold: float = 0.0015,
    adx_soft: float = 28.0,
    adx_hard: float = 35.0,
    stop_loss_pct: float = 0.0020,
    take_profit_pct: float = 0.0065,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [z_col, adx_col])

    plan = _empty_plan(out)

    long_entry = out[z_col] <= long_threshold
    short_entry = out[z_col] >= short_threshold

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    adx = out[adx_col]
    size_arr = np.where(adx <= adx_soft, 1.0, np.where(adx <= adx_hard, 0.5, 0.0))

    plan.loc[entry_mask, "size"] = size_arr[entry_mask]
    plan.loc[entry_mask, "stop_loss_pct"] = stop_loss_pct
    plan.loc[entry_mask, "take_profit_pct"] = take_profit_pct

    if max_hold_bars is not None:
        plan.loc[entry_mask, "max_hold_bars"] = max_hold_bars
    if max_hold_seconds is not None:
        plan.loc[entry_mask, "max_hold_seconds"] = max_hold_seconds

    return plan


def prior_session_breakout(
    df: pd.DataFrame,
    high_col: str = "prev_session_high",
    low_col: str = "prev_session_low",
    breakout_buffer: float = 0.0003,
    volume_col: Optional[str] = "rel_volume_20",
    volume_min: Optional[float] = None,
    stop_loss_pct: float = 0.0018,
    take_profit_pct: float = 0.0050,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [high_col, low_col])

    if volume_col is not None and volume_min is not None:
        _require_columns(out, [volume_col])

    plan = _empty_plan(out)

    long_entry = out["close"] >= out[high_col] * (1.0 + breakout_buffer)
    short_entry = out["close"] <= out[low_col] * (1.0 - breakout_buffer)

    if volume_col is not None and volume_min is not None:
        vol_ok = out[volume_col] >= volume_min
        long_entry = long_entry & vol_ok
        short_entry = short_entry & vol_ok

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def prior_session_failed_breakout(
    df: pd.DataFrame,
    high_col: str = "prev_session_high",
    low_col: str = "prev_session_low",
    sweep_buffer: float = 0.0002,
    volume_col: Optional[str] = "rel_volume_20",
    volume_min: Optional[float] = None,
    stop_loss_pct: float = 0.0015,
    take_profit_pct: float = 0.0045,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [high_col, low_col])

    if volume_col is not None and volume_min is not None:
        _require_columns(out, [volume_col])

    plan = _empty_plan(out)

    short_entry = (
        (out["high"] >= out[high_col] * (1.0 + sweep_buffer))
        & (out["close"] < out[high_col])
    )

    long_entry = (
        (out["low"] <= out[low_col] * (1.0 - sweep_buffer))
        & (out["close"] > out[low_col])
    )

    if volume_col is not None and volume_min is not None:
        vol_ok = out[volume_col] >= volume_min
        long_entry = long_entry & vol_ok
        short_entry = short_entry & vol_ok

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def opening_range_breakout(
    df: pd.DataFrame,
    high_col: str = "opening_range_high_5m",
    low_col: str = "opening_range_low_5m",
    breakout_buffer: float = 0.0002,
    volume_col: Optional[str] = "rel_volume_20",
    volume_min: Optional[float] = None,
    stop_loss_pct: float = 0.0018,
    take_profit_pct: float = 0.0050,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [high_col, low_col])

    if volume_col is not None and volume_min is not None:
        _require_columns(out, [volume_col])

    plan = _empty_plan(out)

    long_entry = out["close"] >= out[high_col] * (1.0 + breakout_buffer)
    short_entry = out["close"] <= out[low_col] * (1.0 - breakout_buffer)

    if volume_col is not None and volume_min is not None:
        vol_ok = out[volume_col] >= volume_min
        long_entry = long_entry & vol_ok
        short_entry = short_entry & vol_ok

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def rolling_range_fade(
    df: pd.DataFrame,
    high_col: str = "rolling_high_60m",
    low_col: str = "rolling_low_60m",
    sweep_buffer: float = 0.0002,
    volume_col: Optional[str] = "rel_volume_20",
    volume_max: Optional[float] = None,
    stop_loss_pct: float = 0.0015,
    take_profit_pct: float = 0.0045,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [high_col, low_col])

    if volume_col is not None and volume_max is not None:
        _require_columns(out, [volume_col])

    plan = _empty_plan(out)

    short_entry = (
        (out["high"] >= out[high_col] * (1.0 + sweep_buffer))
        & (out["close"] < out[high_col])
    )

    long_entry = (
        (out["low"] <= out[low_col] * (1.0 - sweep_buffer))
        & (out["close"] > out[low_col])
    )

    if volume_col is not None and volume_max is not None:
        vol_ok = out[volume_col] <= volume_max
        long_entry = long_entry & vol_ok
        short_entry = short_entry & vol_ok

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def prior_session_failed_breakout_confirmed(
    df: pd.DataFrame,
    high_col: str = "prev_session_high",
    low_col: str = "prev_session_low",
    volume_col: str = "rel_volume_20",
    ret_col: str = "ret_1",
    sweep_buffer: float = 0.0005,
    volume_min: float = 1.5,
    stop_loss_pct: float = 0.0015,
    take_profit_pct: float = 0.0045,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = 900.0,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [high_col, low_col, volume_col, ret_col])

    plan = _empty_plan(out)

    short_entry = (
        (out["high"] >= out[high_col] * (1.0 + sweep_buffer))
        & (out["close"] < out[high_col])
        & (out[volume_col] >= volume_min)
        & (out[ret_col] < 0)
    )

    long_entry = (
        (out["low"] <= out[low_col] * (1.0 - sweep_buffer))
        & (out["close"] > out[low_col])
        & (out[volume_col] >= volume_min)
        & (out[ret_col] > 0)
    )

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def opening_range_failed_breakout(
    df: pd.DataFrame,
    high_col: str = "opening_range_high_5m",
    low_col: str = "opening_range_low_5m",
    volume_col: str = "rel_volume_20",
    sweep_buffer: float = 0.0003,
    volume_min: float = 1.5,
    stop_loss_pct: float = 0.0015,
    take_profit_pct: float = 0.0045,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = 900.0,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [high_col, low_col, volume_col])

    plan = _empty_plan(out)

    short_entry = (
        (out["high"] >= out[high_col] * (1.0 + sweep_buffer))
        & (out["close"] < out[high_col])
        & (out[volume_col] >= volume_min)
    )

    long_entry = (
        (out["low"] <= out[low_col] * (1.0 - sweep_buffer))
        & (out["close"] > out[low_col])
        & (out[volume_col] >= volume_min)
    )

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def ema_structure_mean_reversion(
    df: pd.DataFrame,
    z_col: str = "price_vs_ema90",
    adx_col: str = "adx_14",
    support_col: str = "prev_session_low",
    resistance_col: str = "prev_session_high",
    long_threshold: float = -0.0015,
    short_threshold: float = 0.0015,
    level_tolerance: float = 0.0010,
    adx_max: float = 22.0,
    stop_loss_pct: float = 0.0018,
    take_profit_pct: float = 0.0055,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = 900.0,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [z_col, adx_col, support_col, resistance_col])

    plan = _empty_plan(out)

    near_support = ((out["close"] - out[support_col]).abs() / out["close"]) <= level_tolerance
    near_resistance = ((out["close"] - out[resistance_col]).abs() / out["close"]) <= level_tolerance

    long_entry = (
        (out[z_col] <= long_threshold)
        & (out[adx_col] <= adx_max)
        & near_support
    )

    short_entry = (
        (out[z_col] >= short_threshold)
        & (out[adx_col] <= adx_max)
        & near_resistance
    )

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def bollinger_exhaustion_reversal(
    df: pd.DataFrame,
    bb_col: str = "bb_pos",
    adx_col: str = "adx_14",
    resistance_col: Optional[str] = None,
    support_col: Optional[str] = None,
    upper_threshold: float = 0.95,
    lower_threshold: float = 0.05,
    adx_max: float = 25.0,
    level_tolerance: float = 0.0010,
    stop_loss_pct: float = 0.0018,
    take_profit_pct: float = 0.0045,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = 900.0,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [bb_col, adx_col])

    plan = _empty_plan(out)

    long_entry = (out[bb_col] <= lower_threshold) & (out[adx_col] <= adx_max)
    short_entry = (out[bb_col] >= upper_threshold) & (out[adx_col] <= adx_max)

    if support_col is not None:
        _require_columns(out, [support_col])
        near_support = ((out["close"] - out[support_col]).abs() / out["close"]) <= level_tolerance
        long_entry = long_entry & near_support

    if resistance_col is not None:
        _require_columns(out, [resistance_col])
        near_resistance = ((out["close"] - out[resistance_col]).abs() / out["close"]) <= level_tolerance
        short_entry = short_entry & near_resistance

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def trend_pullback_reentry(
    df: pd.DataFrame,
    fast_ema_col: str = "ema_50",
    slow_ema_col: str = "ema_100",
    adx_col: str = "adx_14",
    price_col: str = "close",
    adx_min: float = 25.0,
    stop_loss_pct: float = 0.0018,
    take_profit_pct: float = 0.0055,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = 1200.0,
    size: float = 1.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [fast_ema_col, slow_ema_col, adx_col, price_col])

    plan = _empty_plan(out)

    trend_up = (out[fast_ema_col] > out[slow_ema_col]) & (out[adx_col] >= adx_min)
    trend_down = (out[fast_ema_col] < out[slow_ema_col]) & (out[adx_col] >= adx_min)

    pullback_long_prev = out[price_col].shift(1) < out[fast_ema_col].shift(1)
    resume_long = out[price_col] > out[fast_ema_col]

    pullback_short_prev = out[price_col].shift(1) > out[fast_ema_col].shift(1)
    resume_short = out[price_col] < out[fast_ema_col]

    long_entry = trend_up & pullback_long_prev & resume_long
    short_entry = trend_down & pullback_short_prev & resume_short

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan