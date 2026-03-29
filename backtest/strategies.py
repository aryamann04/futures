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


def _shift_bool_false(s: pd.Series) -> pd.Series:
    shifted = s.shift(1)
    shifted = shifted.where(shifted.notna(), False)
    return shifted.astype(bool)


def _near_level(close: pd.Series, level: pd.Series, tolerance: float) -> pd.Series:
    out = ((close - level).abs() / close.replace(0, np.nan) <= tolerance)
    return out.where(out.notna(), False).astype(bool)


def _entered_zone(price: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    in_zone = ((price >= lower) & (price <= upper))
    in_zone = in_zone.where(in_zone.notna(), False).astype(bool)
    prev_in_zone = _shift_bool_false(in_zone)
    return in_zone & (~prev_in_zone)


def _crossed_below(series: pd.Series, threshold: float) -> pd.Series:
    curr = series <= threshold
    prev = series.shift(1) > threshold
    out = curr & prev
    return out.where(out.notna(), False).astype(bool)


def _crossed_above(series: pd.Series, threshold: float) -> pd.Series:
    curr = series >= threshold
    prev = series.shift(1) < threshold
    out = curr & prev
    return out.where(out.notna(), False).astype(bool)


def _crossed_into_band(series: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    in_band = ((series >= lower) & (series <= upper))
    in_band = in_band.where(in_band.notna(), False).astype(bool)
    prev_in_band = _shift_bool_false(in_band)
    return in_band & (~prev_in_band)


def _set_exit_signal_from_conditions(
    plan: pd.DataFrame,
    long_exit: pd.Series,
    short_exit: pd.Series,
) -> pd.DataFrame:
    exit_mask = (long_exit | short_exit).where((long_exit | short_exit).notna(), False).astype(bool)
    plan["exit_signal"] = np.where(exit_mask, 1, plan["exit_signal"])
    return plan


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
    exit_z_long: Optional[float] = None,
    exit_z_short: Optional[float] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [z_col])

    plan = _empty_plan(out)

    z = out[z_col]
    long_entry = _crossed_below(z, long_threshold)
    short_entry = _crossed_above(z, short_threshold)

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

    x = out[breakout_col]
    long_entry = _crossed_above(x, long_threshold)
    short_entry = _crossed_below(x, short_threshold)

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
    exit_rsi_long: Optional[float] = None,
    exit_rsi_short: Optional[float] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [rsi_col])

    plan = _empty_plan(out)

    rsi = out[rsi_col]
    long_entry = _crossed_below(rsi, long_threshold)
    short_entry = _crossed_above(rsi, short_threshold)

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

    long_rule = long_rule.where(long_rule.notna(), False).astype(bool)
    short_rule = short_rule.where(short_rule.notna(), False).astype(bool)

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
    exit_z_long: Optional[float] = None,
    exit_z_short: Optional[float] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [z_col, bb_col])

    plan = _empty_plan(out)

    z = out[z_col]
    bb = out[bb_col]

    long_entry = _crossed_below(z, long_z) & _crossed_below(bb, long_bb)
    short_entry = _crossed_above(z, short_z) & _crossed_above(bb, short_bb)

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
    exit_z_long: Optional[float] = None,
    exit_z_short: Optional[float] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [z_col, adx_col])

    plan = _empty_plan(out)

    z = out[z_col]
    adx_ok = out[adx_col] <= adx_max

    long_entry = _crossed_below(z, long_threshold) & adx_ok
    short_entry = _crossed_above(z, short_threshold) & adx_ok

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
    exit_z_long: Optional[float] = None,
    exit_z_short: Optional[float] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [z_col])

    plan = _empty_plan(out)

    z = out[z_col]
    long_entry = _crossed_below(z, long_threshold)
    short_entry = _crossed_above(z, short_threshold)

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
    exit_z_long: Optional[float] = None,
    exit_z_short: Optional[float] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [z_col, adx_col, session_col])

    plan = _empty_plan(out)

    z = out[z_col]
    cond = (out[adx_col] <= adx_max) & (out[session_col] == 1)

    long_entry = _crossed_below(z, long_threshold) & cond
    short_entry = _crossed_above(z, short_threshold) & cond

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
    exit_z_long: Optional[float] = None,
    exit_z_short: Optional[float] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [z_col, adx_col])

    plan = _empty_plan(out)

    z = out[z_col]
    long_entry = _crossed_below(z, long_threshold)
    short_entry = _crossed_above(z, short_threshold)

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    adx = out[adx_col]
    size_arr = np.where(adx <= adx_soft, 1.0, np.where(adx <= adx_hard, 0.5, 0.0))

    entry_mask = long_entry | short_entry
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

    long_state = (out["close"] >= out[high_col] * (1.0 + breakout_buffer)).astype(bool)
    short_state = (out["close"] <= out[low_col] * (1.0 - breakout_buffer)).astype(bool)

    long_entry = long_state & (~_shift_bool_false(long_state))
    short_entry = short_state & (~_shift_bool_false(short_state))

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

    short_state = (
        (out["high"] >= out[high_col] * (1.0 + sweep_buffer))
        & (out["close"] < out[high_col])
    ).astype(bool)
    long_state = (
        (out["low"] <= out[low_col] * (1.0 - sweep_buffer))
        & (out["close"] > out[low_col])
    ).astype(bool)

    short_entry = short_state & (~_shift_bool_false(short_state))
    long_entry = long_state & (~_shift_bool_false(long_state))

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

    long_state = (out["close"] >= out[high_col] * (1.0 + breakout_buffer)).astype(bool)
    short_state = (out["close"] <= out[low_col] * (1.0 - breakout_buffer)).astype(bool)

    long_entry = long_state & (~_shift_bool_false(long_state))
    short_entry = short_state & (~_shift_bool_false(short_state))

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

    short_state = (
        (out["high"] >= out[high_col] * (1.0 + sweep_buffer))
        & (out["close"] < out[high_col])
    ).astype(bool)
    long_state = (
        (out["low"] <= out[low_col] * (1.0 - sweep_buffer))
        & (out["close"] > out[low_col])
    ).astype(bool)

    short_entry = short_state & (~_shift_bool_false(short_state))
    long_entry = long_state & (~_shift_bool_false(long_state))

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

    short_state = (
        (out["high"] >= out[high_col] * (1.0 + sweep_buffer))
        & (out["close"] < out[high_col])
        & (out[volume_col] >= volume_min)
        & (out[ret_col] < 0)
    ).astype(bool)

    long_state = (
        (out["low"] <= out[low_col] * (1.0 - sweep_buffer))
        & (out["close"] > out[low_col])
        & (out[volume_col] >= volume_min)
        & (out[ret_col] > 0)
    ).astype(bool)

    short_entry = short_state & (~_shift_bool_false(short_state))
    long_entry = long_state & (~_shift_bool_false(long_state))

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

    short_state = (
        (out["high"] >= out[high_col] * (1.0 + sweep_buffer))
        & (out["close"] < out[high_col])
        & (out[volume_col] >= volume_min)
    ).astype(bool)

    long_state = (
        (out["low"] <= out[low_col] * (1.0 - sweep_buffer))
        & (out["close"] > out[low_col])
        & (out[volume_col] >= volume_min)
    ).astype(bool)

    short_entry = short_state & (~_shift_bool_false(short_state))
    long_entry = long_state & (~_shift_bool_false(long_state))

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
    exit_z_long: Optional[float] = None,
    exit_z_short: Optional[float] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [z_col, adx_col, support_col, resistance_col])

    plan = _empty_plan(out)

    z = out[z_col]
    near_support = _near_level(out["close"], out[support_col], level_tolerance)
    near_resistance = _near_level(out["close"], out[resistance_col], level_tolerance)

    long_entry = _crossed_below(z, long_threshold) & (out[adx_col] <= adx_max) & near_support
    short_entry = _crossed_above(z, short_threshold) & (out[adx_col] <= adx_max) & near_resistance

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
    exit_bb_long: Optional[float] = None,
    exit_bb_short: Optional[float] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [bb_col, adx_col])

    plan = _empty_plan(out)

    bb = out[bb_col]
    long_entry = _crossed_below(bb, lower_threshold) & (out[adx_col] <= adx_max)
    short_entry = _crossed_above(bb, upper_threshold) & (out[adx_col] <= adx_max)

    if support_col is not None:
        _require_columns(out, [support_col])
        near_support = _near_level(out["close"], out[support_col], level_tolerance)
        long_entry = long_entry & near_support

    if resistance_col is not None:
        _require_columns(out, [resistance_col])
        near_resistance = _near_level(out["close"], out[resistance_col], level_tolerance)
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
    exit_fast_ema: bool = True,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [fast_ema_col, slow_ema_col, adx_col, price_col])

    plan = _empty_plan(out)

    trend_up = (out[fast_ema_col] > out[slow_ema_col]) & (out[adx_col] >= adx_min)
    trend_down = (out[fast_ema_col] < out[slow_ema_col]) & (out[adx_col] >= adx_min)

    pullback_long_prev = out[price_col].shift(1) < out[fast_ema_col].shift(1)
    resume_long = (out[price_col] > out[fast_ema_col]) & (out[price_col].shift(1) <= out[fast_ema_col].shift(1))

    pullback_short_prev = out[price_col].shift(1) > out[fast_ema_col].shift(1)
    resume_short = (out[price_col] < out[fast_ema_col]) & (out[price_col].shift(1) >= out[fast_ema_col].shift(1))

    long_entry = trend_up & pullback_long_prev & resume_long
    short_entry = trend_down & pullback_short_prev & resume_short

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    if exit_fast_ema:
        long_exit = out[price_col] >= out[fast_ema_col]
        short_exit = out[price_col] <= out[fast_ema_col]
        plan = _set_exit_signal_from_conditions(plan, long_exit, short_exit)

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def trend_pullback_reentry_long(
    df: pd.DataFrame,
    fast_ema_col: str = "ema_100",
    slow_ema_col: str = "ema_300",
    adx_col: str = "adx_50",
    price_col: str = "close",
    adx_min: float = 20.0,
    stop_loss_pct: float = 0.0030,
    take_profit_pct: float = 0.0080,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = 28800.0,
    size: float = 1.0,
    exit_fast_ema: bool = True,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [fast_ema_col, slow_ema_col, adx_col, price_col])

    plan = _empty_plan(out)

    trend_up = (out[fast_ema_col] > out[slow_ema_col]) & (out[adx_col] >= adx_min)
    trend_down = (out[fast_ema_col] < out[slow_ema_col]) & (out[adx_col] >= adx_min)

    pullback_long_prev = out[price_col].shift(1) < out[fast_ema_col].shift(1)
    resume_long = (out[price_col] > out[fast_ema_col]) & (out[price_col].shift(1) <= out[fast_ema_col].shift(1))

    pullback_short_prev = out[price_col].shift(1) > out[fast_ema_col].shift(1)
    resume_short = (out[price_col] < out[fast_ema_col]) & (out[price_col].shift(1) >= out[fast_ema_col].shift(1))

    long_entry = trend_up & pullback_long_prev & resume_long
    short_entry = trend_down & pullback_short_prev & resume_short

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    if exit_fast_ema:
        long_exit = out[price_col] >= out[fast_ema_col]
        short_exit = out[price_col] <= out[fast_ema_col]
        plan = _set_exit_signal_from_conditions(plan, long_exit, short_exit)

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def fib_trend_retracement(
    df: pd.DataFrame,
    trend_up_col: str = "ema100_gt_ema300",
    trend_down_col: str = "ema100_lt_ema300",
    fib_prefix: str = "fib_4h",
    range_col: str = "trend_range_4h",
    range_min: float = 0.0040,
    stop_loss_pct: float = 0.0035,
    take_profit_pct: float = 0.0090,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = 43200.0,
    size: float = 1.0,
    exit_on_midpoint: bool = False,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    zone_col = f"{fib_prefix}_in_fib_zone_500_618"
    _cols = [trend_up_col, trend_down_col, zone_col, range_col]
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, _cols + [f"{fib_prefix}_fib_382", f"{fib_prefix}_fib_500", f"{fib_prefix}_fib_618"])

    plan = _empty_plan(out)

    zone_entry = _entered_zone(
        out["close"],
        out[f"{fib_prefix}_fib_618"],
        out[f"{fib_prefix}_fib_500"],
    )

    long_entry = (
        (out[trend_up_col] == 1)
        & zone_entry
        & (out[range_col] >= range_min)
    )

    short_entry = (
        (out[trend_down_col] == 1)
        & zone_entry
        & (out[range_col] >= range_min)
    )

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    if exit_on_midpoint:
        long_exit = out["close"] >= out[f"{fib_prefix}_fib_382"]
        short_exit = out["close"] <= out[f"{fib_prefix}_fib_618"]
        plan = _set_exit_signal_from_conditions(plan, long_exit, short_exit)

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def fib_trend_retracement_rsi(
    df: pd.DataFrame,
    trend_up_col: str = "ema100_gt_ema300",
    trend_down_col: str = "ema100_lt_ema300",
    fib_prefix: str = "fib_4h",
    range_col: str = "trend_range_4h",
    range_min: float = 0.0040,
    rsi_col: str = "rsi_50",
    long_rsi_max: float = 55.0,
    short_rsi_min: float = 45.0,
    stop_loss_pct: float = 0.0035,
    take_profit_pct: float = 0.0090,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = 43200.0,
    size: float = 1.0,
    exit_on_midpoint: bool = False,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(
        out,
        [
            trend_up_col,
            trend_down_col,
            range_col,
            rsi_col,
            f"{fib_prefix}_fib_382",
            f"{fib_prefix}_fib_500",
            f"{fib_prefix}_fib_618",
        ],
    )

    plan = _empty_plan(out)

    zone_entry = _entered_zone(
        out["close"],
        out[f"{fib_prefix}_fib_618"],
        out[f"{fib_prefix}_fib_500"],
    )

    long_entry = (
        (out[trend_up_col] == 1)
        & zone_entry
        & (out[range_col] >= range_min)
        & (out[rsi_col] <= long_rsi_max)
        & (out["ret_1"] > 0)
    )

    short_entry = (
        (out[trend_down_col] == 1)
        & zone_entry
        & (out[range_col] >= range_min)
        & (out[rsi_col] >= short_rsi_min)
        & (out["ret_1"] < 0)
    )

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    if exit_on_midpoint:
        long_exit = out["close"] >= out[f"{fib_prefix}_fib_382"]
        short_exit = out["close"] <= out[f"{fib_prefix}_fib_618"]
        plan = _set_exit_signal_from_conditions(plan, long_exit, short_exit)

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def fib_trend_retracement_structure(
    df: pd.DataFrame,
    trend_up_col: str = "ema100_gt_ema300",
    trend_down_col: str = "ema100_lt_ema300",
    fib_prefix: str = "fib_8h",
    range_col: str = "trend_range_8h",
    range_min: float = 0.0050,
    support_col: str = "prev_session_low",
    resistance_col: str = "prev_session_high",
    level_tolerance: float = 0.0015,
    stop_loss_pct: float = 0.0035,
    take_profit_pct: float = 0.0100,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = 43200.0,
    size: float = 1.0,
    exit_on_midpoint: bool = False,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(
        out,
        [
            trend_up_col,
            trend_down_col,
            range_col,
            support_col,
            resistance_col,
            f"{fib_prefix}_fib_382",
            f"{fib_prefix}_fib_500",
            f"{fib_prefix}_fib_618",
        ],
    )

    plan = _empty_plan(out)

    zone_entry = _entered_zone(
        out["close"],
        out[f"{fib_prefix}_fib_618"],
        out[f"{fib_prefix}_fib_500"],
    )
    near_support = _near_level(out["close"], out[support_col], level_tolerance)
    near_resistance = _near_level(out["close"], out[resistance_col], level_tolerance)

    long_entry = (
        (out[trend_up_col] == 1)
        & zone_entry
        & (out[range_col] >= range_min)
        & near_support
        & (out["ret_1"] > 0)
    )

    short_entry = (
        (out[trend_down_col] == 1)
        & zone_entry
        & (out[range_col] >= range_min)
        & near_resistance
        & (out["ret_1"] < 0)
    )

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    if exit_on_midpoint:
        long_exit = out["close"] >= out[f"{fib_prefix}_fib_382"]
        short_exit = out["close"] <= out[f"{fib_prefix}_fib_618"]
        plan = _set_exit_signal_from_conditions(plan, long_exit, short_exit)

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def fib_deep_retracement_reversal(
    df: pd.DataFrame,
    trend_up_col: str = "ema100_gt_ema300",
    trend_down_col: str = "ema100_lt_ema300",
    fib_prefix: str = "fib_8h",
    range_col: str = "trend_range_8h",
    range_min: float = 0.0060,
    rsi_col: str = "rsi_100",
    long_rsi_max: float = 45.0,
    short_rsi_min: float = 55.0,
    stop_loss_pct: float = 0.0040,
    take_profit_pct: float = 0.0120,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = 57600.0,
    size: float = 1.0,
    exit_on_midpoint: bool = False,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(
        out,
        [
            trend_up_col,
            trend_down_col,
            range_col,
            rsi_col,
            f"{fib_prefix}_fib_500",
            f"{fib_prefix}_fib_618",
            f"{fib_prefix}_fib_786",
        ],
    )

    plan = _empty_plan(out)

    zone_entry = _entered_zone(
        out["close"],
        out[f"{fib_prefix}_fib_786"],
        out[f"{fib_prefix}_fib_618"],
    )

    long_entry = (
        (out[trend_up_col] == 1)
        & zone_entry
        & (out[range_col] >= range_min)
        & (out[rsi_col] <= long_rsi_max)
        & (out["ret_1"] > 0)
    )

    short_entry = (
        (out[trend_down_col] == 1)
        & zone_entry
        & (out[range_col] >= range_min)
        & (out[rsi_col] >= short_rsi_min)
        & (out["ret_1"] < 0)
    )

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    if exit_on_midpoint:
        long_exit = out["close"] >= out[f"{fib_prefix}_fib_500"]
        short_exit = out["close"] <= out[f"{fib_prefix}_fib_618"]
        plan = _set_exit_signal_from_conditions(plan, long_exit, short_exit)

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def fib_trend_retracement_day(
    df: pd.DataFrame,
    trend_up_col: str = "ema300_gt_ema600",
    trend_down_col: str = "ema300_lt_ema600",
    fib_prefix: str = "fib_1d",
    range_col: str = "trend_range_1d",
    range_min: float = 0.0075,
    adx_col: str = "adx_50",
    adx_min: float = 18.0,
    stop_loss_pct: float = 0.0040,
    take_profit_pct: float = 0.0120,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = 86400.0,
    size: float = 1.0,
    exit_on_midpoint: bool = False,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(
        out,
        [
            trend_up_col,
            trend_down_col,
            range_col,
            adx_col,
            f"{fib_prefix}_fib_382",
            f"{fib_prefix}_fib_500",
            f"{fib_prefix}_fib_618",
        ],
    )

    plan = _empty_plan(out)

    zone_entry = _entered_zone(
        out["close"],
        out[f"{fib_prefix}_fib_618"],
        out[f"{fib_prefix}_fib_500"],
    )

    long_entry = (
        (out[trend_up_col] == 1)
        & zone_entry
        & (out[range_col] >= range_min)
        & (out[adx_col] >= adx_min)
        & (out["ret_1"] > 0)
    )

    short_entry = (
        (out[trend_down_col] == 1)
        & zone_entry
        & (out[range_col] >= range_min)
        & (out[adx_col] >= adx_min)
        & (out["ret_1"] < 0)
    )

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    if exit_on_midpoint:
        long_exit = out["close"] >= out[f"{fib_prefix}_fib_382"]
        short_exit = out["close"] <= out[f"{fib_prefix}_fib_618"]
        plan = _set_exit_signal_from_conditions(plan, long_exit, short_exit)

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def bollinger_exhaustion_reversal_long(
    df: pd.DataFrame,
    bb_col: str = "bb_pos",
    adx_col: str = "adx_50",
    resistance_col: Optional[str] = "rolling_high_4h",
    support_col: Optional[str] = "rolling_low_4h",
    upper_threshold: float = 0.98,
    lower_threshold: float = 0.02,
    adx_max: float = 22.0,
    level_tolerance: float = 0.0015,
    stop_loss_pct: float = 0.0030,
    take_profit_pct: float = 0.0080,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = 21600.0,
    size: float = 1.0,
    exit_bb_long: Optional[float] = 0.45,
    exit_bb_short: Optional[float] = 0.55,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [bb_col, adx_col])

    plan = _empty_plan(out)

    bb = out[bb_col]
    long_entry = _crossed_below(bb, lower_threshold) & (out[adx_col] <= adx_max)
    short_entry = _crossed_above(bb, upper_threshold) & (out[adx_col] <= adx_max)

    if support_col is not None:
        _require_columns(out, [support_col])
        long_entry = long_entry & _near_level(out["close"], out[support_col], level_tolerance)

    if resistance_col is not None:
        _require_columns(out, [resistance_col])
        short_entry = short_entry & _near_level(out["close"], out[resistance_col], level_tolerance)

    long_entry = long_entry & (out["ret_1"] > 0)
    short_entry = short_entry & (out["ret_1"] < 0)

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def macd_signal_cross_trend(
    df: pd.DataFrame,
    macd_cross_up_col: str = "macd_cross_up",
    macd_cross_down_col: str = "macd_cross_down",
    trend_up_col: str = "ema100_gt_ema300",
    trend_down_col: str = "ema100_lt_ema300",
    stop_loss_pct: float = 0.0025,
    take_profit_pct: float = 0.0050,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    size: float = 1.0,
    exit_on_opposite_cross: bool = True,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [macd_cross_up_col, macd_cross_down_col, trend_up_col, trend_down_col])

    plan = _empty_plan(out)

    long_entry = (out[macd_cross_up_col] == 1) & (out[trend_up_col] == 1)
    short_entry = (out[macd_cross_down_col] == 1) & (out[trend_down_col] == 1)

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    if exit_on_opposite_cross:
        long_exit = out[macd_cross_down_col] == 1
        short_exit = out[macd_cross_up_col] == 1
        plan = _set_exit_signal_from_conditions(plan, long_exit, short_exit)

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def macd_hist_reversal(
    df: pd.DataFrame,
    macd_hist_norm_col: str = "macd_hist_atr_norm",
    macd_hist_slope_col: str = "macd_hist_slope",
    long_threshold: float = -0.10,
    short_threshold: float = 0.10,
    stop_loss_pct: float = 0.0025,
    take_profit_pct: float = 0.0050,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    size: float = 1.0,
    exit_at_zero: bool = True,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [macd_hist_norm_col, macd_hist_slope_col])

    plan = _empty_plan(out)

    hist = out[macd_hist_norm_col]
    slope = out[macd_hist_slope_col]

    long_entry = _crossed_below(hist, long_threshold) & (slope > 0)
    short_entry = _crossed_above(hist, short_threshold) & (slope < 0)

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    if exit_at_zero:
        long_exit = hist >= 0
        short_exit = hist <= 0
        plan = _set_exit_signal_from_conditions(plan, long_exit, short_exit)

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def macd_rsi_confirmation(
    df: pd.DataFrame,
    macd_cross_up_col: str = "macd_cross_up",
    macd_cross_down_col: str = "macd_cross_down",
    rsi_col: str = "rsi_50",
    long_rsi_max: float = 50.0,
    short_rsi_min: float = 50.0,
    stop_loss_pct: float = 0.0025,
    take_profit_pct: float = 0.0050,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    size: float = 1.0,
    exit_on_opposite_cross: bool = True,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [macd_cross_up_col, macd_cross_down_col, rsi_col])

    plan = _empty_plan(out)

    long_entry = (out[macd_cross_up_col] == 1) & (out[rsi_col] <= long_rsi_max)
    short_entry = (out[macd_cross_down_col] == 1) & (out[rsi_col] >= short_rsi_min)

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    if exit_on_opposite_cross:
        long_exit = out[macd_cross_down_col] == 1
        short_exit = out[macd_cross_up_col] == 1
        plan = _set_exit_signal_from_conditions(plan, long_exit, short_exit)

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan


def macd_fib_retracement_confirmation(
    df: pd.DataFrame,
    macd_hist_slope_col: str = "macd_hist_slope",
    fib_zone_col: str = "fib_4h_in_fib_zone_500_618",
    trend_up_col: str = "ema100_gt_ema300",
    trend_down_col: str = "ema100_lt_ema300",
    stop_loss_pct: float = 0.0030,
    take_profit_pct: float = 0.0060,
    max_hold_bars: Optional[int] = None,
    max_hold_seconds: Optional[float] = None,
    size: float = 1.0,
    exit_on_slope_flip: bool = True,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    _require_columns(out, [macd_hist_slope_col, fib_zone_col, trend_up_col, trend_down_col])

    plan = _empty_plan(out)

    zone = out[fib_zone_col] == 1
    zone_entry = zone & (~_shift_bool_false(zone))
    slope = out[macd_hist_slope_col]

    long_entry = zone_entry & (out[trend_up_col] == 1) & (slope > 0)
    short_entry = zone_entry & (out[trend_down_col] == 1) & (slope < 0)

    plan.loc[long_entry, "entry_signal"] = 1
    plan.loc[short_entry, "entry_signal"] = -1

    if exit_on_slope_flip:
        long_exit = slope <= 0
        short_exit = slope >= 0
        plan = _set_exit_signal_from_conditions(plan, long_exit, short_exit)

    entry_mask = long_entry | short_entry
    _set_trade_params(plan, entry_mask, stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size)

    return plan