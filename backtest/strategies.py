from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd


Condition = Callable[[pd.DataFrame], pd.Series]
ValueLike = Union[str, float, int]


@dataclass(frozen=True)
class TradeParams:
    stop_loss_pct: float
    take_profit_pct: float
    max_hold_bars: Optional[int] = None
    max_hold_seconds: Optional[float] = None
    size: float = 1.0


@dataclass(frozen=True)
class EntryGate:
    trade_start: Optional[str] = None
    trade_end: Optional[str] = None
    cooldown_bars: Optional[int] = None
    max_trades_per_day: Optional[int] = None


def _ny_local_ts(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts, errors="coerce")
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize("UTC")
    return s.dt.tz_convert("America/New_York")


def _ny_time_window_mask(
    ts: pd.Series,
    start_time: str = "09:30",
    end_time: str = "11:30",
) -> pd.Series:
    local = _ny_local_ts(ts)
    mins = local.dt.hour * 60 + local.dt.minute

    sh, sm = map(int, start_time.split(":"))
    eh, em = map(int, end_time.split(":"))
    start_min = sh * 60 + sm
    end_min = eh * 60 + em

    mask = (mins >= start_min) & (mins < end_min)
    return mask.where(mask.notna(), False).astype(bool)


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
    sort_cols = ([symbol_col] if symbol_col else []) + [ts_col]
    out = out.dropna(subset=[ts_col]).sort_values(sort_cols).reset_index(drop=True)
    return out, ts_col, symbol_col


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


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


def _set_trade_params(plan: pd.DataFrame, entry_mask: pd.Series, params: TradeParams) -> None:
    plan.loc[entry_mask, "stop_loss_pct"] = params.stop_loss_pct
    plan.loc[entry_mask, "take_profit_pct"] = params.take_profit_pct
    if params.max_hold_bars is not None:
        plan.loc[entry_mask, "max_hold_bars"] = params.max_hold_bars
    if params.max_hold_seconds is not None:
        plan.loc[entry_mask, "max_hold_seconds"] = params.max_hold_seconds
    plan.loc[entry_mask, "size"] = params.size


def _ensure_bool(s: pd.Series) -> pd.Series:
    return s.where(s.notna(), False).astype(bool)


def _shift_bool_false(s: pd.Series) -> pd.Series:
    return _ensure_bool(s.shift(1))


def _resolve_series(df: pd.DataFrame, value: ValueLike) -> pd.Series:
    if isinstance(value, str) and value in df.columns:
        return df[value]
    return pd.Series(value, index=df.index, dtype="float64")


def _resolve_threshold(df: pd.DataFrame, value: ValueLike):
    if isinstance(value, str) and value in df.columns:
        return df[value]
    return value


def _crossed_above_series(left: pd.Series, right: pd.Series) -> pd.Series:
    out = (left >= right) & (left.shift(1) < right.shift(1))
    return _ensure_bool(out)


def _crossed_below_series(left: pd.Series, right: pd.Series) -> pd.Series:
    out = (left <= right) & (left.shift(1) > right.shift(1))
    return _ensure_bool(out)


def _crossed_above_threshold(series: pd.Series, threshold: float) -> pd.Series:
    out = (series >= threshold) & (series.shift(1) < threshold)
    return _ensure_bool(out)


def _crossed_below_threshold(series: pd.Series, threshold: float) -> pd.Series:
    out = (series <= threshold) & (series.shift(1) > threshold)
    return _ensure_bool(out)


def _entered_band(series: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    in_band = _ensure_bool((series >= lower) & (series <= upper))
    return in_band & (~_shift_bool_false(in_band))


def _near_level(close: pd.Series, level: pd.Series, tolerance: float) -> pd.Series:
    out = (close - level).abs() / close.replace(0, np.nan) <= tolerance
    return _ensure_bool(out)


def _group_keys(df: pd.DataFrame, ts_col: str, symbol_col: Optional[str]) -> list[str]:
    local = _ny_local_ts(df[ts_col])
    day_key = local.dt.strftime("%Y-%m-%d")
    if symbol_col is None:
        return day_key.to_list()
    return (df[symbol_col].astype(str) + "|" + day_key.astype(str)).to_list()


def _calc_session_vwap(df: pd.DataFrame, ts_col: str, symbol_col: Optional[str]) -> pd.Series:
    price = None
    if all(c in df.columns for c in ["high", "low", "close"]):
        price = (df["high"] + df["low"] + df["close"]) / 3.0
    else:
        price = df["close"]

    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    keys = pd.Series(_group_keys(df, ts_col, symbol_col), index=df.index)
    pv = price * vol
    cum_pv = pv.groupby(keys).cumsum()
    cum_v = vol.groupby(keys).cumsum().replace(0, np.nan)
    return cum_pv / cum_v


def _calc_ema(df: pd.DataFrame, col: str, span: int) -> pd.Series:
    return df[col].ewm(span=span, adjust=False).mean()


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_adx_filter(df: pd.DataFrame, min_value: Optional[float]) -> pd.Series:
    if min_value is None:
        return pd.Series(True, index=df.index)
    adx_col = _first_existing(df, ["adx_14", "adx", "adx_20"])
    if adx_col is None:
        return pd.Series(True, index=df.index)
    return _ensure_bool(df[adx_col] >= min_value)


def _get_rel_volume_filter(df: pd.DataFrame, min_value: Optional[float]) -> pd.Series:
    if min_value is None or "volume" not in df.columns:
        return pd.Series(True, index=df.index)
    rel_col = _first_existing(df, ["rel_volume_20", "rel_volume", "volume_ratio_20"])
    if rel_col is not None:
        return _ensure_bool(df[rel_col] >= min_value)
    vol_mean = df["volume"].rolling(20, min_periods=20).mean()
    rel = df["volume"] / vol_mean.replace(0, np.nan)
    return _ensure_bool(rel >= min_value)


def _is_opening_range_complete(local_ts: pd.Series, opening_range_minutes: int) -> pd.Series:
    mins = local_ts.dt.hour * 60 + local_ts.dt.minute
    cutoff = 9 * 60 + 30 + opening_range_minutes
    return _ensure_bool(mins >= cutoff)


def _opening_range_levels(
    df: pd.DataFrame,
    ts_col: str,
    symbol_col: Optional[str],
    opening_range_minutes: int,
) -> tuple[pd.Series, pd.Series]:
    _require_columns(df, ["high", "low"])

    local = _ny_local_ts(df[ts_col])
    mins = local.dt.hour * 60 + local.dt.minute
    in_or = (mins >= 9 * 60 + 30) & (mins < 9 * 60 + 30 + opening_range_minutes)

    group_key = pd.Series(_group_keys(df, ts_col, symbol_col), index=df.index)

    or_high = pd.Series(np.nan, index=df.index)
    or_low = pd.Series(np.nan, index=df.index)

    for _, idx in group_key.groupby(group_key).groups.items():
        idx = list(idx)
        mask = in_or.iloc[idx]
        if mask.sum() == 0:
            continue
        high_val = df.iloc[idx][mask]["high"].max()
        low_val = df.iloc[idx][mask]["low"].min()
        or_high.iloc[idx] = high_val
        or_low.iloc[idx] = low_val

    return or_high.ffill(), or_low.ffill()


def _donchian_levels(
    df: pd.DataFrame,
    lookback_bars: int,
    symbol_col: Optional[str],
) -> tuple[pd.Series, pd.Series]:
    _require_columns(df, ["high", "low"])

    if symbol_col is None:
        upper = df["high"].rolling(lookback_bars, min_periods=lookback_bars).max().shift(1)
        lower = df["low"].rolling(lookback_bars, min_periods=lookback_bars).min().shift(1)
        return upper, lower

    upper_parts = []
    lower_parts = []
    for _, g in df.groupby(symbol_col, sort=False):
        upper_parts.append(g["high"].rolling(lookback_bars, min_periods=lookback_bars).max().shift(1))
        lower_parts.append(g["low"].rolling(lookback_bars, min_periods=lookback_bars).min().shift(1))
    upper = pd.concat(upper_parts).sort_index()
    lower = pd.concat(lower_parts).sort_index()
    return upper, lower


def _gate_signed_entries(
    long_mask: pd.Series,
    short_mask: pd.Series,
    ts: pd.Series,
    gate: Optional[EntryGate],
) -> pd.Series:
    long_mask = _ensure_bool(long_mask)
    short_mask = _ensure_bool(short_mask)

    entry_signal = pd.Series(0, index=long_mask.index, dtype="int64")
    entry_signal.loc[long_mask & ~short_mask] = 1
    entry_signal.loc[short_mask & ~long_mask] = -1

    if gate is None:
        return entry_signal

    active = entry_signal != 0

    if gate.trade_start is not None and gate.trade_end is not None:
        active = active & _ny_time_window_mask(ts, gate.trade_start, gate.trade_end)

    local = _ny_local_ts(ts)
    day_keys = local.dt.strftime("%Y-%m-%d").to_list()

    out = pd.Series(0, index=entry_signal.index, dtype="int64")
    counts: dict[str, int] = {}
    last_fire = -10**12

    for i, proposed in enumerate(entry_signal.to_numpy()):
        if proposed == 0 or not bool(active.iloc[i]):
            continue

        if gate.cooldown_bars is not None and gate.cooldown_bars > 0:
            if i - last_fire <= gate.cooldown_bars:
                continue

        if gate.max_trades_per_day is not None and gate.max_trades_per_day > 0:
            day = day_keys[i]
            cnt = counts.get(day, 0)
            if cnt >= gate.max_trades_per_day:
                continue
            counts[day] = cnt + 1

        out.iloc[i] = proposed
        last_fire = i

    return out


def require_columns(*cols: str) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        _require_columns(df, list(cols))
        return pd.Series(True, index=df.index)
    return _cond


def col_eq(col: str, value: ValueLike) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        _require_columns(df, [col])
        return _ensure_bool(df[col] == _resolve_series(df, value))
    return _cond


def col_gt(col: str, value: ValueLike) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        _require_columns(df, [col])
        return _ensure_bool(df[col] > _resolve_series(df, value))
    return _cond


def col_gte(col: str, value: ValueLike) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        _require_columns(df, [col])
        return _ensure_bool(df[col] >= _resolve_series(df, value))
    return _cond


def col_lt(col: str, value: ValueLike) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        _require_columns(df, [col])
        return _ensure_bool(df[col] < _resolve_series(df, value))
    return _cond


def col_lte(col: str, value: ValueLike) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        _require_columns(df, [col])
        return _ensure_bool(df[col] <= _resolve_series(df, value))
    return _cond


def abs_col_gte(col: str, value: float) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        _require_columns(df, [col])
        return _ensure_bool(df[col].abs() >= value)
    return _cond


def crossed_above(col: str, value: ValueLike) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        _require_columns(df, [col] + ([value] if isinstance(value, str) and value in df.columns else []))
        threshold = _resolve_threshold(df, value)
        if isinstance(threshold, pd.Series):
            return _crossed_above_series(df[col], threshold)
        return _crossed_above_threshold(df[col], float(threshold))
    return _cond


def crossed_below(col: str, value: ValueLike) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        _require_columns(df, [col] + ([value] if isinstance(value, str) and value in df.columns else []))
        threshold = _resolve_threshold(df, value)
        if isinstance(threshold, pd.Series):
            return _crossed_below_series(df[col], threshold)
        return _crossed_below_threshold(df[col], float(threshold))
    return _cond


def entered_band(col: str, lower: ValueLike, upper: ValueLike) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        needed = [col]
        if isinstance(lower, str) and lower in df.columns:
            needed.append(lower)
        if isinstance(upper, str) and upper in df.columns:
            needed.append(upper)
        _require_columns(df, needed)
        return _entered_band(df[col], _resolve_series(df, lower), _resolve_series(df, upper))
    return _cond


def near_level(price_col: str, level_col: str, tolerance: float) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        _require_columns(df, [price_col, level_col])
        return _near_level(df[price_col], df[level_col], tolerance)
    return _cond


def state_turns_true(cond: Condition) -> Condition:
    def _wrapped(df: pd.DataFrame) -> pd.Series:
        state = _ensure_bool(cond(df))
        return state & (~_shift_bool_false(state))
    return _wrapped


def and_(*conditions: Condition) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        if not conditions:
            return pd.Series(True, index=df.index)
        out = pd.Series(True, index=df.index)
        for c in conditions:
            out = out & _ensure_bool(c(df))
        return _ensure_bool(out)
    return _cond


def or_(*conditions: Condition) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        if not conditions:
            return pd.Series(False, index=df.index)
        out = pd.Series(False, index=df.index)
        for c in conditions:
            out = out | _ensure_bool(c(df))
        return _ensure_bool(out)
    return _cond


def not_(condition: Condition) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        return _ensure_bool(~_ensure_bool(condition(df)))
    return _cond


def build_plan(
    df: pd.DataFrame,
    *,
    long_entry: Condition,
    short_entry: Condition,
    trade_params: TradeParams,
    gate: Optional[EntryGate] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    plan = _empty_plan(out)

    long_mask = _ensure_bool(long_entry(out))
    short_mask = _ensure_bool(short_entry(out))
    signed_entries = _gate_signed_entries(long_mask, short_mask, out[ts_col], gate)

    plan["entry_signal"] = signed_entries
    entry_mask = plan["entry_signal"] != 0
    _set_trade_params(plan, entry_mask, trade_params)
    return plan


def confluence_strategy(
    df: pd.DataFrame,
    *,
    long_conditions: list[Condition],
    short_conditions: list[Condition],
    trade_params: TradeParams,
    gate: Optional[EntryGate] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    return build_plan(
        df,
        long_entry=and_(*long_conditions),
        short_entry=and_(*short_conditions),
        trade_params=trade_params,
        gate=gate,
        ts_col=ts_col,
        symbol_col=symbol_col,
    )


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
    params = TradeParams(
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_hold_bars=max_hold_bars,
        max_hold_seconds=max_hold_seconds,
        size=size,
    )
    return build_plan(
        df,
        long_entry=lambda x: long_rule,
        short_entry=lambda x: short_rule,
        trade_params=params,
    )


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
    trade_start: Optional[str] = None,
    trade_end: Optional[str] = None,
    cooldown_bars: Optional[int] = None,
    max_trades_per_day: Optional[int] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    return confluence_strategy(
        df,
        long_conditions=[crossed_below(z_col, long_threshold)],
        short_conditions=[crossed_above(z_col, short_threshold)],
        trade_params=TradeParams(stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size),
        gate=EntryGate(trade_start, trade_end, cooldown_bars, max_trades_per_day),
        ts_col=ts_col,
        symbol_col=symbol_col,
    )


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
    trade_start: Optional[str] = None,
    trade_end: Optional[str] = None,
    cooldown_bars: Optional[int] = None,
    max_trades_per_day: Optional[int] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    return confluence_strategy(
        df,
        long_conditions=[crossed_below(rsi_col, long_threshold)],
        short_conditions=[crossed_above(rsi_col, short_threshold)],
        trade_params=TradeParams(stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size),
        gate=EntryGate(trade_start, trade_end, cooldown_bars, max_trades_per_day),
        ts_col=ts_col,
        symbol_col=symbol_col,
    )


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
    trade_start: Optional[str] = None,
    trade_end: Optional[str] = None,
    cooldown_bars: Optional[int] = None,
    max_trades_per_day: Optional[int] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    return confluence_strategy(
        df,
        long_conditions=[col_eq(macd_cross_up_col, 1), col_eq(trend_up_col, 1)],
        short_conditions=[col_eq(macd_cross_down_col, 1), col_eq(trend_down_col, 1)],
        trade_params=TradeParams(stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size),
        gate=EntryGate(trade_start, trade_end, cooldown_bars, max_trades_per_day),
        ts_col=ts_col,
        symbol_col=symbol_col,
    )


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
    trade_start: Optional[str] = None,
    trade_end: Optional[str] = None,
    cooldown_bars: Optional[int] = None,
    max_trades_per_day: Optional[int] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    return confluence_strategy(
        df,
        long_conditions=[col_eq(macd_cross_up_col, 1), col_lte(rsi_col, long_rsi_max)],
        short_conditions=[col_eq(macd_cross_down_col, 1), col_gte(rsi_col, short_rsi_min)],
        trade_params=TradeParams(stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size),
        gate=EntryGate(trade_start, trade_end, cooldown_bars, max_trades_per_day),
        ts_col=ts_col,
        symbol_col=symbol_col,
    )


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
    trade_start: Optional[str] = None,
    trade_end: Optional[str] = None,
    cooldown_bars: Optional[int] = None,
    max_trades_per_day: Optional[int] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    return confluence_strategy(
        df,
        long_conditions=[crossed_below(macd_hist_norm_col, long_threshold), col_gt(macd_hist_slope_col, 0)],
        short_conditions=[crossed_above(macd_hist_norm_col, short_threshold), col_lt(macd_hist_slope_col, 0)],
        trade_params=TradeParams(stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size),
        gate=EntryGate(trade_start, trade_end, cooldown_bars, max_trades_per_day),
        ts_col=ts_col,
        symbol_col=symbol_col,
    )


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
    trade_start: Optional[str] = None,
    trade_end: Optional[str] = None,
    cooldown_bars: Optional[int] = None,
    max_trades_per_day: Optional[int] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    fib_382 = f"{fib_prefix}_fib_382"
    fib_500 = f"{fib_prefix}_fib_500"
    fib_618 = f"{fib_prefix}_fib_618"

    return confluence_strategy(
        df,
        long_conditions=[
            col_eq(trend_up_col, 1),
            entered_band("close", fib_618, fib_500),
            col_gte(range_col, range_min),
            col_lte(rsi_col, long_rsi_max),
            col_gt("ret_1", 0),
        ],
        short_conditions=[
            col_eq(trend_down_col, 1),
            entered_band("close", fib_618, fib_500),
            col_gte(range_col, range_min),
            col_gte(rsi_col, short_rsi_min),
            col_lt("ret_1", 0),
        ],
        trade_params=TradeParams(stop_loss_pct, take_profit_pct, max_hold_bars, max_hold_seconds, size),
        gate=EntryGate(trade_start, trade_end, cooldown_bars, max_trades_per_day),
        ts_col=ts_col,
        symbol_col=symbol_col,
    )


def ema600_adx50_high_conviction(
    df: pd.DataFrame,
    trade_params: TradeParams = TradeParams(
        stop_loss_pct=0.0032,
        take_profit_pct=0.0058,
        max_hold_seconds=14400.0,
        size=1.0,
    ),
    gate: Optional[EntryGate] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    df2, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)

    if "ema600" not in df2.columns:
        df2["ema600"] = _calc_ema(df2, "close", 600)

    if "price_vs_ema600" not in df2.columns:
        df2["price_vs_ema600"] = df2["close"] / df2["ema600"] - 1.0

    adx_col = _first_existing(df2, ["adx_14", "adx", "adx_20"])

    long_conditions = [
        crossed_above("close", "ema600"),
        col_gt("price_vs_ema600", 0.0),
    ]
    short_conditions = [
        crossed_below("close", "ema600"),
        col_lt("price_vs_ema600", 0.0),
    ]

    if adx_col is not None:
        long_conditions.append(col_gte(adx_col, 50.0))
        short_conditions.append(col_gte(adx_col, 50.0))

    return confluence_strategy(
        df2,
        long_conditions=long_conditions,
        short_conditions=short_conditions,
        trade_params=trade_params,
        gate=gate,
        ts_col=ts_col,
        symbol_col=symbol_col,
    )


def opening_range_breakout(
    df: pd.DataFrame,
    opening_range_minutes: int = 30,
    breakout_buffer_pct: float = 0.00025,
    confirm_trend: bool = True,
    confirm_vwap: bool = True,
    confirm_adx_min: Optional[float] = 18.0,
    trade_params: TradeParams = TradeParams(
        stop_loss_pct=0.0022,
        take_profit_pct=0.0055,
        max_hold_seconds=5400.0,
        size=1.0,
    ),
    gate: Optional[EntryGate] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    plan = _empty_plan(out)

    local = _ny_local_ts(out[ts_col])
    or_high, or_low = _opening_range_levels(out, ts_col, symbol_col, opening_range_minutes)
    or_done = _is_opening_range_complete(local, opening_range_minutes)

    vwap = _calc_session_vwap(out, ts_col, symbol_col)
    adx_ok = _get_adx_filter(out, confirm_adx_min)

    trend_up = pd.Series(True, index=out.index)
    trend_down = pd.Series(True, index=out.index)
    if confirm_trend:
        if "ema50" not in out.columns:
            out["ema50"] = _calc_ema(out, "close", 50)
        if "ema200" not in out.columns:
            out["ema200"] = _calc_ema(out, "close", 200)
        trend_up = _ensure_bool(out["ema50"] > out["ema200"])
        trend_down = _ensure_bool(out["ema50"] < out["ema200"])

    vwap_long = pd.Series(True, index=out.index)
    vwap_short = pd.Series(True, index=out.index)
    if confirm_vwap:
        vwap_long = _ensure_bool(out["close"] > vwap)
        vwap_short = _ensure_bool(out["close"] < vwap)

    long_break = _crossed_above_series(out["close"], or_high * (1.0 + breakout_buffer_pct))
    short_break = _crossed_below_series(out["close"], or_low * (1.0 - breakout_buffer_pct))

    long_mask = long_break & or_done & trend_up & vwap_long & adx_ok
    short_mask = short_break & or_done & trend_down & vwap_short & adx_ok

    signed_entries = _gate_signed_entries(long_mask, short_mask, out[ts_col], gate)
    plan["entry_signal"] = signed_entries

    entry_mask = plan["entry_signal"] != 0
    _set_trade_params(plan, entry_mask, trade_params)
    return plan


def opening_range_breakout_retest(
    df: pd.DataFrame,
    opening_range_minutes: int = 30,
    retest_tolerance_pct: float = 0.0008,
    confirm_trend: bool = True,
    confirm_adx_min: Optional[float] = 18.0,
    trade_params: TradeParams = TradeParams(
        stop_loss_pct=0.0022,
        take_profit_pct=0.0055,
        max_hold_seconds=5400.0,
        size=1.0,
    ),
    gate: Optional[EntryGate] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    plan = _empty_plan(out)

    local = _ny_local_ts(out[ts_col])
    or_high, or_low = _opening_range_levels(out, ts_col, symbol_col, opening_range_minutes)
    or_done = _is_opening_range_complete(local, opening_range_minutes)

    trend_up = pd.Series(True, index=out.index)
    trend_down = pd.Series(True, index=out.index)
    if confirm_trend:
        if "ema50" not in out.columns:
            out["ema50"] = _calc_ema(out, "close", 50)
        if "ema200" not in out.columns:
            out["ema200"] = _calc_ema(out, "close", 200)
        trend_up = _ensure_bool(out["ema50"] > out["ema200"])
        trend_down = _ensure_bool(out["ema50"] < out["ema200"])

    adx_ok = _get_adx_filter(out, confirm_adx_min)

    broke_above = _ensure_bool(out["close"].shift(1) > or_high.shift(1))
    broke_below = _ensure_bool(out["close"].shift(1) < or_low.shift(1))

    long_retest = broke_above & _near_level(out["low"], or_high, retest_tolerance_pct) & _ensure_bool(out["close"] > or_high)
    short_retest = broke_below & _near_level(out["high"], or_low, retest_tolerance_pct) & _ensure_bool(out["close"] < or_low)

    long_mask = long_retest & or_done & trend_up & adx_ok
    short_mask = short_retest & or_done & trend_down & adx_ok

    signed_entries = _gate_signed_entries(long_mask, short_mask, out[ts_col], gate)
    plan["entry_signal"] = signed_entries

    entry_mask = plan["entry_signal"] != 0
    _set_trade_params(plan, entry_mask, trade_params)
    return plan


def donchian_breakout_adx(
    df: pd.DataFrame,
    lookback_bars: int = 60,
    adx_min: Optional[float] = 20.0,
    rel_volume_min: Optional[float] = 1.05,
    trade_params: TradeParams = TradeParams(
        stop_loss_pct=0.0022,
        take_profit_pct=0.0055,
        max_hold_seconds=5400.0,
        size=1.0,
    ),
    gate: Optional[EntryGate] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    plan = _empty_plan(out)

    upper, lower = _donchian_levels(out, lookback_bars, symbol_col)
    adx_ok = _get_adx_filter(out, adx_min)
    vol_ok = _get_rel_volume_filter(out, rel_volume_min)

    if "ema50" not in out.columns:
        out["ema50"] = _calc_ema(out, "close", 50)
    if "ema200" not in out.columns:
        out["ema200"] = _calc_ema(out, "close", 200)

    trend_up = _ensure_bool(out["ema50"] > out["ema200"])
    trend_down = _ensure_bool(out["ema50"] < out["ema200"])

    long_mask = _crossed_above_series(out["close"], upper) & trend_up & adx_ok & vol_ok
    short_mask = _crossed_below_series(out["close"], lower) & trend_down & adx_ok & vol_ok

    signed_entries = _gate_signed_entries(long_mask, short_mask, out[ts_col], gate)
    plan["entry_signal"] = signed_entries

    entry_mask = plan["entry_signal"] != 0
    _set_trade_params(plan, entry_mask, trade_params)
    return plan


def ema_slope_momentum_pullback(
    df: pd.DataFrame,
    pullback_to_ema: str = "ema20",
    trend_fast_col: str = "ema50",
    trend_slow_col: str = "ema200",
    slope_ema_col: str = "ema50",
    slope_lookback: int = 10,
    slope_min_pct: float = 0.00035,
    rsi_long_min: float = 52.0,
    rsi_short_max: float = 48.0,
    adx_min: Optional[float] = 16.0,
    pullback_tolerance_pct: float = 0.0009,
    trade_params: TradeParams = TradeParams(
        stop_loss_pct=0.0020,
        take_profit_pct=0.0045,
        max_hold_seconds=3600.0,
        size=1.0,
    ),
    gate: Optional[EntryGate] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    plan = _empty_plan(out)

    for col, span in [(pullback_to_ema, 20), (trend_fast_col, 50), (trend_slow_col, 200), (slope_ema_col, 50)]:
        if col not in out.columns and col.startswith("ema"):
            try:
                span_int = int(col.replace("ema", ""))
                out[col] = _calc_ema(out, "close", span_int)
            except Exception:
                out[col] = _calc_ema(out, "close", span)

    _require_columns(out, [pullback_to_ema, trend_fast_col, trend_slow_col, "close"])

    slope_pct = out[slope_ema_col] / out[slope_ema_col].shift(slope_lookback) - 1.0
    adx_ok = _get_adx_filter(out, adx_min)

    rsi_col = _first_existing(out, ["rsi_14", "rsi_50", "rsi"])
    rsi_long_ok = pd.Series(True, index=out.index)
    rsi_short_ok = pd.Series(True, index=out.index)
    if rsi_col is not None:
        rsi_long_ok = _ensure_bool(out[rsi_col] >= rsi_long_min)
        rsi_short_ok = _ensure_bool(out[rsi_col] <= rsi_short_max)

    trend_up = _ensure_bool(out[trend_fast_col] > out[trend_slow_col])
    trend_down = _ensure_bool(out[trend_fast_col] < out[trend_slow_col])

    near_pullback = _near_level(out["close"], out[pullback_to_ema], pullback_tolerance_pct)
    close_reclaim_up = _crossed_above_series(out["close"], out[pullback_to_ema])
    close_reclaim_down = _crossed_below_series(out["close"], out[pullback_to_ema])

    long_mask = trend_up & _ensure_bool(slope_pct >= slope_min_pct) & near_pullback & close_reclaim_up & rsi_long_ok & adx_ok
    short_mask = trend_down & _ensure_bool(slope_pct <= -slope_min_pct) & near_pullback & close_reclaim_down & rsi_short_ok & adx_ok

    signed_entries = _gate_signed_entries(long_mask, short_mask, out[ts_col], gate)
    plan["entry_signal"] = signed_entries

    entry_mask = plan["entry_signal"] != 0
    _set_trade_params(plan, entry_mask, trade_params)
    return plan


def vwap_trend_pullback(
    df: pd.DataFrame,
    trend_fast_col: str = "ema50",
    trend_slow_col: str = "ema200",
    rsi_long_min: float = 52.0,
    rsi_short_max: float = 48.0,
    adx_min: Optional[float] = 16.0,
    vwap_tolerance_pct: float = 0.0009,
    trade_params: TradeParams = TradeParams(
        stop_loss_pct=0.0020,
        take_profit_pct=0.0045,
        max_hold_seconds=3600.0,
        size=1.0,
    ),
    gate: Optional[EntryGate] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out, ts_col, symbol_col = _prepare(df, ts_col=ts_col, symbol_col=symbol_col)
    plan = _empty_plan(out)

    for col, span in [(trend_fast_col, 50), (trend_slow_col, 200)]:
        if col not in out.columns and col.startswith("ema"):
            try:
                span_int = int(col.replace("ema", ""))
                out[col] = _calc_ema(out, "close", span_int)
            except Exception:
                out[col] = _calc_ema(out, "close", span)

    vwap = _calc_session_vwap(out, ts_col, symbol_col)
    adx_ok = _get_adx_filter(out, adx_min)

    rsi_col = _first_existing(out, ["rsi_14", "rsi_50", "rsi"])
    rsi_long_ok = pd.Series(True, index=out.index)
    rsi_short_ok = pd.Series(True, index=out.index)
    if rsi_col is not None:
        rsi_long_ok = _ensure_bool(out[rsi_col] >= rsi_long_min)
        rsi_short_ok = _ensure_bool(out[rsi_col] <= rsi_short_max)

    trend_up = _ensure_bool(out[trend_fast_col] > out[trend_slow_col])
    trend_down = _ensure_bool(out[trend_fast_col] < out[trend_slow_col])

    near_vwap = _near_level(out["close"], vwap, vwap_tolerance_pct)
    reclaim_up = _crossed_above_series(out["close"], vwap)
    reclaim_down = _crossed_below_series(out["close"], vwap)

    long_mask = trend_up & near_vwap & reclaim_up & rsi_long_ok & adx_ok
    short_mask = trend_down & near_vwap & reclaim_down & rsi_short_ok & adx_ok

    signed_entries = _gate_signed_entries(long_mask, short_mask, out[ts_col], gate)
    plan["entry_signal"] = signed_entries

    entry_mask = plan["entry_signal"] != 0
    _set_trade_params(plan, entry_mask, trade_params)
    return plan

# ===========================================================================
# ML-INFORMED STRATEGIES  (derived from LightGBM feature importance analysis)
# Key ML findings:
#   1. Rolling-range distances (dist_rolling_low/high_60m/2h/4h) dominate
#   2. Time of day (minute_of_day_utc) is critical — gate to 09:00-13:30 NY
#   3. Fibonacci levels (fib_4h/8h dist_382, 618, 786) consistently strong
#   4. Long-horizon trend (ema_1200 vs ema_3600, price_vs_ema3600) matters
#   5. Session levels (dist_prev_session_high/low, dist_or_high/low_15m) useful
#   6. Volume features have near-zero permutation importance — not used here
#   7. Momentum (RSI/MACD) weaker than expected — used only as light filter
#
# All strategies target 2-3 trades/day with 15-120 min hold times.
# Cooldown = 3600 bars (1 hour) between entries on 1-second bar data.
# ===========================================================================


def rolling_range_bounce(
    df: pd.DataFrame,
    *,
    range_label_near: str = "60m",
    near_threshold: float = 0.0025,
    long_trend_filter: float = -0.003,
    short_trend_filter: float = 0.003,
    adx_max: float = 38.0,
    trade_params: Optional[TradeParams] = None,
    gate: Optional[EntryGate] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    """Mean-reversion off 60-min rolling range extremes filtered by long-horizon trend.

    ML rationale: dist_rolling_low_60m / dist_rolling_high_60m are top permutation
    features at the 300s horizon.  Fade near-extremes in a ranging regime (low ADX)
    with the 60-min EMA acting as a trend filter.

    dist_rolling_low_60m  = close / rolling_low_60m  - 1  (> 0 = above the low)
    dist_rolling_high_60m = close / rolling_high_60m - 1  (near 0 = near the high)
    """
    if trade_params is None:
        trade_params = TradeParams(
            stop_loss_pct=0.0018,
            take_profit_pct=0.0038,
            max_hold_seconds=2700.0,
        )
    if gate is None:
        gate = EntryGate(trade_start="09:00", trade_end="14:00",
                         cooldown_bars=3600, max_trades_per_day=3)

    dist_low  = f"dist_rolling_low_{range_label_near}"
    dist_high = f"dist_rolling_high_{range_label_near}"
    adx_col   = _first_existing(df, ["adx_50", "adx_100", "adx_14"])

    long_conds: list[Condition] = [
        col_lte(dist_low, near_threshold),
        col_gte(dist_low, 0.0),
        col_gte("price_vs_ema3600", long_trend_filter),
    ]
    short_conds: list[Condition] = [
        col_lte(dist_high, near_threshold),
        col_gte(dist_high, 0.0),
        col_lte("price_vs_ema3600", short_trend_filter),
    ]
    if adx_col is not None:
        long_conds.append(col_lte(adx_col, adx_max))
        short_conds.append(col_lte(adx_col, adx_max))

    return confluence_strategy(
        df,
        long_conditions=long_conds,
        short_conditions=short_conds,
        trade_params=trade_params,
        gate=gate,
        ts_col=ts_col,
        symbol_col=symbol_col,
    )


def fib_golden_zone_trend(
    df: pd.DataFrame,
    *,
    fib_prefix: str = "fib_4h",
    range_min: float = 0.0035,
    range_pos_long_max: float = 0.55,
    range_pos_short_min: float = 0.45,
    trade_params: Optional[TradeParams] = None,
    gate: Optional[EntryGate] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    """Trade the Fibonacci golden zone (38.2-61.8 % retracement band) with trend.

    ML rationale: fib_4h_dist_fib_382 and fib_8h_dist_fib_382/786 are top-5
    permutation features at all three horizons tested.

    in_fib_zone_382_618 == 1  <=>  fib_618 <= close <= fib_382
    range_pos = (close - swing_low) / range  (0=low, 1=high)
    """
    if trade_params is None:
        trade_params = TradeParams(
            stop_loss_pct=0.0022,
            take_profit_pct=0.0048,
            max_hold_seconds=5400.0,
        )
    if gate is None:
        gate = EntryGate(trade_start="09:00", trade_end="14:00",
                         cooldown_bars=3600, max_trades_per_day=3)

    zone_col      = f"{fib_prefix}_in_fib_zone_382_618"
    range_pos_col = f"{fib_prefix}_range_pos"
    range_col     = f"{fib_prefix}_range"

    return confluence_strategy(
        df,
        long_conditions=[
            col_eq(zone_col, 1),
            col_lte(range_pos_col, range_pos_long_max),
            col_gt("ema_1200", "ema_3600"),
            col_gt(range_col, range_min),
        ],
        short_conditions=[
            col_eq(zone_col, 1),
            col_gte(range_pos_col, range_pos_short_min),
            col_lt("ema_1200", "ema_3600"),
            col_gt(range_col, range_min),
        ],
        trade_params=trade_params,
        gate=gate,
        ts_col=ts_col,
        symbol_col=symbol_col,
    )


def rolling_range_breakout_trend(
    df: pd.DataFrame,
    *,
    range_label: str = "2h",
    fast_ema: str = "ema_300",
    slow_ema: str = "ema_1200",
    adx_min: float = 20.0,
    trade_params: Optional[TradeParams] = None,
    gate: Optional[EntryGate] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    """Breakout of 2-hour rolling range confirmed by medium + long EMA alignment.

    ML rationale: dist_rolling_high_2h/4h are top-3 permutation features at the
    600s horizon and top-10 at 300s.  Breakouts in the direction of the trend
    deliver the best forward-return edge.

    dist_rolling_high_2h = close / rolling_high_2h_shifted - 1
      crossing 0 upward  =>  price just made a NEW 2-hour high (breakout long)
    dist_rolling_low_2h crossing 0 downward => new 2-hour low (breakout short)
    """
    if trade_params is None:
        trade_params = TradeParams(
            stop_loss_pct=0.0025,
            take_profit_pct=0.0055,
            max_hold_seconds=7200.0,
        )
    if gate is None:
        gate = EntryGate(trade_start="09:00", trade_end="13:00",
                         cooldown_bars=7200, max_trades_per_day=2)

    dist_high = f"dist_rolling_high_{range_label}"
    dist_low  = f"dist_rolling_low_{range_label}"
    adx_col   = _first_existing(df, ["adx_50", "adx_100", "adx_14"])

    long_conds: list[Condition] = [
        crossed_above(dist_high, 0.0),
        col_gt(fast_ema, slow_ema),
    ]
    short_conds: list[Condition] = [
        crossed_below(dist_low, 0.0),
        col_lt(fast_ema, slow_ema),
    ]
    if adx_col is not None:
        long_conds.append(col_gte(adx_col, adx_min))
        short_conds.append(col_gte(adx_col, adx_min))

    return confluence_strategy(
        df,
        long_conditions=long_conds,
        short_conditions=short_conds,
        trade_params=trade_params,
        gate=gate,
        ts_col=ts_col,
        symbol_col=symbol_col,
    )


def session_level_reversal(
    df: pd.DataFrame,
    *,
    near_pct: float = 0.0025,
    break_buffer: float = 0.0010,
    trend_col: str = "price_vs_ema3600",
    trend_filter: float = 0.004,
    trade_params: Optional[TradeParams] = None,
    gate: Optional[EntryGate] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    """Reversal at the previous session high/low used as intraday S/R.

    ML rationale: dist_prev_session_high and dist_prev_session_low land in
    the top-15 features across all horizons and importance types.

    dist_prev_session_high = close / prev_session_high - 1
      near 0 from below => resistance test (short)
    dist_prev_session_low = close / prev_session_low - 1
      near 0 from above => support test (long)
    """
    if trade_params is None:
        trade_params = TradeParams(
            stop_loss_pct=0.0018,
            take_profit_pct=0.0040,
            max_hold_seconds=3600.0,
        )
    if gate is None:
        gate = EntryGate(trade_start="09:30", trade_end="12:00",
                         cooldown_bars=3600, max_trades_per_day=2)

    return confluence_strategy(
        df,
        long_conditions=[
            col_lte("dist_prev_session_low", near_pct),
            col_gte("dist_prev_session_low", -break_buffer),
            col_gte(trend_col, -trend_filter),
        ],
        short_conditions=[
            col_gte("dist_prev_session_high", -near_pct),
            col_lte("dist_prev_session_high", break_buffer),
            col_lte(trend_col, trend_filter),
        ],
        trade_params=trade_params,
        gate=gate,
        ts_col=ts_col,
        symbol_col=symbol_col,
    )


def or15m_breakout_trend(
    df: pd.DataFrame,
    *,
    fast_ema: str = "ema_1200",
    slow_ema: str = "ema_3600",
    adx_min: float = 18.0,
    trade_params: Optional[TradeParams] = None,
    gate: Optional[EntryGate] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    """Breakout of the 15-min opening range confirmed by EMA trend alignment.

    ML rationale: dist_or_high_15m and dist_or_low_15m are top session-level
    features at the 60s and 300s horizons (permutation importance).

    dist_or_high_15m = close / opening_range_high_15m - 1
      crossing 0 upward  => breakout above the 15-min opening range
    dist_or_low_15m crossing 0 downward => breakdown below the OR
    """
    if trade_params is None:
        trade_params = TradeParams(
            stop_loss_pct=0.0022,
            take_profit_pct=0.0050,
            max_hold_seconds=5400.0,
        )
    if gate is None:
        gate = EntryGate(trade_start="10:00", trade_end="12:30",
                         cooldown_bars=5400, max_trades_per_day=2)

    adx_col = _first_existing(df, ["adx_50", "adx_14"])

    long_conds: list[Condition] = [
        crossed_above("dist_or_high_15m", 0.0),
        col_gt(fast_ema, slow_ema),
    ]
    short_conds: list[Condition] = [
        crossed_below("dist_or_low_15m", 0.0),
        col_lt(fast_ema, slow_ema),
    ]
    if adx_col is not None:
        long_conds.append(col_gte(adx_col, adx_min))
        short_conds.append(col_gte(adx_col, adx_min))

    return confluence_strategy(
        df,
        long_conditions=long_conds,
        short_conditions=short_conds,
        trade_params=trade_params,
        gate=gate,
        ts_col=ts_col,
        symbol_col=symbol_col,
    )


def long_ema_pullback(
    df: pd.DataFrame,
    *,
    fast_ema: str = "ema_1200",
    slow_ema: str = "ema_3600",
    pullback_near: float = 0.0020,
    pullback_max: float = 0.0035,
    adx_min: float = 20.0,
    trade_params: Optional[TradeParams] = None,
    gate: Optional[EntryGate] = None,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    """Pullback to the 60-min EMA (dominant trend line) as a trend re-entry.

    ML rationale: ema_3600, ema_1200 and sma_1800 are the top trend features
    at the 600s horizon by permutation importance, far outranking shorter EMAs.
    The 60-min EMA acts as a dynamic trend line.

    price_vs_ema3600 = close / ema_3600 - 1
      near 0 from above => touching EMA from above (long re-entry in uptrend)
      near 0 from below => touching EMA from below (short re-entry in downtrend)
    """
    if trade_params is None:
        trade_params = TradeParams(
            stop_loss_pct=0.0022,
            take_profit_pct=0.0050,
            max_hold_seconds=7200.0,
        )
    if gate is None:
        gate = EntryGate(trade_start="09:00", trade_end="13:00",
                         cooldown_bars=7200, max_trades_per_day=2)

    adx_col = _first_existing(df, ["adx_100", "adx_50", "adx_14"])

    long_conds: list[Condition] = [
        col_gt(fast_ema, slow_ema),
        col_gte("price_vs_ema3600", -pullback_max),
        col_lte("price_vs_ema3600", pullback_near),
    ]
    short_conds: list[Condition] = [
        col_lt(fast_ema, slow_ema),
        col_lte("price_vs_ema3600", pullback_max),
        col_gte("price_vs_ema3600", -pullback_near),
    ]
    if adx_col is not None:
        long_conds.append(col_gte(adx_col, adx_min))
        short_conds.append(col_gte(adx_col, adx_min))

    return confluence_strategy(
        df,
        long_conditions=long_conds,
        short_conditions=short_conds,
        trade_params=trade_params,
        gate=gate,
        ts_col=ts_col,
        symbol_col=symbol_col,
    )
