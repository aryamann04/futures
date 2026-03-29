from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


REQUIRED_PRICE_COLUMNS = ["open", "high", "low", "close"]

CONTRACT_SPECS = {
    "M6E": {"multiplier": 12500.0},
    "M6B": {"multiplier": 6250.0},
    "M6A": {"multiplier": 10000.0},
    "M6C": {"multiplier": 10000.0},
    "M6J": {"multiplier": 1250000.0},
    "M6S": {"multiplier": 12500.0},
    "MJY": {"multiplier": 1250000.0},
    "MNH": {"multiplier": 10000.0},
}


@dataclass
class Trade:
    symbol: Optional[str]
    symbol_root: Optional[str]
    side: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    multiplier: float
    notional_entry: float
    bars_held: int
    reason: str
    stop_loss: float
    take_profit: float
    pnl_points: float
    pnl_dollars: float
    ret: float
    ret_notional: float


def _pick_column(columns, candidates, required=True):
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def _validate_input(df: pd.DataFrame):
    cols = [c.lower() for c in df.columns]
    for col in REQUIRED_PRICE_COLUMNS:
        if col not in cols:
            raise ValueError(f"Missing required price column: {col}")
    if "entry_signal" not in cols:
        raise ValueError("Missing required signal column: entry_signal")


def _prepare_df(
    df: pd.DataFrame,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> tuple[pd.DataFrame, str, Optional[str]]:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]

    _validate_input(out)

    ts_col = ts_col or _pick_column(out.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    symbol_col = symbol_col or _pick_column(out.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"], required=False)

    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    out = out.dropna(subset=[ts_col]).sort_values(([symbol_col] if symbol_col else []) + [ts_col]).reset_index(drop=True)

    default_cols = {
        "entry_signal": 0,
        "exit_signal": 0,
        "stop_loss": np.nan,
        "take_profit": np.nan,
        "stop_loss_pct": np.nan,
        "take_profit_pct": np.nan,
        "max_hold_bars": np.nan,
        "max_hold_seconds": np.nan,
        "size": 1.0,
    }
    for col, default_val in default_cols.items():
        if col not in out.columns:
            out[col] = default_val

    out["entry_signal"] = pd.to_numeric(out["entry_signal"], errors="coerce").fillna(0).astype(np.int8)
    out["exit_signal"] = pd.to_numeric(out["exit_signal"], errors="coerce").fillna(0).astype(np.int8)
    out["size"] = pd.to_numeric(out["size"], errors="coerce").fillna(1.0)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "stop_loss",
        "take_profit",
        "stop_loss_pct",
        "take_profit_pct",
        "max_hold_bars",
        "max_hold_seconds",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out, ts_col, symbol_col


def _get_symbol_root(symbol: Optional[str]) -> Optional[str]:
    if symbol is None:
        return None
    s = str(symbol)
    return s[:3] if len(s) >= 3 else s


def _get_multiplier(symbol: Optional[str], contract_specs: Optional[dict] = None) -> float:
    if contract_specs is None:
        contract_specs = CONTRACT_SPECS
    root = _get_symbol_root(symbol)
    if root is None:
        return 1.0
    return float(contract_specs.get(root, {}).get("multiplier", 1.0))


def _absolute_levels_from_pct(side: int, entry_price: float, stop_loss_pct: float, take_profit_pct: float):
    stop_loss = np.nan
    take_profit = np.nan

    if np.isfinite(stop_loss_pct):
        stop_loss = entry_price * (1.0 - stop_loss_pct) if side == 1 else entry_price * (1.0 + stop_loss_pct)

    if np.isfinite(take_profit_pct):
        take_profit = entry_price * (1.0 + take_profit_pct) if side == 1 else entry_price * (1.0 - take_profit_pct)

    return stop_loss, take_profit


def _valid_absolute_levels(side: int, entry_price: float, stop_loss: float, take_profit: float):
    valid_stop = np.nan
    valid_tp = np.nan

    if side == 1:
        if np.isfinite(stop_loss) and stop_loss < entry_price:
            valid_stop = stop_loss
        if np.isfinite(take_profit) and take_profit > entry_price:
            valid_tp = take_profit
    else:
        if np.isfinite(stop_loss) and stop_loss > entry_price:
            valid_stop = stop_loss
        if np.isfinite(take_profit) and take_profit < entry_price:
            valid_tp = take_profit

    return valid_stop, valid_tp


def _resolve_risk_levels(
    side: int,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    stop_loss_pct: float,
    take_profit_pct: float,
):
    if np.isfinite(stop_loss_pct) or np.isfinite(take_profit_pct):
        return _absolute_levels_from_pct(
            side=side,
            entry_price=entry_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )

    return _valid_absolute_levels(
        side=side,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )


def _exit_price_for_bar(side: int, bar_high: float, bar_low: float, stop_loss: float, take_profit: float):
    stop_hit = False
    tp_hit = False

    if side == 1:
        if np.isfinite(stop_loss) and bar_low <= stop_loss:
            stop_hit = True
        if np.isfinite(take_profit) and bar_high >= take_profit:
            tp_hit = True
        if stop_hit and tp_hit:
            return stop_loss, "stop_loss"
        if stop_hit:
            return stop_loss, "stop_loss"
        if tp_hit:
            return take_profit, "take_profit"

    if side == -1:
        if np.isfinite(stop_loss) and bar_high >= stop_loss:
            stop_hit = True
        if np.isfinite(take_profit) and bar_low <= take_profit:
            tp_hit = True
        if stop_hit and tp_hit:
            return stop_loss, "stop_loss"
        if stop_hit:
            return stop_loss, "stop_loss"
        if tp_hit:
            return take_profit, "take_profit"

    return None, None


def _run_single_symbol(
    df: pd.DataFrame,
    ts_col: str,
    symbol_col: Optional[str] = None,
    contract_specs: Optional[dict] = None,
    commission_per_side: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)

    open_arr = df["open"].to_numpy(dtype=float)
    high_arr = df["high"].to_numpy(dtype=float)
    low_arr = df["low"].to_numpy(dtype=float)
    close_arr = df["close"].to_numpy(dtype=float)
    entry_signal_arr = df["entry_signal"].to_numpy(dtype=np.int8)
    exit_signal_arr = df["exit_signal"].to_numpy(dtype=np.int8)
    size_arr = df["size"].to_numpy(dtype=float)
    stop_loss_arr = df["stop_loss"].to_numpy(dtype=float)
    take_profit_arr = df["take_profit"].to_numpy(dtype=float)
    stop_loss_pct_arr = df["stop_loss_pct"].to_numpy(dtype=float)
    take_profit_pct_arr = df["take_profit_pct"].to_numpy(dtype=float)
    max_hold_bars_arr = df["max_hold_bars"].to_numpy(dtype=float)
    max_hold_seconds_arr = df["max_hold_seconds"].to_numpy(dtype=float)
    ts_arr = df[ts_col].to_numpy()

    position = np.zeros(n, dtype=np.int8)
    entry_price_col = np.full(n, np.nan, dtype=float)
    exit_price_col = np.full(n, np.nan, dtype=float)
    trade_id_col = np.full(n, np.nan, dtype=float)
    stop_loss_col = np.full(n, np.nan, dtype=float)
    take_profit_col = np.full(n, np.nan, dtype=float)
    pnl_points_col = np.zeros(n, dtype=float)
    pnl_dollars_col = np.zeros(n, dtype=float)
    reason_col = np.array([""] * n, dtype=object)

    trades: list[Trade] = []

    symbol_value = df[symbol_col].iloc[0] if symbol_col else None
    symbol_root = _get_symbol_root(symbol_value)
    multiplier = _get_multiplier(symbol_value, contract_specs=contract_specs)

    in_pos = False
    side = 0
    size = 0.0
    entry_time = None
    entry_price = np.nan
    stop_loss = np.nan
    take_profit = np.nan
    max_hold_bars = np.nan
    max_hold_seconds = np.nan
    bars_held = 0
    trade_id = 0

    for i in range(1, n):
        prev_i = i - 1

        if not in_pos:
            signal = int(entry_signal_arr[prev_i])
            if signal != 0 and np.isfinite(open_arr[i]):
                in_pos = True
                side = 1 if signal > 0 else -1
                size = size_arr[prev_i] if np.isfinite(size_arr[prev_i]) else 1.0
                entry_time = pd.Timestamp(ts_arr[i])
                entry_price = float(open_arr[i])

                stop_loss, take_profit = _resolve_risk_levels(
                    side=side,
                    entry_price=entry_price,
                    stop_loss=stop_loss_arr[prev_i],
                    take_profit=take_profit_arr[prev_i],
                    stop_loss_pct=stop_loss_pct_arr[prev_i],
                    take_profit_pct=take_profit_pct_arr[prev_i],
                )

                max_hold_bars = max_hold_bars_arr[prev_i]
                max_hold_seconds = max_hold_seconds_arr[prev_i]
                bars_held = 0
                trade_id += 1

                position[i] = side
                entry_price_col[i] = entry_price
                trade_id_col[i] = trade_id
                stop_loss_col[i] = stop_loss
                take_profit_col[i] = take_profit
            continue

        bars_held += 1
        position[i] = side
        entry_price_col[i] = entry_price
        trade_id_col[i] = trade_id
        stop_loss_col[i] = stop_loss
        take_profit_col[i] = take_profit

        px, reason = _exit_price_for_bar(
            side=side,
            bar_high=float(high_arr[i]),
            bar_low=float(low_arr[i]),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        if px is None:
            reverse_signal = int(entry_signal_arr[prev_i]) == -side
            explicit_exit = int(exit_signal_arr[prev_i]) != 0

            time_exit_bars = np.isfinite(max_hold_bars) and bars_held >= int(max_hold_bars)
            if np.isfinite(max_hold_seconds):
                elapsed_seconds = (pd.Timestamp(ts_arr[i]) - entry_time).total_seconds()
                time_exit_seconds = elapsed_seconds >= float(max_hold_seconds)
            else:
                time_exit_seconds = False

            time_exit = time_exit_bars or time_exit_seconds

            if explicit_exit:
                px = float(open_arr[i])
                reason = "exit_signal"
            elif reverse_signal:
                px = float(open_arr[i])
                reason = "reverse_signal"
            elif time_exit:
                px = float(close_arr[i])
                reason = "time_exit"

        if px is not None:
            exit_time = pd.Timestamp(ts_arr[i])
            exit_price = float(px)

            pnl_points = side * size * (exit_price - entry_price)
            pnl_dollars = side * size * multiplier * (exit_price - entry_price)
            pnl_dollars -= 2.0 * commission_per_side

            ret = side * (exit_price / entry_price - 1.0) if entry_price != 0 else np.nan
            notional_entry = abs(size * multiplier * entry_price) if entry_price != 0 else np.nan
            ret_notional = pnl_dollars / notional_entry if np.isfinite(notional_entry) and notional_entry != 0 else np.nan

            exit_price_col[i] = exit_price
            pnl_points_col[i] = pnl_points
            pnl_dollars_col[i] = pnl_dollars
            reason_col[i] = reason

            trades.append(
                Trade(
                    symbol=symbol_value,
                    symbol_root=symbol_root,
                    side=side,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=size,
                    multiplier=multiplier,
                    notional_entry=notional_entry,
                    bars_held=bars_held,
                    reason=reason,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    pnl_points=pnl_points,
                    pnl_dollars=pnl_dollars,
                    ret=ret,
                    ret_notional=ret_notional,
                )
            )

            in_pos = False
            side = 0
            size = 0.0
            entry_time = None
            entry_price = np.nan
            stop_loss = np.nan
            take_profit = np.nan
            max_hold_bars = np.nan
            max_hold_seconds = np.nan
            bars_held = 0

    out = df.copy()
    out["position"] = position
    out["entry_price_filled"] = entry_price_col
    out["exit_price_filled"] = exit_price_col
    out["trade_id"] = trade_id_col
    out["active_stop_loss"] = stop_loss_col
    out["active_take_profit"] = take_profit_col
    out["trade_pnl_points"] = pnl_points_col
    out["trade_pnl_dollars"] = pnl_dollars_col
    out["trade_exit_reason"] = reason_col

    close_ret = pd.Series(close_arr, index=out.index).pct_change().fillna(0.0).to_numpy(dtype=float)
    out["strategy_ret"] = pd.Series(np.roll(position, 1), index=out.index).fillna(0).astype(float)
    out["strategy_ret"] = out["strategy_ret"] * close_ret
    out["equity_curve"] = (1.0 + out["strategy_ret"]).cumprod()

    trades_df = pd.DataFrame([t.__dict__ for t in trades])

    return out, trades_df


def run_backtest(
    df: pd.DataFrame,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
    contract_specs: Optional[dict] = None,
    commission_per_side: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, ts_col, symbol_col = _prepare_df(df, ts_col=ts_col, symbol_col=symbol_col)

    if symbol_col is None:
        return _run_single_symbol(
            df,
            ts_col=ts_col,
            symbol_col=None,
            contract_specs=contract_specs,
            commission_per_side=commission_per_side,
        )

    bars_out = []
    trades_out = []

    for _, g in df.groupby(symbol_col, sort=False):
        bars_g, trades_g = _run_single_symbol(
            g.reset_index(drop=True),
            ts_col=ts_col,
            symbol_col=symbol_col,
            contract_specs=contract_specs,
            commission_per_side=commission_per_side,
        )
        bars_out.append(bars_g)
        if not trades_g.empty:
            trades_out.append(trades_g)

    bars = pd.concat(bars_out, axis=0, ignore_index=True)
    trades = pd.concat(trades_out, axis=0, ignore_index=True) if trades_out else pd.DataFrame(
        columns=[
            "symbol",
            "symbol_root",
            "side",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "size",
            "multiplier",
            "notional_entry",
            "bars_held",
            "reason",
            "stop_loss",
            "take_profit",
            "pnl_points",
            "pnl_dollars",
            "ret",
            "ret_notional",
        ]
    )

    return bars, trades