from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from features.intraday_utils import (
    DEFAULT_SESSION_WINDOWS,
    SessionWindow,
    group_key,
    pick_column,
    time_window_mask,
    to_timezone,
    trading_day_key,
)


def _session_cumulative(
    df: pd.DataFrame,
    *,
    local_ts: pd.Series,
    symbol_col: Optional[str],
    session_name: str,
    start: str,
    end: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mask = time_window_mask(local_ts, start, end)
    tday = trading_day_key(local_ts)
    keys = (df[symbol_col].astype(str) + "|" + tday.astype(str)) if symbol_col else tday.astype(str)

    masked_high = df["high"].where(mask)
    masked_low = df["low"].where(mask)

    cum_high = masked_high.groupby(keys).cummax()
    cum_low = masked_low.groupby(keys).cummin()

    final_high = masked_high.groupby(keys).transform("max")
    final_low = masked_low.groupby(keys).transform("min")

    completed = (~mask) & keys.isin(keys[mask])
    out_high = cum_high.where(mask, np.nan)
    out_low = cum_low.where(mask, np.nan)
    out_high = out_high.where(mask, final_high.where(completed))
    out_low = out_low.where(mask, final_low.where(completed))

    so_far_high = cum_high
    so_far_low = cum_low
    so_far_high.name = f"{session_name}_high_so_far"
    so_far_low.name = f"{session_name}_low_so_far"
    return out_high, out_low, so_far_high, so_far_low


def add_session_level_features(
    df: pd.DataFrame,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
    timezone: str = "America/New_York",
    ny_open_range: tuple[str, str] = ("09:30", "09:45"),
    futures_open_range: tuple[str, str] = ("08:30", "09:00"),
    session_windows: tuple[SessionWindow, ...] = DEFAULT_SESSION_WINDOWS,
) -> pd.DataFrame:
    """Add session and prior-day reference levels without lookahead."""
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    ts_col = ts_col or pick_column(out.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    symbol_col = symbol_col or pick_column(out.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"], required=False)

    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
    out = out.dropna(subset=[ts_col]).sort_values(([symbol_col] if symbol_col else []) + [ts_col]).reset_index(drop=True)
    local_ts = to_timezone(out[ts_col], timezone)
    out["local_session_time"] = local_ts
    out["trading_day"] = trading_day_key(local_ts)

    if symbol_col:
        grouped = out.groupby([symbol_col, "trading_day"], sort=False)
    else:
        grouped = out.groupby(["trading_day"], sort=False)

    daily = grouped.agg(day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last")).reset_index()
    if symbol_col:
        daily["prev_day_high"] = daily.groupby(symbol_col, sort=False)["day_high"].shift(1)
        daily["prev_day_low"] = daily.groupby(symbol_col, sort=False)["day_low"].shift(1)
        daily["prev_day_close"] = daily.groupby(symbol_col, sort=False)["day_close"].shift(1)
        out = out.merge(daily[[symbol_col, "trading_day", "prev_day_high", "prev_day_low", "prev_day_close"]], on=[symbol_col, "trading_day"], how="left")
    else:
        daily["prev_day_high"] = daily["day_high"].shift(1)
        daily["prev_day_low"] = daily["day_low"].shift(1)
        daily["prev_day_close"] = daily["day_close"].shift(1)
        out = out.merge(daily[["trading_day", "prev_day_high", "prev_day_low", "prev_day_close"]], on="trading_day", how="left")

    out["current_session_high"] = grouped["high"].cummax().to_numpy()
    out["current_session_low"] = grouped["low"].cummin().to_numpy()

    for window in session_windows:
        high_col, low_col, so_far_high, so_far_low = _session_cumulative(
            out,
            local_ts=local_ts,
            symbol_col=symbol_col,
            session_name=window.name,
            start=window.start,
            end=window.end,
        )
        out[f"{window.name}_high"] = high_col
        out[f"{window.name}_low"] = low_col
        out[f"{window.name}_high_so_far"] = so_far_high
        out[f"{window.name}_low_so_far"] = so_far_low

    def _opening_range(start: str, end: str, prefix: str) -> None:
        mask = time_window_mask(local_ts, start, end)
        keys = group_key(out, ts_col=ts_col, symbol_col=symbol_col)
        high = out["high"].where(mask).groupby(keys).cummax()
        low = out["low"].where(mask).groupby(keys).cummin()
        final_high = out["high"].where(mask).groupby(keys).transform("max")
        final_low = out["low"].where(mask).groupby(keys).transform("min")
        complete = (~mask) & keys.isin(keys[mask])
        out[f"{prefix}_high"] = high.where(mask, final_high.where(complete))
        out[f"{prefix}_low"] = low.where(mask, final_low.where(complete))

    _opening_range(*ny_open_range, prefix="ny_open_range")
    _opening_range(*futures_open_range, prefix="futures_open_range")
    return out
