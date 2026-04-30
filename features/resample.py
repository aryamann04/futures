from __future__ import annotations

from typing import Optional

import pandas as pd

from features.intraday_utils import ensure_timestamp_series, pick_column


OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    aliases = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
    }
    return pd.Timedelta(aliases.get(timeframe, timeframe))


def infer_base_timedelta(df: pd.DataFrame, ts_col: str) -> pd.Timedelta:
    ts = ensure_timestamp_series(df[ts_col]).sort_values()
    deltas = ts.diff().dropna()
    if deltas.empty:
        return pd.Timedelta(0)
    return deltas.mode().iloc[0]


def resample_ohlcv(
    df: pd.DataFrame,
    timeframe: str,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    """Resample intraday OHLCV bars while preserving timezone-aware timestamps."""
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    ts_col = ts_col or pick_column(out.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    symbol_col = symbol_col or pick_column(out.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"], required=False)

    out[ts_col] = ensure_timestamp_series(out[ts_col])
    sort_cols = ([symbol_col] if symbol_col else []) + [ts_col]
    out = out.dropna(subset=[ts_col]).sort_values(sort_cols).reset_index(drop=True)

    agg_map = {col: rule for col, rule in OHLCV_AGG.items() if col in out.columns}
    if not agg_map:
        raise ValueError("Expected OHLCV columns for resampling.")

    pieces: list[pd.DataFrame] = []
    groups = [(None, out)] if symbol_col is None else out.groupby(symbol_col, sort=False)
    for symbol, group in groups:
        resampled = (
            group.set_index(ts_col)
            .resample(timeframe, label="right", closed="right")
            .agg(agg_map)
            .dropna(subset=[c for c in ["open", "high", "low", "close"] if c in agg_map])
            .reset_index()
        )
        if symbol_col is not None:
            resampled[symbol_col] = symbol
        pieces.append(resampled)

    result = pd.concat(pieces, axis=0, ignore_index=True)
    sort_cols = ([symbol_col] if symbol_col else []) + [ts_col]
    return result.sort_values(sort_cols).reset_index(drop=True)
