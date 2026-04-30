from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from features.intraday_utils import pick_column
from features.resample import infer_base_timedelta, resample_ohlcv, timeframe_to_timedelta


def _merge_asof_by_symbol(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    ts_col: str,
    symbol_col: Optional[str],
) -> pd.DataFrame:
    if symbol_col is None:
        return pd.merge_asof(
            left.sort_values(ts_col),
            right.sort_values(ts_col),
            on=ts_col,
            direction="backward",
        )

    left_work = left.copy()
    left_work["_orig_order"] = np.arange(len(left_work))
    merged = pd.merge_asof(
        left_work.sort_values([ts_col, symbol_col]).reset_index(drop=True),
        right.sort_values([ts_col, symbol_col]).reset_index(drop=True),
        on=ts_col,
        by=symbol_col,
        direction="backward",
    )
    merged = merged.sort_values("_orig_order").drop(columns="_orig_order").reset_index(drop=True)
    return merged


def compute_atr(
    df: pd.DataFrame,
    period: int = 14,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    ts_col = ts_col or pick_column(out.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    symbol_col = symbol_col or pick_column(out.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"], required=False)

    out = out.sort_values(([symbol_col] if symbol_col else []) + [ts_col]).reset_index(drop=True)

    pieces: list[pd.Series] = []
    groups = [(None, out)] if symbol_col is None else out.groupby(symbol_col, sort=False)
    for _, group in groups:
        prev_close = group[close_col].shift(1)
        tr = pd.concat(
            [
                group[high_col] - group[low_col],
                (group[high_col] - prev_close).abs(),
                (group[low_col] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        pieces.append(pd.Series(atr.to_numpy(), index=group.index))

    return pd.concat(pieces).sort_index()


def add_atr_features(
    df: pd.DataFrame,
    periods: tuple[int, ...] = (14,),
    timeframes: tuple[str, ...] = ("1min", "5min", "15min"),
    percentile_window: int = 100,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    ts_col = ts_col or pick_column(out.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    symbol_col = symbol_col or pick_column(out.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"], required=False)
    base_td = infer_base_timedelta(out, ts_col)

    for timeframe in timeframes:
        target_td = timeframe_to_timedelta(timeframe)
        tf_df = out if base_td == target_td else resample_ohlcv(out, timeframe, ts_col=ts_col, symbol_col=symbol_col)
        for period in periods:
            atr = compute_atr(tf_df, period=period, ts_col=ts_col, symbol_col=symbol_col)
            atr_col = f"atr_{timeframe}_{period}"
            tf_df = tf_df.copy()
            tf_df[atr_col] = atr
            if timeframe == "1min":
                out[atr_col] = atr.to_numpy()
            else:
                merge_cols = [ts_col, atr_col] + ([symbol_col] if symbol_col else [])
                out = _merge_asof_by_symbol(
                    out,
                    tf_df[merge_cols],
                    ts_col=ts_col,
                    symbol_col=symbol_col,
                )

            rank_col = f"{atr_col}_pct_rank"
            out[rank_col] = (
                out.groupby(symbol_col, sort=False)[atr_col].transform(
                    lambda s: s.rolling(percentile_window, min_periods=max(10, percentile_window // 5))
                    .apply(lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]), raw=False)
                )
                if symbol_col
                else out[atr_col].rolling(percentile_window, min_periods=max(10, percentile_window // 5))
                .apply(lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]), raw=False)
            )

            regime_col = f"{atr_col}_regime"
            regime = pd.Series("normal", index=out.index, dtype="object")
            regime.loc[out[rank_col] <= 0.2] = "low_volatility"
            regime.loc[out[rank_col] >= 0.8] = "expanding"
            regime.loc[out[rank_col] >= 0.95] = "extended"
            out[regime_col] = regime

    return out
