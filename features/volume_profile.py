from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from features.intraday_utils import group_key, pick_column


def add_volume_profile_features(
    df: pd.DataFrame,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
    bins: int = 20,
    window: str = "session",
) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    ts_col = ts_col or pick_column(out.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    symbol_col = symbol_col or pick_column(out.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"], required=False)
    if "volume" not in out.columns:
        return out

    keys = group_key(out, ts_col=ts_col, symbol_col=symbol_col)
    out["volume_ma_20"] = out.groupby(symbol_col, sort=False)["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean()) if symbol_col else out["volume"].rolling(20, min_periods=5).mean()
    out["relative_volume"] = out["volume"] / out["volume_ma_20"].replace(0, np.nan)
    out["volume_spike"] = (out["relative_volume"] >= 2.0).astype("int8")
    out["volume_poc"] = np.nan
    out["value_area_high"] = np.nan
    out["value_area_low"] = np.nan

    for _, idx in keys.groupby(keys).groups.items():
        idx = list(idx)
        session = out.loc[idx]
        if len(session) < 3:
            continue
        prices = ((session["high"] + session["low"] + session["close"]) / 3.0).to_numpy(dtype=float)
        volumes = session["volume"].to_numpy(dtype=float)
        hist, edges = np.histogram(prices, bins=bins, weights=volumes)
        if hist.sum() <= 0:
            continue
        poc_bin = int(np.argmax(hist))
        poc = float((edges[poc_bin] + edges[poc_bin + 1]) / 2.0)
        cumulative = np.cumsum(hist[np.argsort(hist)[::-1]])
        ranked_bins = np.argsort(hist)[::-1]
        threshold = 0.7 * hist.sum()
        keep = ranked_bins[cumulative <= threshold]
        if len(keep) == 0:
            keep = ranked_bins[:1]
        vah = float(edges[max(keep) + 1])
        val = float(edges[min(keep)])
        out.loc[idx, "volume_poc"] = poc
        out.loc[idx, "value_area_high"] = vah
        out.loc[idx, "value_area_low"] = val

    return out
