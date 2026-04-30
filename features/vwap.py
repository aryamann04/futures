from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from features.intraday_utils import group_key, pick_column


def add_vwap_features(
    df: pd.DataFrame,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
    reset: str = "daily",
    price_mode: str = "typical",
    atr_col: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    ts_col = ts_col or pick_column(out.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    symbol_col = symbol_col or pick_column(out.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"], required=False)

    if "volume" not in out.columns:
        out["vwap"] = np.nan
        out["price_above_vwap"] = 0
        out["price_below_vwap"] = 0
        out["vwap_reclaim_up"] = 0
        out["vwap_reclaim_down"] = 0
        return out

    if price_mode == "typical":
        price = (out["high"] + out["low"] + out["close"]) / 3.0
    elif price_mode == "close":
        price = out["close"]
    else:
        raise ValueError("price_mode must be 'typical' or 'close'")

    keys = group_key(out, ts_col=ts_col, symbol_col=symbol_col, roll_hour=18 if reset == "daily" else 0)
    pv = price * out["volume"]
    out["vwap"] = pv.groupby(keys).cumsum() / out["volume"].groupby(keys).cumsum().replace(0, np.nan)
    out["price_above_vwap"] = (out["close"] > out["vwap"]).astype("int8")
    out["price_below_vwap"] = (out["close"] < out["vwap"]).astype("int8")
    out["vwap_distance_points"] = out["close"] - out["vwap"]
    if atr_col and atr_col in out.columns:
        out["vwap_distance_atr"] = out["vwap_distance_points"] / out[atr_col].replace(0, np.nan)
    out["vwap_reclaim_up"] = ((out["close"] > out["vwap"]) & (out["close"].shift(1) <= out["vwap"].shift(1))).astype("int8")
    out["vwap_reclaim_down"] = ((out["close"] < out["vwap"]) & (out["close"].shift(1) >= out["vwap"].shift(1))).astype("int8")
    return out
