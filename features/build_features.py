from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import talib


def _pick_column(columns, candidates, required=True):
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def _to_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def _safe_talib_series(values, index, shift=True):
    s = pd.Series(values, index=index)
    return s.shift(1) if shift else s


def _sanitize_ohlcv(g: pd.DataFrame, open_col, high_col, low_col, close_col, volume_col=None) -> pd.DataFrame:
    g = g.copy()
    for col in [open_col, high_col, low_col, close_col, volume_col]:
        if col is not None and col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce")
    return g


def build_features(
    df: pd.DataFrame,
    symbol_col: Optional[str] = None,
    ts_col: Optional[str] = None,
    add_basic_returns: bool = True,
    add_trend: bool = True,
    add_momentum: bool = True,
    add_volatility: bool = True,
    add_volume: bool = True,
    shift_features: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    symbol_col = symbol_col or _pick_column(df.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"])
    ts_col = ts_col or _pick_column(df.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])

    open_col = _pick_column(df.columns, ["open", "open_"])
    high_col = _pick_column(df.columns, ["high", "high_"])
    low_col = _pick_column(df.columns, ["low", "low_"])
    close_col = _pick_column(df.columns, ["close", "close_", "settlement"])
    volume_col = _pick_column(df.columns, ["volume"], required=False)

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values([symbol_col, ts_col]).reset_index(drop=True)

    out = []

    for symbol, g in df.groupby(symbol_col, sort=False):
        g = g.copy()
        g = _sanitize_ohlcv(g, open_col, high_col, low_col, close_col, volume_col)

        close = _to_float_series(g, close_col)
        open_ = _to_float_series(g, open_col)
        high = _to_float_series(g, high_col)
        low = _to_float_series(g, low_col)
        volume = _to_float_series(g, volume_col) if volume_col is not None else None

        if add_basic_returns:
            g["ret_1"] = close.pct_change(1).shift(1 if shift_features else 0)
            g["ret_5"] = close.pct_change(5).shift(1 if shift_features else 0)
            g["ret_10"] = close.pct_change(10).shift(1 if shift_features else 0)
            g["logret_1"] = np.log(close).diff(1).shift(1 if shift_features else 0)
            g["logret_5"] = np.log(close).diff(5).shift(1 if shift_features else 0)
            g["hl_spread"] = ((high - low) / close.replace(0, np.nan)).shift(1 if shift_features else 0)
            g["oc_spread"] = ((close - open_) / open_.replace(0, np.nan)).shift(1 if shift_features else 0)

        if add_trend:
            range_list = [20, 60, 70, 80, 90, 95, 100, 105, 110, 120, 150, 200, 300, 400, 600]
            for range in range_list: 
                g[f"sma_{range}"] = _safe_talib_series(talib.SMA(close.values, timeperiod=range), g.index, shift_features)
                g[f"ema_{range}"] = _safe_talib_series(talib.EMA(close.values, timeperiod=range), g.index, shift_features)

                g[f"price_vs_sma{range}"] = close / g[f"sma_{range}"] - 1.0
                g[f"price_vs_ema{range}"] = close / g[f"ema_{range}"] - 1.0

            g["macd"], g["macd_signal"], g["macd_hist"] = [
                _safe_talib_series(x, g.index, shift_features)
                for x in talib.MACD(close.values, fastperiod=12, slowperiod=26, signalperiod=9)
            ]

        if add_momentum:
            g["rsi_14"] = _safe_talib_series(talib.RSI(close.values, timeperiod=14), g.index, shift_features)
            g["mom_10"] = _safe_talib_series(talib.MOM(close.values, timeperiod=10), g.index, shift_features)
            g["roc_10"] = _safe_talib_series(talib.ROC(close.values, timeperiod=10), g.index, shift_features)
            g["willr_14"] = _safe_talib_series(
                talib.WILLR(high.values, low.values, close.values, timeperiod=14),
                g.index,
                shift_features,
            )
            stoch_k, stoch_d = talib.STOCH(
                high.values,
                low.values,
                close.values,
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0,
            )
            g["stoch_k"] = _safe_talib_series(stoch_k, g.index, shift_features)
            g["stoch_d"] = _safe_talib_series(stoch_d, g.index, shift_features)

        if add_volatility:
            g["atr_14"] = _safe_talib_series(
                talib.ATR(high.values, low.values, close.values, timeperiod=14),
                g.index,
                shift_features,
            )
            g["natr_14"] = _safe_talib_series(
                talib.NATR(high.values, low.values, close.values, timeperiod=14),
                g.index,
                shift_features,
            )
            upper, middle, lower = talib.BBANDS(
                close.values,
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0,
            )
            g["bb_upper"] = _safe_talib_series(upper, g.index, shift_features)
            g["bb_middle"] = _safe_talib_series(middle, g.index, shift_features)
            g["bb_lower"] = _safe_talib_series(lower, g.index, shift_features)
            g["bb_width"] = (g["bb_upper"] - g["bb_lower"]) / g["bb_middle"].replace(0, np.nan)
            g["bb_pos"] = (close - g["bb_lower"]) / (g["bb_upper"] - g["bb_lower"]).replace(0, np.nan)
            g["adx_14"] = _safe_talib_series(
                talib.ADX(high.values, low.values, close.values, timeperiod=14),
                g.index,
                shift_features,
            )

            g["ema80_atr_dist"] = (close - g["ema_80"]) / g["atr_14"].replace(0, np.nan)
            g["ema100_atr_dist"] = (close - g["ema_100"]) / g["atr_14"].replace(0, np.nan)

        if add_volume and volume is not None:
            g["obv"] = _safe_talib_series(talib.OBV(close.values, volume.values), g.index, shift_features)
            g["vol_sma_20"] = _safe_talib_series(talib.SMA(volume.values, timeperiod=20), g.index, shift_features)
            g["rel_volume_20"] = volume / g["vol_sma_20"]
        
        g["hour_utc"] = g[ts_col].dt.hour
        g["is_london_ny"] = ((g["hour_utc"] >= 8) & (g["hour_utc"] <= 16)).astype(int)
        g["is_us_morning"] = ((g["hour_utc"] >= 13) & (g["hour_utc"] <= 17)).astype(int)

        out.append(g)

    out = pd.concat(out, axis=0, ignore_index=True)
    return out


def add_targets(
    df: pd.DataFrame,
    symbol_col: Optional[str] = None,
    ts_col: Optional[str] = None,
    close_col: Optional[str] = None,
    horizons: Iterable[int] = (60, 300, 600),
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    symbol_col = symbol_col or _pick_column(df.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"])
    ts_col = ts_col or _pick_column(df.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    close_col = close_col or _pick_column(df.columns, ["close", "close_", "settlement"])

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values([symbol_col, ts_col]).reset_index(drop=True)

    out = []

    for _, g in df.groupby(symbol_col, sort=False):
        g = g.copy()
        close = pd.to_numeric(g[close_col], errors="coerce").astype(float)

        for h in horizons:
            g[f"target_fwd_ret_{h}s"] = close.shift(-h) / close - 1.0
            g[f"target_fwd_logret_{h}s"] = np.log(close.shift(-h)) - np.log(close)

        out.append(g)

    out = pd.concat(out, axis=0, ignore_index=True)
    return out