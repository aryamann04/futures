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


def _to_float_series(df: pd.DataFrame, col: Optional[str]) -> Optional[pd.Series]:
    if col is None:
        return None
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


def _rolling_level(series: pd.Series, window: int, shift_features: bool = True, fn: str = "max") -> pd.Series:
    base = series.shift(1) if shift_features else series
    if fn == "max":
        return base.rolling(window, min_periods=window).max()
    if fn == "min":
        return base.rolling(window, min_periods=window).min()
    raise ValueError("fn must be 'max' or 'min'")


def _opening_range_levels(
    g: pd.DataFrame,
    ts_col: str,
    high_col: str,
    low_col: str,
    minutes: int,
) -> tuple[pd.Series, pd.Series]:
    minute_of_day = g[ts_col].dt.hour * 60 + g[ts_col].dt.minute
    session_date = g[ts_col].dt.floor("D")

    first_window_mask = minute_of_day < minutes

    or_high = g.loc[first_window_mask].groupby(session_date[first_window_mask])[high_col].max()
    or_low = g.loc[first_window_mask].groupby(session_date[first_window_mask])[low_col].min()

    mapped_high = session_date.map(or_high)
    mapped_low = session_date.map(or_low)

    available = minute_of_day >= minutes
    mapped_high = mapped_high.where(available)
    mapped_low = mapped_low.where(available)

    return mapped_high.astype(float), mapped_low.astype(float)


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
    df = df.dropna(subset=[ts_col]).sort_values([symbol_col, ts_col]).reset_index(drop=True)

    out = []

    ema_sma_windows = [20, 60, 70, 80, 90, 95, 100, 105, 110, 120, 150, 200, 300, 400, 600]

    for _, g in df.groupby(symbol_col, sort=False):
        g = _sanitize_ohlcv(g, open_col, high_col, low_col, close_col, volume_col)

        close = _to_float_series(g, close_col)
        open_ = _to_float_series(g, open_col)
        high = _to_float_series(g, high_col)
        low = _to_float_series(g, low_col)
        volume = _to_float_series(g, volume_col) if volume_col is not None else None

        safe_close = close.where(close > 0)

        feature_cols: dict[str, pd.Series] = {}

        if add_basic_returns:
            feature_cols["ret_1"] = close.pct_change(1).shift(1 if shift_features else 0)
            feature_cols["ret_5"] = close.pct_change(5).shift(1 if shift_features else 0)
            feature_cols["ret_10"] = close.pct_change(10).shift(1 if shift_features else 0)
            feature_cols["logret_1"] = np.log(safe_close).diff(1).shift(1 if shift_features else 0)
            feature_cols["logret_5"] = np.log(safe_close).diff(5).shift(1 if shift_features else 0)
            feature_cols["hl_spread"] = ((high - low) / close.replace(0, np.nan)).shift(1 if shift_features else 0)
            feature_cols["oc_spread"] = ((close - open_) / open_.replace(0, np.nan)).shift(1 if shift_features else 0)

        if add_trend:
            for win in ema_sma_windows:
                sma = _safe_talib_series(talib.SMA(close.values, timeperiod=win), g.index, shift_features)
                ema = _safe_talib_series(talib.EMA(close.values, timeperiod=win), g.index, shift_features)

                feature_cols[f"sma_{win}"] = sma
                feature_cols[f"ema_{win}"] = ema
                feature_cols[f"price_vs_sma{win}"] = close / sma - 1.0
                feature_cols[f"price_vs_ema{win}"] = close / ema - 1.0

            macd, macd_signal, macd_hist = talib.MACD(close.values, fastperiod=12, slowperiod=26, signalperiod=9)
            feature_cols["macd"] = _safe_talib_series(macd, g.index, shift_features)
            feature_cols["macd_signal"] = _safe_talib_series(macd_signal, g.index, shift_features)
            feature_cols["macd_hist"] = _safe_talib_series(macd_hist, g.index, shift_features)

        if add_momentum:
            feature_cols["rsi_14"] = _safe_talib_series(talib.RSI(close.values, timeperiod=14), g.index, shift_features)
            feature_cols["mom_10"] = _safe_talib_series(talib.MOM(close.values, timeperiod=10), g.index, shift_features)
            feature_cols["roc_10"] = _safe_talib_series(talib.ROC(close.values, timeperiod=10), g.index, shift_features)
            feature_cols["willr_14"] = _safe_talib_series(
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
            feature_cols["stoch_k"] = _safe_talib_series(stoch_k, g.index, shift_features)
            feature_cols["stoch_d"] = _safe_talib_series(stoch_d, g.index, shift_features)

        if add_volatility:
            atr_14 = _safe_talib_series(
                talib.ATR(high.values, low.values, close.values, timeperiod=14),
                g.index,
                shift_features,
            )
            natr_14 = _safe_talib_series(
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

            bb_upper = _safe_talib_series(upper, g.index, shift_features)
            bb_middle = _safe_talib_series(middle, g.index, shift_features)
            bb_lower = _safe_talib_series(lower, g.index, shift_features)
            adx_14 = _safe_talib_series(
                talib.ADX(high.values, low.values, close.values, timeperiod=14),
                g.index,
                shift_features,
            )

            feature_cols["atr_14"] = atr_14
            feature_cols["natr_14"] = natr_14
            feature_cols["bb_upper"] = bb_upper
            feature_cols["bb_middle"] = bb_middle
            feature_cols["bb_lower"] = bb_lower
            feature_cols["bb_width"] = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan)
            feature_cols["bb_pos"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
            feature_cols["adx_14"] = adx_14

            if "ema_80" in feature_cols:
                feature_cols["ema80_atr_dist"] = (close - feature_cols["ema_80"]) / atr_14.replace(0, np.nan)
            if "ema_100" in feature_cols:
                feature_cols["ema100_atr_dist"] = (close - feature_cols["ema_100"]) / atr_14.replace(0, np.nan)

        if add_volume and volume is not None:
            obv = _safe_talib_series(talib.OBV(close.values, volume.values), g.index, shift_features)
            vol_sma_20 = _safe_talib_series(talib.SMA(volume.values, timeperiod=20), g.index, shift_features)

            feature_cols["obv"] = obv
            feature_cols["vol_sma_20"] = vol_sma_20
            feature_cols["rel_volume_20"] = volume / vol_sma_20.replace(0, np.nan)
            feature_cols["volume_spike_20"] = (feature_cols["rel_volume_20"] >= 2.0).astype(float)

        date_utc = g[ts_col].dt.floor("D")
        hour_utc = g[ts_col].dt.hour
        minute_of_day_utc = g[ts_col].dt.hour * 60 + g[ts_col].dt.minute

        feature_cols["date_utc"] = date_utc
        feature_cols["hour_utc"] = hour_utc
        feature_cols["minute_of_day_utc"] = minute_of_day_utc
        feature_cols["is_london_ny"] = ((hour_utc >= 8) & (hour_utc <= 16)).astype(int)
        feature_cols["is_us_morning"] = ((hour_utc >= 13) & (hour_utc <= 17)).astype(int)

        session_high = g.groupby(date_utc)[high_col].max()
        session_low = g.groupby(date_utc)[low_col].min()

        prev_session_high = date_utc.map(session_high.shift(1)).astype(float)
        prev_session_low = date_utc.map(session_low.shift(1)).astype(float)

        feature_cols["prev_session_high"] = prev_session_high
        feature_cols["prev_session_low"] = prev_session_low
        feature_cols["dist_prev_session_high"] = close / prev_session_high - 1.0
        feature_cols["dist_prev_session_low"] = close / prev_session_low - 1.0

        or_high_5, or_low_5 = _opening_range_levels(g, ts_col=ts_col, high_col=high_col, low_col=low_col, minutes=5)
        or_high_15, or_low_15 = _opening_range_levels(g, ts_col=ts_col, high_col=high_col, low_col=low_col, minutes=15)

        feature_cols["opening_range_high_5m"] = or_high_5
        feature_cols["opening_range_low_5m"] = or_low_5
        feature_cols["opening_range_high_15m"] = or_high_15
        feature_cols["opening_range_low_15m"] = or_low_15

        feature_cols["dist_or_high_5m"] = close / or_high_5 - 1.0
        feature_cols["dist_or_low_5m"] = close / or_low_5 - 1.0
        feature_cols["dist_or_high_15m"] = close / or_high_15 - 1.0
        feature_cols["dist_or_low_15m"] = close / or_low_15 - 1.0

        rolling_high_30m = _rolling_level(high, window=1800, shift_features=shift_features, fn="max")
        rolling_low_30m = _rolling_level(low, window=1800, shift_features=shift_features, fn="min")
        rolling_high_60m = _rolling_level(high, window=3600, shift_features=shift_features, fn="max")
        rolling_low_60m = _rolling_level(low, window=3600, shift_features=shift_features, fn="min")

        feature_cols["rolling_high_30m"] = rolling_high_30m
        feature_cols["rolling_low_30m"] = rolling_low_30m
        feature_cols["rolling_high_60m"] = rolling_high_60m
        feature_cols["rolling_low_60m"] = rolling_low_60m

        feature_cols["dist_rolling_high_30m"] = close / rolling_high_30m - 1.0
        feature_cols["dist_rolling_low_30m"] = close / rolling_low_30m - 1.0
        feature_cols["dist_rolling_high_60m"] = close / rolling_high_60m - 1.0
        feature_cols["dist_rolling_low_60m"] = close / rolling_low_60m - 1.0

        bullish_fvg = low > high.shift(2)
        bearish_fvg = high < low.shift(2)

        bullish_fvg_lower = high.shift(2).where(bullish_fvg)
        bullish_fvg_upper = low.where(bullish_fvg)
        bearish_fvg_lower = high.where(bearish_fvg)
        bearish_fvg_upper = low.shift(2).where(bearish_fvg)

        feature_cols["bullish_fvg"] = bullish_fvg.astype(int)
        feature_cols["bearish_fvg"] = bearish_fvg.astype(int)
        feature_cols["bullish_fvg_lower"] = bullish_fvg_lower
        feature_cols["bullish_fvg_upper"] = bullish_fvg_upper
        feature_cols["bearish_fvg_lower"] = bearish_fvg_lower
        feature_cols["bearish_fvg_upper"] = bearish_fvg_upper
        feature_cols["inside_bullish_fvg"] = (
            (close >= bullish_fvg_lower) & (close <= bullish_fvg_upper)
        ).fillna(False).astype(int)
        feature_cols["inside_bearish_fvg"] = (
            (close >= bearish_fvg_lower) & (close <= bearish_fvg_upper)
        ).fillna(False).astype(int)

        feature_df = pd.DataFrame(feature_cols, index=g.index)
        g_out = pd.concat([g, feature_df], axis=1).copy()
        out.append(g_out)

    return pd.concat(out, axis=0, ignore_index=True)


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
    df = df.dropna(subset=[ts_col]).sort_values([symbol_col, ts_col]).reset_index(drop=True)

    out = []

    for _, g in df.groupby(symbol_col, sort=False):
        g = g.copy()
        close = pd.to_numeric(g[close_col], errors="coerce").astype(float)

        target_cols = {}
        safe_close = close.where(close > 0)

        for h in horizons:
            target_cols[f"target_fwd_ret_{h}s"] = close.shift(-h) / close - 1.0
            target_cols[f"target_fwd_logret_{h}s"] = np.log(safe_close.shift(-h)) - np.log(safe_close)

        g_out = pd.concat([g, pd.DataFrame(target_cols, index=g.index)], axis=1).copy()
        out.append(g_out)

    return pd.concat(out, axis=0, ignore_index=True)