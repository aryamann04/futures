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


def _to_float_series(df: pd.DataFrame, col: Optional[str], dtype: str = "float64") -> Optional[pd.Series]:
    if col is None:
        return None
    return pd.to_numeric(df[col], errors="coerce").astype(dtype)


def _safe_talib_series(values, index, shift=True, dtype: str = "float64"):
    s = pd.Series(values, index=index, dtype=dtype)
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

    return mapped_high.astype("float32"), mapped_low.astype("float32")


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="float")
        elif pd.api.types.is_integer_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="integer")
    return out


def _add_fib_features(
    feature_cols: dict[str, pd.Series],
    close: pd.Series,
    swing_high: pd.Series,
    swing_low: pd.Series,
    prefix: str,
) -> None:
    rng = (swing_high - swing_low).replace(0, np.nan)

    fib_236 = (swing_high - 0.236 * rng).astype("float32")
    fib_382 = (swing_high - 0.382 * rng).astype("float32")
    fib_500 = (swing_high - 0.500 * rng).astype("float32")
    fib_618 = (swing_high - 0.618 * rng).astype("float32")
    fib_786 = (swing_high - 0.786 * rng).astype("float32")

    feature_cols[f"{prefix}_swing_high"] = swing_high.astype("float32")
    feature_cols[f"{prefix}_swing_low"] = swing_low.astype("float32")
    feature_cols[f"{prefix}_range"] = rng.astype("float32")

    feature_cols[f"{prefix}_fib_236"] = fib_236
    feature_cols[f"{prefix}_fib_382"] = fib_382
    feature_cols[f"{prefix}_fib_500"] = fib_500
    feature_cols[f"{prefix}_fib_618"] = fib_618
    feature_cols[f"{prefix}_fib_786"] = fib_786

    feature_cols[f"{prefix}_range_pos"] = ((close - swing_low) / rng).astype("float32")

    feature_cols[f"{prefix}_dist_fib_236"] = (close / fib_236 - 1.0).astype("float32")
    feature_cols[f"{prefix}_dist_fib_382"] = (close / fib_382 - 1.0).astype("float32")
    feature_cols[f"{prefix}_dist_fib_500"] = (close / fib_500 - 1.0).astype("float32")
    feature_cols[f"{prefix}_dist_fib_618"] = (close / fib_618 - 1.0).astype("float32")
    feature_cols[f"{prefix}_dist_fib_786"] = (close / fib_786 - 1.0).astype("float32")

    feature_cols[f"{prefix}_in_fib_zone_382_618"] = (
        ((close <= fib_382) & (close >= fib_618)).astype("int8")
    )
    feature_cols[f"{prefix}_in_fib_zone_500_618"] = (
        ((close <= fib_500) & (close >= fib_618)).astype("int8")
    )


def _bars_for_minutes(minutes: int, bar_seconds: int) -> int:
    return max(1, int((minutes * 60) / bar_seconds))


def _bars_for_hours(hours: int, bar_seconds: int) -> int:
    return max(1, int((hours * 3600) / bar_seconds))


def _bars_for_days(days: int, bar_seconds: int) -> int:
    return max(1, int((days * 86400) / bar_seconds))


def build_features(
    df: pd.DataFrame,
    symbol_col: Optional[str] = None,
    ts_col: Optional[str] = None,
    add_basic_returns: bool = True,
    add_trend: bool = True,
    add_momentum: bool = True,
    add_volatility: bool = True,
    add_volume: bool = True,
    add_session_levels: bool = True,
    add_opening_ranges: bool = True,
    add_rolling_ranges: bool = True,
    add_fvg: bool = True,
    shift_features: bool = True,
    bar_seconds: int = 1,
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

    ema_sma_windows = [
        20, 50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 150, 200, 300, 400, 600,
        900, 1200, 1800, 3600
    ]
    rsi_windows = [14, 21, 50, 100, 300, 600]
    roc_windows = [10, 30, 60, 300, 900, 1800, 3600]
    atr_windows = [14, 50, 100, 300]
    adx_windows = [14, 50, 100]
    long_return_windows = [30, 60, 300, 900, 1800, 3600, 7200, 14400]

    rolling_windows = {
        "30m": _bars_for_minutes(30, bar_seconds),
        "60m": _bars_for_minutes(60, bar_seconds),
        "2h": _bars_for_hours(2, bar_seconds),
        "4h": _bars_for_hours(4, bar_seconds),
        "8h": _bars_for_hours(8, bar_seconds),
        "1d": _bars_for_days(1, bar_seconds),
    }

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
            feature_cols["ret_1"] = close.pct_change(1, fill_method=None).shift(1 if shift_features else 0).astype("float32")
            feature_cols["ret_5"] = close.pct_change(5, fill_method=None).shift(1 if shift_features else 0).astype("float32")
            feature_cols["ret_10"] = close.pct_change(10, fill_method=None).shift(1 if shift_features else 0).astype("float32")
            feature_cols["logret_1"] = np.log(safe_close).diff(1).shift(1 if shift_features else 0).astype("float32")
            feature_cols["logret_5"] = np.log(safe_close).diff(5).shift(1 if shift_features else 0).astype("float32")
            feature_cols["hl_spread"] = ((high - low) / close.replace(0, np.nan)).shift(1 if shift_features else 0).astype("float32")
            feature_cols["oc_spread"] = ((close - open_) / open_.replace(0, np.nan)).shift(1 if shift_features else 0).astype("float32")

            for win in long_return_windows:
                feature_cols[f"ret_{win}"] = close.pct_change(win, fill_method=None).shift(1 if shift_features else 0).astype("float32")
                feature_cols[f"logret_{win}"] = np.log(safe_close).diff(win).shift(1 if shift_features else 0).astype("float32")

        if add_trend:
            for win in ema_sma_windows:
                sma = _safe_talib_series(talib.SMA(close.values, timeperiod=win), g.index, shift_features)
                ema = _safe_talib_series(talib.EMA(close.values, timeperiod=win), g.index, shift_features)

                feature_cols[f"sma_{win}"] = sma
                feature_cols[f"ema_{win}"] = ema
                feature_cols[f"price_vs_sma{win}"] = (close / sma - 1.0).astype("float32")
                feature_cols[f"price_vs_ema{win}"] = (close / ema - 1.0).astype("float32")
                feature_cols[f"ema_slope_{win}"] = ema.diff(1).astype("float32")
                feature_cols[f"ema_slope_pct_{win}"] = ema.pct_change(1, fill_method=None).astype("float32")

            macd, macd_signal, macd_hist = talib.MACD(close.values, fastperiod=12, slowperiod=26, signalperiod=9)
            macd_s = _safe_talib_series(macd, g.index, shift_features)
            macd_signal_s = _safe_talib_series(macd_signal, g.index, shift_features)
            macd_hist_s = _safe_talib_series(macd_hist, g.index, shift_features)

            feature_cols["macd"] = macd_s
            feature_cols["macd_signal"] = macd_signal_s
            feature_cols["macd_hist"] = macd_hist_s
            feature_cols["macd_above_signal"] = (macd_s > macd_signal_s).astype("int8")
            feature_cols["macd_below_signal"] = (macd_s < macd_signal_s).astype("int8")
            feature_cols["macd_cross_up"] = ((macd_s > macd_signal_s) & (macd_s.shift(1) <= macd_signal_s.shift(1))).astype("int8")
            feature_cols["macd_cross_down"] = ((macd_s < macd_signal_s) & (macd_s.shift(1) >= macd_signal_s.shift(1))).astype("int8")
            feature_cols["macd_hist_slope"] = macd_hist_s.diff(1).astype("float32")
            feature_cols["macd_zero_cross_up"] = ((macd_s > 0) & (macd_s.shift(1) <= 0)).astype("int8")
            feature_cols["macd_zero_cross_down"] = ((macd_s < 0) & (macd_s.shift(1) >= 0)).astype("int8")

            if "ema_100" in feature_cols and "ema_300" in feature_cols:
                feature_cols["ema100_gt_ema300"] = (feature_cols["ema_100"] > feature_cols["ema_300"]).astype("int8")
                feature_cols["ema100_lt_ema300"] = (feature_cols["ema_100"] < feature_cols["ema_300"]).astype("int8")
                feature_cols["ema100_300_spread"] = (
                    feature_cols["ema_100"] / feature_cols["ema_300"] - 1.0
                ).astype("float32")

            if "ema_300" in feature_cols and "ema_600" in feature_cols:
                feature_cols["ema300_gt_ema600"] = (feature_cols["ema_300"] > feature_cols["ema_600"]).astype("int8")
                feature_cols["ema300_lt_ema600"] = (feature_cols["ema_300"] < feature_cols["ema_600"]).astype("int8")
                feature_cols["ema300_600_spread"] = (
                    feature_cols["ema_300"] / feature_cols["ema_600"] - 1.0
                ).astype("float32")

        if add_momentum:
            for win in rsi_windows:
                feature_cols[f"rsi_{win}"] = _safe_talib_series(
                    talib.RSI(close.values, timeperiod=win),
                    g.index,
                    shift_features,
                )

            feature_cols["mom_10"] = _safe_talib_series(talib.MOM(close.values, timeperiod=10), g.index, shift_features)
            feature_cols["mom_60"] = _safe_talib_series(talib.MOM(close.values, timeperiod=60), g.index, shift_features)
            feature_cols["mom_300"] = _safe_talib_series(talib.MOM(close.values, timeperiod=300), g.index, shift_features)

            for win in roc_windows:
                feature_cols[f"roc_{win}"] = _safe_talib_series(
                    talib.ROC(close.values, timeperiod=win),
                    g.index,
                    shift_features,
                )

            feature_cols["willr_14"] = _safe_talib_series(
                talib.WILLR(high.values, low.values, close.values, timeperiod=14),
                g.index,
                shift_features,
            )
            feature_cols["willr_50"] = _safe_talib_series(
                talib.WILLR(high.values, low.values, close.values, timeperiod=50),
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
            for win in atr_windows:
                atr = _safe_talib_series(
                    talib.ATR(high.values, low.values, close.values, timeperiod=win),
                    g.index,
                    shift_features,
                )
                natr = _safe_talib_series(
                    talib.NATR(high.values, low.values, close.values, timeperiod=win),
                    g.index,
                    shift_features,
                )
                feature_cols[f"atr_{win}"] = atr
                feature_cols[f"natr_{win}"] = natr

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

            feature_cols["bb_upper"] = bb_upper
            feature_cols["bb_middle"] = bb_middle
            feature_cols["bb_lower"] = bb_lower
            feature_cols["bb_width"] = ((bb_upper - bb_lower) / bb_middle.replace(0, np.nan)).astype("float32")
            feature_cols["bb_pos"] = ((close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)).astype("float32")

            for win in adx_windows:
                feature_cols[f"adx_{win}"] = _safe_talib_series(
                    talib.ADX(high.values, low.values, close.values, timeperiod=win),
                    g.index,
                    shift_features,
                )

            if "macd" in feature_cols and "macd_hist" in feature_cols and "atr_14" in feature_cols:
                feature_cols["macd_atr_norm"] = (
                    feature_cols["macd"] / feature_cols["atr_14"].replace(0, np.nan)
                ).astype("float32")
                feature_cols["macd_hist_atr_norm"] = (
                    feature_cols["macd_hist"] / feature_cols["atr_14"].replace(0, np.nan)
                ).astype("float32")

            if "ema_80" in feature_cols and "atr_14" in feature_cols:
                feature_cols["ema80_atr_dist"] = (
                    (close - feature_cols["ema_80"]) / feature_cols["atr_14"].replace(0, np.nan)
                ).astype("float32")
            if "ema_100" in feature_cols and "atr_14" in feature_cols:
                feature_cols["ema100_atr_dist"] = (
                    (close - feature_cols["ema_100"]) / feature_cols["atr_14"].replace(0, np.nan)
                ).astype("float32")
            if "ema_300" in feature_cols and "atr_50" in feature_cols:
                feature_cols["ema300_atr_dist"] = (
                    (close - feature_cols["ema_300"]) / feature_cols["atr_50"].replace(0, np.nan)
                ).astype("float32")
            if "ema_600" in feature_cols and "atr_100" in feature_cols:
                feature_cols["ema600_atr_dist"] = (
                    (close - feature_cols["ema_600"]) / feature_cols["atr_100"].replace(0, np.nan)
                ).astype("float32")

        if add_volume and volume is not None:
            obv = _safe_talib_series(talib.OBV(close.values, volume.values), g.index, shift_features)
            vol_sma_20 = _safe_talib_series(talib.SMA(volume.values, timeperiod=20), g.index, shift_features)
            vol_sma_300 = _safe_talib_series(talib.SMA(volume.values, timeperiod=300), g.index, shift_features)
            vol_sma_3600 = _safe_talib_series(talib.SMA(volume.values, timeperiod=3600), g.index, shift_features)

            rel_volume_20 = (volume / vol_sma_20.replace(0, np.nan)).astype("float32")
            rel_volume_300 = (volume / vol_sma_300.replace(0, np.nan)).astype("float32")
            rel_volume_3600 = (volume / vol_sma_3600.replace(0, np.nan)).astype("float32")

            feature_cols["obv"] = obv
            feature_cols["vol_sma_20"] = vol_sma_20
            feature_cols["vol_sma_300"] = vol_sma_300
            feature_cols["vol_sma_3600"] = vol_sma_3600
            feature_cols["rel_volume_20"] = rel_volume_20
            feature_cols["rel_volume_300"] = rel_volume_300
            feature_cols["rel_volume_3600"] = rel_volume_3600
            feature_cols["volume_spike_20"] = (rel_volume_20 >= 2.0).astype("int8")
            feature_cols["volume_spike_300"] = (rel_volume_300 >= 2.0).astype("int8")

        if add_session_levels or add_opening_ranges:
            date_utc = g[ts_col].dt.floor("D")
            hour_utc = g[ts_col].dt.hour.astype("int16")
            minute_of_day_utc = (g[ts_col].dt.hour * 60 + g[ts_col].dt.minute).astype("int16")

            feature_cols["date_utc"] = date_utc
            feature_cols["hour_utc"] = hour_utc
            feature_cols["minute_of_day_utc"] = minute_of_day_utc
            feature_cols["is_london_ny"] = ((hour_utc >= 8) & (hour_utc <= 16)).astype("int8")
            feature_cols["is_us_morning"] = ((hour_utc >= 13) & (hour_utc <= 17)).astype("int8")

        if add_session_levels:
            session_high = g.groupby(date_utc)[high_col].max()
            session_low = g.groupby(date_utc)[low_col].min()

            prev_session_high = date_utc.map(session_high.shift(1)).astype("float32")
            prev_session_low = date_utc.map(session_low.shift(1)).astype("float32")

            feature_cols["prev_session_high"] = prev_session_high
            feature_cols["prev_session_low"] = prev_session_low
            feature_cols["dist_prev_session_high"] = (close / prev_session_high - 1.0).astype("float32")
            feature_cols["dist_prev_session_low"] = (close / prev_session_low - 1.0).astype("float32")

        if add_opening_ranges:
            or_high_5, or_low_5 = _opening_range_levels(g, ts_col=ts_col, high_col=high_col, low_col=low_col, minutes=5)
            or_high_15, or_low_15 = _opening_range_levels(g, ts_col=ts_col, high_col=high_col, low_col=low_col, minutes=15)

            feature_cols["opening_range_high_5m"] = or_high_5
            feature_cols["opening_range_low_5m"] = or_low_5
            feature_cols["opening_range_high_15m"] = or_high_15
            feature_cols["opening_range_low_15m"] = or_low_15

            feature_cols["dist_or_high_5m"] = (close / or_high_5 - 1.0).astype("float32")
            feature_cols["dist_or_low_5m"] = (close / or_low_5 - 1.0).astype("float32")
            feature_cols["dist_or_high_15m"] = (close / or_high_15 - 1.0).astype("float32")
            feature_cols["dist_or_low_15m"] = (close / or_low_15 - 1.0).astype("float32")

        if add_rolling_ranges:
            for label, win in rolling_windows.items():
                rh = _rolling_level(high, window=win, shift_features=shift_features, fn="max").astype("float32")
                rl = _rolling_level(low, window=win, shift_features=shift_features, fn="min").astype("float32")

                feature_cols[f"rolling_high_{label}"] = rh
                feature_cols[f"rolling_low_{label}"] = rl
                feature_cols[f"dist_rolling_high_{label}"] = (close / rh - 1.0).astype("float32")
                feature_cols[f"dist_rolling_low_{label}"] = (close / rl - 1.0).astype("float32")

            _add_fib_features(
                feature_cols,
                close=close,
                swing_high=feature_cols["rolling_high_4h"],
                swing_low=feature_cols["rolling_low_4h"],
                prefix="fib_4h",
            )
            _add_fib_features(
                feature_cols,
                close=close,
                swing_high=feature_cols["rolling_high_8h"],
                swing_low=feature_cols["rolling_low_8h"],
                prefix="fib_8h",
            )
            _add_fib_features(
                feature_cols,
                close=close,
                swing_high=feature_cols["rolling_high_1d"],
                swing_low=feature_cols["rolling_low_1d"],
                prefix="fib_1d",
            )

            feature_cols["trend_range_4h"] = (
                feature_cols["rolling_high_4h"] / feature_cols["rolling_low_4h"] - 1.0
            ).astype("float32")
            feature_cols["trend_range_8h"] = (
                feature_cols["rolling_high_8h"] / feature_cols["rolling_low_8h"] - 1.0
            ).astype("float32")
            feature_cols["trend_range_1d"] = (
                feature_cols["rolling_high_1d"] / feature_cols["rolling_low_1d"] - 1.0
            ).astype("float32")

        if add_fvg:
            bullish_fvg = (low > high.shift(2))
            bearish_fvg = (high < low.shift(2))

            bullish_fvg_lower = high.shift(2).where(bullish_fvg).astype("float32")
            bullish_fvg_upper = low.where(bullish_fvg).astype("float32")
            bearish_fvg_lower = high.where(bearish_fvg).astype("float32")
            bearish_fvg_upper = low.shift(2).where(bearish_fvg).astype("float32")

            feature_cols["bullish_fvg"] = bullish_fvg.astype("int8")
            feature_cols["bearish_fvg"] = bearish_fvg.astype("int8")
            feature_cols["bullish_fvg_lower"] = bullish_fvg_lower
            feature_cols["bullish_fvg_upper"] = bullish_fvg_upper
            feature_cols["bearish_fvg_lower"] = bearish_fvg_lower
            feature_cols["bearish_fvg_upper"] = bearish_fvg_upper
            feature_cols["inside_bullish_fvg"] = (
                ((close >= bullish_fvg_lower) & (close <= bullish_fvg_upper)).astype("int8")
            )
            feature_cols["inside_bearish_fvg"] = (
                ((close >= bearish_fvg_lower) & (close <= bearish_fvg_upper)).astype("int8")
            )

        feature_df = pd.DataFrame(feature_cols, index=g.index)
        g_out = pd.concat([g, feature_df], axis=1)
        g_out = _downcast_numeric(g_out)
        out.append(g_out)

    final = pd.concat(out, axis=0, ignore_index=True)
    final = _downcast_numeric(final)
    return final


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
        close = pd.to_numeric(g[close_col], errors="coerce").astype("float32")
        safe_close = close.where(close > 0)

        target_cols = {}
        for h in horizons:
            target_cols[f"target_fwd_ret_{h}s"] = (close.shift(-h) / close - 1.0).astype("float32")
            target_cols[f"target_fwd_logret_{h}s"] = (np.log(safe_close.shift(-h)) - np.log(safe_close)).astype("float32")

        g_out = pd.concat([g, pd.DataFrame(target_cols, index=g.index)], axis=1)
        g_out = _downcast_numeric(g_out)
        out.append(g_out)

    final = pd.concat(out, axis=0, ignore_index=True)
    final = _downcast_numeric(final)
    return final