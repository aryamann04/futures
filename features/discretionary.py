from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from features.atr import add_atr_features
from features.confluence import add_confluence_features
from features.fvg import add_fvg_features
from features.session_levels import add_session_level_features
from features.structure import add_market_structure_features
from features.sweeps import detect_sweeps_and_reclaims
from features.volume_profile import add_volume_profile_features
from features.vwap import add_vwap_features
from features.resample import resample_ohlcv, timeframe_to_timedelta


def _normalize_timeframe(value: str) -> str:
    aliases = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
        "60m": "1h",
        "native": "native",
    }
    return aliases.get(value, value)


def _eligible_timeframes(base_timeframe: str, requested: Sequence[str]) -> tuple[str, ...]:
    if base_timeframe == "native":
        return tuple(dict.fromkeys(_normalize_timeframe(tf) for tf in requested))
    base_td = timeframe_to_timedelta(base_timeframe)
    eligible = [_normalize_timeframe(tf) for tf in requested if timeframe_to_timedelta(tf) >= base_td]
    if not eligible:
        eligible = [_normalize_timeframe(base_timeframe)]
    return tuple(dict.fromkeys(eligible))


def build_discretionary_features(
    df: pd.DataFrame,
    *,
    atr_periods: tuple[int, ...] = (14,),
    fvg_timeframes: Sequence[str] = ("1min", "5min", "15min", "1h"),
    base_timeframe: str = "native",
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_timeframe = _normalize_timeframe(base_timeframe)
    out = df.copy()
    if base_timeframe != "native":
        out = resample_ohlcv(out, base_timeframe, ts_col=ts_col, symbol_col=symbol_col)
    else:
        out = out.copy()

    active_fvg_timeframes = _eligible_timeframes(base_timeframe, fvg_timeframes)
    atr_timeframes = _eligible_timeframes(base_timeframe, ("1min", "5min", "15min", "1h"))

    out = add_session_level_features(out, ts_col=ts_col, symbol_col=symbol_col)
    out = add_atr_features(out, periods=atr_periods, timeframes=atr_timeframes, ts_col=ts_col, symbol_col=symbol_col)

    base_atr_prefix = f"atr_{base_timeframe}_"
    atr_col = next((c for c in out.columns if c.startswith(base_atr_prefix)), None)
    if atr_col is None:
        atr_col = next((c for c in out.columns if c.startswith("atr_") and c.endswith("_14")), None)
    if atr_col is not None and "atr_1min_14" not in out.columns:
        out["atr_1min_14"] = out[atr_col]
        regime_col = f"{atr_col}_regime"
        if regime_col in out.columns and "atr_1min_14_regime" not in out.columns:
            out["atr_1min_14_regime"] = out[regime_col]

    out = add_vwap_features(out, ts_col=ts_col, symbol_col=symbol_col, atr_col=atr_col)
    out = add_market_structure_features(out, ts_col=ts_col, symbol_col=symbol_col)
    out = add_volume_profile_features(out, ts_col=ts_col, symbol_col=symbol_col)
    out, fvg_events = add_fvg_features(out, timeframes=active_fvg_timeframes, ts_col=ts_col, symbol_col=symbol_col)
    out = add_confluence_features(out, atr_col=atr_col)

    sweep_levels = [
        "prev_day_high",
        "prev_day_low",
        "asia_high",
        "asia_low",
        "london_high",
        "london_low",
        "ny_open_range_high",
        "ny_open_range_low",
        "last_confirmed_swing_high",
        "last_confirmed_swing_low",
    ]
    sweep_events = detect_sweeps_and_reclaims(
        out,
        level_columns=sweep_levels,
        structure_bull_col="choch_bullish",
        structure_bear_col="choch_bearish",
    )
    return out, fvg_events, sweep_events
