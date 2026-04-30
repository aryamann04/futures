from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from features.atr import compute_atr
from features.intraday_utils import ensure_timestamp_series, pick_column
from features.resample import infer_base_timedelta, resample_ohlcv, timeframe_to_timedelta


def detect_fvg_events(
    df: pd.DataFrame,
    timeframe: str,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
    atr_period: int = 14,
) -> pd.DataFrame:
    base = df.copy()
    base.columns = [c.lower() for c in base.columns]
    ts_col = ts_col or pick_column(base.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    symbol_col = symbol_col or pick_column(base.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"], required=False)
    base_td = infer_base_timedelta(base, ts_col)
    target_td = timeframe_to_timedelta(timeframe)
    bars = base if base_td == target_td else resample_ohlcv(base, timeframe, ts_col=ts_col, symbol_col=symbol_col)
    bars.columns = [c.lower() for c in bars.columns]
    bars[ts_col] = ensure_timestamp_series(bars[ts_col])
    bars = bars.sort_values(([symbol_col] if symbol_col else []) + [ts_col]).reset_index(drop=True)
    bars["atr_ref"] = compute_atr(bars, period=atr_period, ts_col=ts_col, symbol_col=symbol_col)

    events: list[dict[str, object]] = []
    groups = [(None, bars)] if symbol_col is None else bars.groupby(symbol_col, sort=False)
    for symbol, group in groups:
        group = group.reset_index(drop=True)
        for i in range(2, len(group)):
            left = group.iloc[i - 2]
            current = group.iloc[i]
            if left["high"] < current["low"]:
                lower = float(left["high"])
                upper = float(current["low"])
                direction = "bullish"
            elif left["low"] > current["high"]:
                lower = float(current["high"])
                upper = float(left["low"])
                direction = "bearish"
            else:
                continue
            size = upper - lower
            atr_value = float(group.iloc[i]["atr_ref"]) if np.isfinite(group.iloc[i]["atr_ref"]) else np.nan
            events.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "direction": direction,
                    "creation_time": current[ts_col],
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "midpoint": (lower + upper) / 2.0,
                    "size_points": size,
                    "size_atr": size / atr_value if np.isfinite(atr_value) and atr_value != 0 else np.nan,
                }
            )

    return pd.DataFrame(events)


def add_fvg_features(
    df: pd.DataFrame,
    timeframes: Sequence[str] = ("1min", "5min", "15min", "1h"),
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project active FVG state onto the base timeframe without exposing future bars."""
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    ts_col = ts_col or pick_column(out.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    symbol_col = symbol_col or pick_column(out.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"], required=False)
    out[ts_col] = ensure_timestamp_series(out[ts_col])
    out = out.sort_values(([symbol_col] if symbol_col else []) + [ts_col]).reset_index(drop=True)

    all_events: list[pd.DataFrame] = []
    for timeframe in timeframes:
        events = detect_fvg_events(out, timeframe=timeframe, ts_col=ts_col, symbol_col=symbol_col)
        if events.empty:
            continue
        events = events.sort_values((["symbol"] if symbol_col else []) + ["creation_time"]).reset_index(drop=True)
        all_events.append(events)

        prefix = timeframe.replace("min", "m").replace("h", "h")
        for col in [
            f"nearest_bullish_fvg_below_{prefix}",
            f"nearest_bearish_fvg_above_{prefix}",
            f"inside_any_fvg_{prefix}",
            f"touches_fvg_mid_{prefix}",
            f"active_bullish_fvg_count_{prefix}",
            f"active_bearish_fvg_count_{prefix}",
        ]:
            out[col] = np.nan if "nearest" in col else 0

        groups = [(None, out)] if symbol_col is None else out.groupby(symbol_col, sort=False)
        event_groups = {key: grp.copy() for key, grp in ([(None, events)] if symbol_col is None else events.groupby("symbol", sort=False))}
        for symbol, group in groups:
            active: list[dict[str, object]] = []
            symbol_events = event_groups.get(symbol, pd.DataFrame())
            event_ptr = 0
            symbol_events = symbol_events.reset_index(drop=True)
            for global_i, row in group.iterrows():
                while event_ptr < len(symbol_events) and symbol_events.loc[event_ptr, "creation_time"] <= row[ts_col]:
                    event = symbol_events.loc[event_ptr].to_dict()
                    event.update(
                        {
                            "active": True,
                            "filled_pct": 0.0,
                            "first_touch_time": pd.NaT,
                            "full_fill_time": pd.NaT,
                            "age_bars": 0,
                        }
                    )
                    active.append(event)
                    event_ptr += 1

                for event in active:
                    if not event["active"]:
                        continue
                    event["age_bars"] += 1
                    lower = float(event["lower_bound"])
                    upper = float(event["upper_bound"])
                    overlap_low = max(lower, float(row["low"]))
                    overlap_high = min(upper, float(row["high"]))
                    if overlap_low <= overlap_high:
                        fill = (overlap_high - overlap_low) / max(upper - lower, 1e-12)
                        event["filled_pct"] = max(float(event["filled_pct"]), float(fill))
                        if pd.isna(event["first_touch_time"]):
                            event["first_touch_time"] = row[ts_col]
                        if event["direction"] == "bullish" and float(row["low"]) <= lower:
                            event["active"] = False
                            event["filled_pct"] = 1.0
                            event["full_fill_time"] = row[ts_col]
                        if event["direction"] == "bearish" and float(row["high"]) >= upper:
                            event["active"] = False
                            event["filled_pct"] = 1.0
                            event["full_fill_time"] = row[ts_col]

                active_bullish = [e for e in active if e["active"] and e["direction"] == "bullish"]
                active_bearish = [e for e in active if e["active"] and e["direction"] == "bearish"]

                bullish_below = [e for e in active_bullish if float(e["upper_bound"]) <= float(row["close"])]
                bearish_above = [e for e in active_bearish if float(e["lower_bound"]) >= float(row["close"])]
                if bullish_below:
                    nearest = max(bullish_below, key=lambda e: float(e["upper_bound"]))
                    out.at[global_i, f"nearest_bullish_fvg_below_{prefix}"] = nearest["midpoint"]
                if bearish_above:
                    nearest = min(bearish_above, key=lambda e: float(e["lower_bound"]))
                    out.at[global_i, f"nearest_bearish_fvg_above_{prefix}"] = nearest["midpoint"]

                inside = any(float(e["lower_bound"]) <= float(row["close"]) <= float(e["upper_bound"]) for e in active)
                touches_mid = any(float(row["low"]) <= float(e["midpoint"]) <= float(row["high"]) for e in active)
                out.at[global_i, f"inside_any_fvg_{prefix}"] = int(inside)
                out.at[global_i, f"touches_fvg_mid_{prefix}"] = int(touches_mid)
                out.at[global_i, f"active_bullish_fvg_count_{prefix}"] = len(active_bullish)
                out.at[global_i, f"active_bearish_fvg_count_{prefix}"] = len(active_bearish)

            if active:
                finalized = pd.DataFrame(active)
                if not finalized.empty:
                    all_events.append(finalized)

    combined = pd.concat(all_events, axis=0, ignore_index=True) if all_events else pd.DataFrame()
    return out, combined
