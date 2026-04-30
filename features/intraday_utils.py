from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


DEFAULT_TIMEZONE = "America/New_York"


@dataclass(frozen=True)
class SessionWindow:
    name: str
    start: str
    end: str


DEFAULT_SESSION_WINDOWS: tuple[SessionWindow, ...] = (
    SessionWindow(name="asia", start="18:00", end="03:00"),
    SessionWindow(name="london", start="03:00", end="08:30"),
    SessionWindow(name="new_york", start="08:30", end="16:00"),
)


def pick_column(columns: Iterable[str], candidates: list[str], required: bool = True) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def ensure_timestamp_series(ts: pd.Series) -> pd.Series:
    out = pd.to_datetime(ts, errors="coerce")
    if getattr(out.dt, "tz", None) is None:
        out = out.dt.tz_localize("UTC")
    return out


def to_timezone(ts: pd.Series, timezone: str = DEFAULT_TIMEZONE) -> pd.Series:
    return ensure_timestamp_series(ts).dt.tz_convert(timezone)


def parse_hhmm(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def time_window_mask(local_ts: pd.Series, start: str, end: str) -> pd.Series:
    start_min = parse_hhmm(start)
    end_min = parse_hhmm(end)
    minute_of_day = local_ts.dt.hour * 60 + local_ts.dt.minute

    if start_min <= end_min:
        mask = (minute_of_day >= start_min) & (minute_of_day < end_min)
    else:
        mask = (minute_of_day >= start_min) | (minute_of_day < end_min)
    return mask.fillna(False)


def trading_day_key(local_ts: pd.Series, roll_hour: int = 18) -> pd.Series:
    shifted = local_ts - pd.to_timedelta((local_ts.dt.hour < roll_hour).astype(int), unit="D")
    return shifted.dt.floor("D")


def group_key(
    df: pd.DataFrame,
    ts_col: str,
    symbol_col: Optional[str] = None,
    timezone: str = DEFAULT_TIMEZONE,
    roll_hour: int = 18,
) -> pd.Series:
    local_ts = to_timezone(df[ts_col], timezone)
    day_key = trading_day_key(local_ts, roll_hour=roll_hour).astype(str)
    if symbol_col is None:
        return day_key
    return df[symbol_col].astype(str) + "|" + day_key

