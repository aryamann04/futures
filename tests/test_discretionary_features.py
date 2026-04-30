from __future__ import annotations

import pandas as pd

from features.fvg import add_fvg_features
from features.session_levels import add_session_level_features
from features.structure import add_market_structure_features
from features.vwap import add_vwap_features


def _base_intraday_df() -> pd.DataFrame:
    times = pd.date_range("2025-01-01 23:00:00+00:00", periods=8, freq="1h")
    return pd.DataFrame(
        {
            "ts_event": times,
            "symbol": ["MESH5"] * len(times),
            "open": [100, 101, 102, 103, 104, 105, 106, 107],
            "high": [101, 103, 104, 105, 106, 107, 108, 109],
            "low": [99, 100, 101, 102, 103, 104, 105, 106],
            "close": [100.5, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "volume": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )


def test_session_levels_only_use_completed_prior_day() -> None:
    df = _base_intraday_df()
    out = add_session_level_features(df)
    assert out["prev_day_high"].isna().all()

    day2 = df.copy()
    day2["ts_event"] = day2["ts_event"] + pd.Timedelta(days=1)
    merged = pd.concat([df, day2], ignore_index=True)
    out2 = add_session_level_features(merged)
    second_day = out2[out2["trading_day"] == out2["trading_day"].max()]
    assert second_day["prev_day_high"].notna().all()
    assert float(second_day["prev_day_high"].iloc[0]) == 109.0
    assert float(second_day["prev_day_low"].iloc[0]) == 99.0


def test_vwap_resets_by_trading_day() -> None:
    df = _base_intraday_df()
    df2 = df.copy()
    df2["ts_event"] = df2["ts_event"] + pd.Timedelta(days=1)
    out = add_vwap_features(pd.concat([df, df2], ignore_index=True))
    first = out.iloc[0]["vwap"]
    ninth = out.iloc[len(df)]["vwap"]
    typical = (df.iloc[0]["high"] + df.iloc[0]["low"] + df.iloc[0]["close"]) / 3.0
    assert round(first, 6) == round(typical, 6)
    assert round(ninth, 6) == round(typical, 6)


def test_structure_confirmation_is_delayed_by_right_bars() -> None:
    times = pd.date_range("2025-01-01 14:30:00+00:00", periods=7, freq="1min")
    df = pd.DataFrame(
        {
            "ts_event": times,
            "symbol": ["MESH5"] * 7,
            "open": [1, 2, 3, 4, 3, 2, 1],
            "high": [1, 2, 3, 5, 3, 2, 1],
            "low": [0, 1, 2, 3, 2, 1, 0],
            "close": [1, 2, 3, 4, 3, 2, 1],
        }
    )
    out = add_market_structure_features(df, swing_left=2, swing_right=2)
    assert int(out.loc[3, "swing_high_confirmed"]) == 0
    assert int(out.loc[5, "swing_high_confirmed"]) == 1
    assert float(out.loc[5, "swing_high_price"]) == 5.0


def test_fvg_available_only_after_creation_bar() -> None:
    times = pd.date_range("2025-01-01 14:30:00+00:00", periods=5, freq="1min")
    df = pd.DataFrame(
        {
            "ts_event": times,
            "symbol": ["MESH5"] * 5,
            "open": [100, 101, 102, 103, 104],
            "high": [100, 101, 102, 103, 104],
            "low": [99, 100, 103, 104, 105],
            "close": [99.5, 100.5, 103.5, 104.5, 105.5],
            "volume": [1, 1, 1, 1, 1],
        }
    )
    out, events = add_fvg_features(df, timeframes=("1min",))
    assert events[events["direction"] == "bullish"].shape[0] >= 1
    assert pd.isna(out.loc[1, "nearest_bullish_fvg_below_1m"])
    assert out.loc[2, "active_bullish_fvg_count_1m"] >= 1

