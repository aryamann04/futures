from __future__ import annotations

import pandas as pd

from backtest.engine import run_backtest


def test_engine_uses_next_bar_open_and_can_flatten_eod() -> None:
    df = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2025-01-01 14:30:00+00:00",
                    "2025-01-01 14:31:00+00:00",
                    "2025-01-01 20:59:00+00:00",
                    "2025-01-01 21:00:00+00:00",
                ]
            ),
            "symbol": ["MESH5"] * 4,
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [100.5, 101.5, 102.5, 103.5],
            "low": [99.5, 100.5, 101.5, 102.5],
            "close": [100.0, 101.2, 102.1, 103.0],
            "entry_signal": [1, 0, 0, 0],
            "exit_signal": [0, 0, 0, 0],
            "stop_loss": [99.0, None, None, None],
            "take_profit": [110.0, None, None, None],
            "max_hold_bars": [None, None, None, None],
            "max_hold_seconds": [None, None, None, None],
            "flatten_eod": [1, 1, 1, 1],
            "setup": ["test", "", "", ""],
            "session_name": ["ny", "", "", ""],
        }
    )
    _, trades = run_backtest(df, commission_per_side=0.0)
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert float(trade["entry_price"]) == 101.0
    assert trade["reason"] == "end_of_day"
    assert trade["setup"] == "test"
