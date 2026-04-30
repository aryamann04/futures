from __future__ import annotations

import pandas as pd

from backtest.metrics import compute_extended_metrics
from backtest.strategy_runner import (
    StrategyRunSpec,
    prepare_research_data,
    run_multi_strategy_research,
    run_multi_walk_forward_research,
)
from strategies.baselines import random_time_entry, vwap_reclaim_only


def _small_raw() -> pd.DataFrame:
    ts = pd.date_range("2025-01-02 14:30:00+00:00", periods=120, freq="1min")
    close = pd.Series(range(120), dtype="float64") * 0.25 + 5000.0
    return pd.DataFrame(
        {
            "ts_event": ts,
            "symbol": ["MESH5"] * len(ts),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close + 0.1,
            "volume": 100.0,
        }
    )


def test_compute_extended_metrics_is_silent_by_default(capsys) -> None:
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2025-01-01 14:31:00+00:00"]),
            "exit_time": pd.to_datetime(["2025-01-01 14:35:00+00:00"]),
            "pnl_dollars": [10.0],
            "pnl_r": [1.0],
        }
    )
    metrics = compute_extended_metrics(trades)
    captured = capsys.readouterr()
    assert metrics["total_trades"] == 1
    assert captured.out == ""


def test_multi_strategy_runner_returns_summary() -> None:
    raw = _small_raw()
    specs = [
        StrategyRunSpec(name="random", strategy_fn=random_time_entry, strategy_kwargs={"probability": 0.05, "seed": 1}),
        StrategyRunSpec(name="vwap", strategy_fn=vwap_reclaim_only),
    ]
    result = run_multi_strategy_research(raw, specs, output_dir=None, max_workers=2)
    assert len(result["results"]) == 2
    assert set(result["summary"]["name"]) == {"random", "vwap"}


def test_prepare_research_data_resamples_global_base_timeframe() -> None:
    raw = _small_raw()
    prepared = prepare_research_data(raw, base_timeframe="15m")
    assert len(prepared.features) < len(raw)
    diffs = prepared.features["ts_event"].sort_values().diff().dropna()
    assert not diffs.empty
    assert diffs.min() >= pd.Timedelta(minutes=15)


def test_multi_walk_forward_runner_returns_summary() -> None:
    raw = _small_raw()
    specs = [
        StrategyRunSpec(name="random", strategy_fn=random_time_entry, strategy_kwargs={"probability": 0.05, "seed": 1}),
        StrategyRunSpec(name="vwap", strategy_fn=vwap_reclaim_only),
    ]
    result = run_multi_walk_forward_research(
        raw,
        specs,
        train_period="30min",
        test_period="30min",
        max_workers=2,
    )
    assert len(result["results"]) == 2
    assert set(result["summary"]["name"]) == {"random", "vwap"}
