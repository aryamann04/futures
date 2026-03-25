from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional

import pandas as pd

from data.load import micro_futures_data
from features.build_features import build_features
from features.research import _filter_raw_data
from backtest.strategies import (ema_mean_reversion, 
                                 breakout_momentum, 
                                 rsi_reversal, 
                                 ema_adx_mean_reversion, 
                                 ema_atr_mean_reversion, 
                                 ema_bb_mean_reversion,
                                 ema_adx_session_mean_reversion,
                                 )

from backtest.engine import run_backtest
from backtest.metrics import compute_basic_metrics, plot_equity_curve


StrategyFn = Callable[..., pd.DataFrame]


@dataclass
class BacktestSpec:
    name: str
    strategy_fn: StrategyFn
    strategy_kwargs: Dict[str, Any]
    plot: bool = False


def load_and_prepare_data(
    symbols: Optional[list[str]] = None,
    symbols_prefix: Optional[str] = None,
    include_spreads: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    raw = micro_futures_data(columns=["ts_event", "symbol", "open", "high", "low", "close", "volume"])

    raw = _filter_raw_data(
        raw,
        symbols=symbols,
        symbols_prefix=symbols_prefix,
        include_spreads=include_spreads,
        start=start,
        end=end,
    )

    print(f"Filtered rows: {len(raw):,}")
    if not raw.empty:
        print(f"Symbols: {sorted(raw['symbol'].astype(str).unique().tolist())[:20]}")
        print(f"Time range: {raw['ts_event'].min()} -> {raw['ts_event'].max()}")

    feat = build_features(raw)
    return feat


def run_single_backtest(
    feat: pd.DataFrame,
    spec: BacktestSpec,
) -> Dict[str, Any]:
    print("\n" + "=" * 100)
    print(f"RUNNING: {spec.name}")
    print("=" * 100)

    plan = spec.strategy_fn(feat.copy(), **spec.strategy_kwargs)
    bars, trades = run_backtest(plan)

    print(trades.head())
    print(f"Trades: {len(trades):,}")

    metrics = compute_basic_metrics(trades)

    if spec.plot:
        plot_equity_curve(trades)

    return {
        "name": spec.name,
        "bars": bars,
        "trades": trades,
        "metrics": metrics,
    }


def summarize_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for result in results:
        metrics = result["metrics"] or {}
        trades = result["trades"]

        rows.append(
            {
                "name": result["name"],
                "n_trades": len(trades),
                "total_return": metrics.get("total_return"),
                "cagr": metrics.get("cagr"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "win_rate": metrics.get("win_rate"),
                "avg_return": metrics.get("avg_return"),
                "median_return": metrics.get("median_return"),
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("sharpe", ascending=False, na_position="last").reset_index(drop=True)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    if summary.empty:
        print("No results.")
    else:
        print(summary.to_string(index=False))

    return summary


def build_experiments() -> List[BacktestSpec]:
    return [
        
        BacktestSpec(
            name="ema100_adx_session_london_ny",
            strategy_fn=ema_adx_session_mean_reversion,
            strategy_kwargs={
                "z_col": "price_vs_ema100",
                "adx_col": "adx_14",
                "session_col": "is_london_ny",
                "long_threshold": -0.0018,
                "short_threshold": 0.0018,
                "adx_max": 20.0,
                "stop_loss_pct": 0.0018,
                "take_profit_pct": 0.0058,
                "max_hold_bars": 1500,
                "size": 1.0,
            },
        ),
        BacktestSpec(
            name="ema100_adx_session_us_morning",
            strategy_fn=ema_adx_session_mean_reversion,
            strategy_kwargs={
                "z_col": "price_vs_ema100",
                "adx_col": "adx_14",
                "session_col": "is_us_morning",
                "long_threshold": -0.0018,
                "short_threshold": 0.0018,
                "adx_max": 20.0,
                "stop_loss_pct": 0.0018,
                "take_profit_pct": 0.0058,
                "max_hold_bars": 1500,
                "size": 1.0,
            },
        ),
        BacktestSpec(
            name="ema90_shorter_hold",
            strategy_fn=ema_mean_reversion,
            strategy_kwargs={
                "z_col": "price_vs_ema90",
                "long_threshold": -0.0016,
                "short_threshold": 0.0016,
                "stop_loss_pct": 0.0020,
                "take_profit_pct": 0.0058,
                "max_hold_bars": 1000,
                "size": 1.0,
            },
        ),
        BacktestSpec(
            name="ema100_shorter_hold",
            strategy_fn=ema_mean_reversion,
            strategy_kwargs={
                "z_col": "price_vs_ema100",
                "long_threshold": -0.0018,
                "short_threshold": 0.0018,
                "stop_loss_pct": 0.0020,
                "take_profit_pct": 0.0058,
                "max_hold_bars": 1000,
                "size": 1.0,
            },
        ),
    ]

def main():
    feat = load_and_prepare_data(
        symbols_prefix="M6E",
        include_spreads=False,
        start="2025-01-01",
        end="2026-03-01",
    )

    experiments = build_experiments()

    results = []
    for spec in experiments:
        result = run_single_backtest(feat, spec)
        results.append(result)

    summary = summarize_results(results)
    return results, summary


if __name__ == "__main__":
    main()