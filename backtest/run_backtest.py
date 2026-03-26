from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional

import pandas as pd

from data.load import micro_futures_data
from features.build_features import build_features
from features.research import _filter_raw_data
import backtest.strategies as bt 

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
            name="ema90_soft_adx",
            strategy_fn=bt.ema_adx_mean_reversion,
            strategy_kwargs={
                "z_col": "price_vs_ema90",
                "adx_col": "adx_14",
                "long_threshold": -0.0015,
                "short_threshold": 0.0015,
                "adx_max": 28.0,
                "stop_loss_pct": 0.0020,
                "take_profit_pct": 0.0065,
                "max_hold_bars": 1000,
            },
        ),

        BacktestSpec(
            name="ema90_scaled_adx",
            strategy_fn=bt.ema_adx_scaled_mean_reversion,
            strategy_kwargs={
                "z_col": "price_vs_ema90",
                "adx_col": "adx_14",
                "long_threshold": -0.0015,
                "short_threshold": 0.0015,
                "adx_soft": 28.0,
                "adx_hard": 35.0,
                "stop_loss_pct": 0.0020,
                "take_profit_pct": 0.0065,
                "max_hold_bars": 1000,
            },
        ),

        BacktestSpec(
            name="ema90_high_asymmetry",
            strategy_fn=bt.ema_mean_reversion,
            strategy_kwargs={
                "z_col": "price_vs_ema90",
                "long_threshold": -0.0015,
                "short_threshold": 0.0015,
                "stop_loss_pct": 0.0020,
                "take_profit_pct": 0.0075,
                "max_hold_bars": 1000,
            },
        ),

        BacktestSpec(
            name="ema90_short_hold_strict",
            strategy_fn=bt.ema_mean_reversion,
            strategy_kwargs={
                "z_col": "price_vs_ema90",
                "long_threshold": -0.0016,
                "short_threshold": 0.0016,
                "stop_loss_pct": 0.0020,
                "take_profit_pct": 0.0060,
                "max_hold_bars": 800,
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