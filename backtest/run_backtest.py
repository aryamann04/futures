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
from backtest.validation import (
    ValidationSpec,
    run_out_of_sample_test,
    compare_specs_walk_forward,
)


StrategyFn = Callable[..., pd.DataFrame]


@dataclass
class BacktestSpec:
    name: str
    strategy_fn: StrategyFn
    strategy_kwargs: Dict[str, Any]
    feature_family: str
    plot: bool = False


def load_raw_data(
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

    return raw


def build_features_for_family(raw: pd.DataFrame, family: str) -> pd.DataFrame:
    if family == "ema":
        return build_features(
            raw,
            add_basic_returns=False,
            add_trend=True,
            add_momentum=False,
            add_volatility=True,
            add_volume=False,
            add_session_levels=False,
            add_opening_ranges=False,
            add_rolling_ranges=False,
            add_fvg=False,
            shift_features=True,
        )

    if family == "levels":
        return build_features(
            raw,
            add_basic_returns=False,
            add_trend=False,
            add_momentum=False,
            add_volatility=False,
            add_volume=True,
            add_session_levels=True,
            add_opening_ranges=True,
            add_rolling_ranges=True,
            add_fvg=False,
            shift_features=True,
        )

    raise ValueError(f"Unknown feature family: {family}")


def prepare_feature_sets(raw: pd.DataFrame, experiments: List[BacktestSpec]) -> Dict[str, pd.DataFrame]:
    families = sorted({spec.feature_family for spec in experiments})
    feat_map: Dict[str, pd.DataFrame] = {}

    for family in families:
        print("\n" + "=" * 100)
        print(f"BUILDING FEATURES: {family}")
        print("=" * 100)
        feat_map[family] = build_features_for_family(raw, family)

    return feat_map


def run_single_backtest(
    feat: pd.DataFrame,
    spec: BacktestSpec,
    initial_capital: float = 1000.0,
) -> Dict[str, Any]:
    print("\n" + "=" * 100)
    print(f"RUNNING: {spec.name}")
    print("=" * 100)

    plan = spec.strategy_fn(feat, **spec.strategy_kwargs)
    bars, trades = run_backtest(plan)

    print(trades.head())
    print(f"Trades: {len(trades):,}")

    metrics = compute_basic_metrics(trades, initial_capital=initial_capital)

    if spec.plot:
        plot_equity_curve(trades, initial_capital=initial_capital)

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
                "avg_trade_pnl": metrics.get("avg_trade_pnl"),
                "median_trade_pnl": metrics.get("median_trade_pnl"),
                "avg_return": metrics.get("avg_return"),
                "median_return": metrics.get("median_return"),
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["sharpe", "total_return"], ascending=[False, False], na_position="last").reset_index(drop=True)

    print("\n" + "=" * 100)
    print("FULL-SAMPLE SUMMARY")
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
            feature_family="ema",
            strategy_kwargs={
                "z_col": "price_vs_ema90",
                "adx_col": "adx_14",
                "long_threshold": -0.0015,
                "short_threshold": 0.0015,
                "adx_max": 28.0,
                "stop_loss_pct": 0.0020,
                "take_profit_pct": 0.0065,
                "max_hold_seconds": 600,
                "size": 1.0,
            },
            plot=True,
        ),
        BacktestSpec(
            name="ema100_low_adx_retest",
            strategy_fn=bt.ema_adx_mean_reversion,
            feature_family="ema",
            strategy_kwargs={
                "z_col": "price_vs_ema100",
                "adx_col": "adx_14",
                "long_threshold": -0.0018,
                "short_threshold": 0.0018,
                "adx_max": 20.0,
                "stop_loss_pct": 0.0018,
                "take_profit_pct": 0.0058,
                "max_hold_seconds": 900,
                "size": 1.0,
            },
        ),
        BacktestSpec(
            name="ema95_low_adx",
            strategy_fn=bt.ema_adx_mean_reversion,
            feature_family="ema",
            strategy_kwargs={
                "z_col": "price_vs_ema95",
                "adx_col": "adx_14",
                "long_threshold": -0.0017,
                "short_threshold": 0.0017,
                "adx_max": 20.0,
                "stop_loss_pct": 0.0018,
                "take_profit_pct": 0.0058,
                "max_hold_seconds": 900,
                "size": 1.0,
            },
        ),
        BacktestSpec(
            name="ema105_low_adx",
            strategy_fn=bt.ema_adx_mean_reversion,
            feature_family="ema",
            strategy_kwargs={
                "z_col": "price_vs_ema105",
                "adx_col": "adx_14",
                "long_threshold": -0.0018,
                "short_threshold": 0.0018,
                "adx_max": 20.0,
                "stop_loss_pct": 0.0018,
                "take_profit_pct": 0.0058,
                "max_hold_seconds": 900,
                "size": 1.0,
            },
        ),
        BacktestSpec(
            name="ema90_wider_tp",
            strategy_fn=bt.ema_mean_reversion,
            feature_family="ema",
            strategy_kwargs={
                "z_col": "price_vs_ema90",
                "long_threshold": -0.0016,
                "short_threshold": 0.0016,
                "stop_loss_pct": 0.0020,
                "take_profit_pct": 0.0065,
                "max_hold_seconds": 900,
                "size": 1.0,
            },
        ),
        BacktestSpec(
            name="ema90_short_hold_strict",
            strategy_fn=bt.ema_mean_reversion,
            feature_family="ema",
            strategy_kwargs={
                "z_col": "price_vs_ema90",
                "long_threshold": -0.0016,
                "short_threshold": 0.0016,
                "stop_loss_pct": 0.0020,
                "take_profit_pct": 0.0060,
                "max_hold_seconds": 600,
                "size": 1.0,
            },
        ),
        BacktestSpec(
            name="prior_session_breakout",
            strategy_fn=bt.prior_session_breakout,
            feature_family="levels",
            strategy_kwargs={
                "high_col": "prev_session_high",
                "low_col": "prev_session_low",
                "breakout_buffer": 0.0003,
                "volume_col": "rel_volume_20",
                "volume_min": 1.2,
                "stop_loss_pct": 0.0018,
                "take_profit_pct": 0.0055,
                "max_hold_seconds": 1200,
                "size": 1.0,
            },
        ),
        BacktestSpec(
            name="prior_session_failed_breakout",
            strategy_fn=bt.prior_session_failed_breakout,
            feature_family="levels",
            strategy_kwargs={
                "high_col": "prev_session_high",
                "low_col": "prev_session_low",
                "sweep_buffer": 0.0002,
                "volume_col": "rel_volume_20",
                "volume_min": 1.0,
                "stop_loss_pct": 0.0015,
                "take_profit_pct": 0.0045,
                "max_hold_seconds": 900,
                "size": 1.0,
            },
        ),
        BacktestSpec(
            name="opening_range_breakout_5m",
            strategy_fn=bt.opening_range_breakout,
            feature_family="levels",
            strategy_kwargs={
                "high_col": "opening_range_high_5m",
                "low_col": "opening_range_low_5m",
                "breakout_buffer": 0.0002,
                "volume_col": "rel_volume_20",
                "volume_min": 1.2,
                "stop_loss_pct": 0.0018,
                "take_profit_pct": 0.0050,
                "max_hold_seconds": 1200,
                "size": 1.0,
            },
        ),
        BacktestSpec(
            name="opening_range_breakout_15m",
            strategy_fn=bt.opening_range_breakout,
            feature_family="levels",
            strategy_kwargs={
                "high_col": "opening_range_high_15m",
                "low_col": "opening_range_low_15m",
                "breakout_buffer": 0.0002,
                "volume_col": "rel_volume_20",
                "volume_min": 1.2,
                "stop_loss_pct": 0.0018,
                "take_profit_pct": 0.0050,
                "max_hold_seconds": 1200,
                "size": 1.0,
            },
        ),
        BacktestSpec(
            name="rolling_range_fade_30m",
            strategy_fn=bt.rolling_range_fade,
            feature_family="levels",
            strategy_kwargs={
                "high_col": "rolling_high_30m",
                "low_col": "rolling_low_30m",
                "sweep_buffer": 0.0002,
                "volume_col": "rel_volume_20",
                "volume_max": 1.5,
                "stop_loss_pct": 0.0015,
                "take_profit_pct": 0.0045,
                "max_hold_seconds": 900,
                "size": 1.0,
            },
        ),
        BacktestSpec(
            name="rolling_range_fade_60m",
            strategy_fn=bt.rolling_range_fade,
            feature_family="levels",
            strategy_kwargs={
                "high_col": "rolling_high_60m",
                "low_col": "rolling_low_60m",
                "sweep_buffer": 0.0002,
                "volume_col": "rel_volume_20",
                "volume_max": 1.5,
                "stop_loss_pct": 0.0015,
                "take_profit_pct": 0.0045,
                "max_hold_seconds": 900,
                "size": 1.0,
            },
        ),
    ]


def to_validation_specs(experiments: List[BacktestSpec]) -> List[ValidationSpec]:
    return [
        ValidationSpec(
            name=spec.name,
            strategy_fn=spec.strategy_fn,
            strategy_kwargs=spec.strategy_kwargs,
        )
        for spec in experiments
    ]


def run_full_sample_suite(
    feat_map: Dict[str, pd.DataFrame],
    experiments: List[BacktestSpec],
    initial_capital: float = 1000.0,
):
    results = []
    for spec in experiments:
        feat = feat_map[spec.feature_family]
        result = run_single_backtest(feat, spec, initial_capital=initial_capital)
        results.append(result)

    summary = summarize_results(results)
    return results, summary


def run_oos_suite(
    feat_map: Dict[str, pd.DataFrame],
    experiments: List[BacktestSpec],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    initial_capital: float = 1000.0,
) -> pd.DataFrame:
    rows = []

    print("\n" + "=" * 100)
    print("OUT-OF-SAMPLE TESTS")
    print("=" * 100)

    for spec in experiments:
        feat = feat_map[spec.feature_family]
        val_spec = ValidationSpec(
            name=spec.name,
            strategy_fn=spec.strategy_fn,
            strategy_kwargs=spec.strategy_kwargs,
        )

        out = run_out_of_sample_test(
            df=feat,
            spec=val_spec,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            initial_capital=initial_capital,
            keep_train_outputs=False,
            keep_test_outputs=False,
        )

        summary = out["summary"]
        train_row = summary.loc[summary["segment"] == "train"].iloc[0]
        test_row = summary.loc[summary["segment"] == "test"].iloc[0]

        rows.append(
            {
                "name": spec.name,
                "feature_family": spec.feature_family,
                "train_trades": train_row["n_trades"],
                "train_return": train_row["total_return"],
                "train_sharpe": train_row["sharpe"],
                "test_trades": test_row["n_trades"],
                "test_return": test_row["total_return"],
                "test_sharpe": test_row["sharpe"],
                "test_max_drawdown": test_row["max_drawdown"],
                "test_win_rate": test_row["win_rate"],
            }
        )

    oos_summary = pd.DataFrame(rows)
    if not oos_summary.empty:
        oos_summary = oos_summary.sort_values(["test_sharpe", "test_return"], ascending=[False, False]).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("OUT-OF-SAMPLE SUMMARY")
    print("=" * 100)
    if oos_summary.empty:
        print("No OOS results.")
    else:
        print(oos_summary.to_string(index=False))

    return oos_summary


def run_walk_forward_suite(
    feat_map: Dict[str, pd.DataFrame],
    experiments: List[BacktestSpec],
    train_period: str = "120D",
    test_period: str = "30D",
    step_period: str = "30D",
    start: str = "2025-01-01",
    end: str = "2026-03-01",
    initial_capital: float = 1000.0,
) -> pd.DataFrame:
    print("\n" + "=" * 100)
    print("WALK-FORWARD VALIDATION")
    print("=" * 100)

    rows = []
    for spec in experiments:
        feat = feat_map[spec.feature_family]
        val_spec = ValidationSpec(
            name=spec.name,
            strategy_fn=spec.strategy_fn,
            strategy_kwargs=spec.strategy_kwargs,
        )

        wf = compare_specs_walk_forward(
            df=feat,
            specs=[val_spec],
            train_period=train_period,
            test_period=test_period,
            step_period=step_period,
            start=start,
            end=end,
            initial_capital=initial_capital,
        )

        if not wf.empty:
            row = wf.iloc[0].to_dict()
            row["feature_family"] = spec.feature_family
            rows.append(row)

    wf_summary = pd.DataFrame(rows)
    if not wf_summary.empty:
        wf_summary = wf_summary.sort_values(["mean_sharpe", "mean_total_return"], ascending=[False, False]).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("WALK-FORWARD SUMMARY")
    print("=" * 100)
    if wf_summary.empty:
        print("No WF results.")
    else:
        print(wf_summary.to_string(index=False))

    return wf_summary


def main():
    initial_capital = 1000.0

    raw = load_raw_data(
        symbols_prefix="M6E",
        include_spreads=False,
        start="2025-01-01",
        end="2026-03-01",
    )

    experiments = build_experiments()
    feat_map = prepare_feature_sets(raw, experiments)

    full_results, full_summary = run_full_sample_suite(
        feat_map=feat_map,
        experiments=experiments,
        initial_capital=initial_capital,
    )

    oos_summary = run_oos_suite(
        feat_map=feat_map,
        experiments=experiments,
        train_start="2025-01-01",
        train_end="2025-10-01",
        test_start="2025-10-01",
        test_end="2026-03-01",
        initial_capital=initial_capital,
    )

    wf_summary = run_walk_forward_suite(
        feat_map=feat_map,
        experiments=experiments,
        train_period="120D",
        test_period="30D",
        step_period="30D",
        start="2025-01-01",
        end="2026-03-01",
        initial_capital=initial_capital,
    )

    return {
        "full_sample_summary": full_summary,
        "out_of_sample_summary": oos_summary,
        "walk_forward_summary": wf_summary,
        "full_sample_results": full_results,
    }


if __name__ == "__main__":
    main()