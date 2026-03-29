from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional

import gc
import pandas as pd

from data.load import micro_futures_data
from features.build_features import build_features
from features.research import _filter_raw_data
import backtest.strategies as bt

from backtest.engine import run_backtest
from backtest.metrics import compute_basic_metrics


StrategyFn = Callable[..., pd.DataFrame]


@dataclass
class BacktestSpec:
    name: str
    strategy_fn: StrategyFn
    strategy_kwargs: Dict[str, Any]
    feature_family: str
    group: str
    print_head: bool = True


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    configs = {
        "micro_currency_futures": {
            "dataset": "micro_currency_futures",
            "symbols_prefix": "M6E",
            "start": "2025-01-01",
            "end": "2026-03-01",
            "default_group": "regular",
            "bar_seconds": 1,
        },
        "micro_sp_futures": {
            "dataset": "micro_sp_futures",
            "symbols_prefix": "MES",
            "start": "2022-03-27",
            "end": "2026-03-01",
            "default_group": "all",
            "bar_seconds": 60,
        },
    }

    if dataset_name not in configs:
        valid = ", ".join(configs.keys())
        raise ValueError(f"Unknown dataset_name '{dataset_name}'. Valid options: {valid}")

    return configs[dataset_name]


def load_raw_data(
    dataset: str,
    symbols: Optional[list[str]] = None,
    symbols_prefix: Optional[str] = None,
    include_spreads: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    raw = micro_futures_data(
        dataset=dataset,
        columns=["ts_event", "symbol", "open", "high", "low", "close", "volume"],
    )

    raw = _filter_raw_data(
        raw,
        symbols=symbols,
        symbols_prefix=symbols_prefix,
        include_spreads=include_spreads,
        start=start,
        end=end,
    )

    print(f"Dataset: {dataset}")
    print(f"Filtered rows: {len(raw):,}")
    if not raw.empty:
        print(f"Symbols: {sorted(raw['symbol'].astype(str).unique().tolist())[:20]}")
        print(f"Time range: {raw['ts_event'].min()} -> {raw['ts_event'].max()}")

    return raw


def build_features_for_family(raw: pd.DataFrame, family: str, bar_seconds: int) -> pd.DataFrame:
    if family == "regular":
        return build_features(
            raw,
            add_basic_returns=True,
            add_trend=True,
            add_momentum=True,
            add_volatility=True,
            add_volume=True,
            add_session_levels=False,
            add_opening_ranges=False,
            add_rolling_ranges=True,
            add_fvg=False,
            shift_features=True,
            bar_seconds=bar_seconds,
        )

    if family == "fib":
        return build_features(
            raw,
            add_basic_returns=True,
            add_trend=True,
            add_momentum=True,
            add_volatility=True,
            add_volume=True,
            add_session_levels=True,
            add_opening_ranges=False,
            add_rolling_ranges=True,
            add_fvg=False,
            shift_features=True,
            bar_seconds=bar_seconds,
        )

    raise ValueError(f"Unknown feature family: {family}")


def prepare_feature_sets(raw: pd.DataFrame, experiments: List[BacktestSpec], bar_seconds: int) -> Dict[str, pd.DataFrame]:
    families = sorted({spec.feature_family for spec in experiments})
    feat_map: Dict[str, pd.DataFrame] = {}

    for family in families:
        print("\n" + "=" * 100)
        print(f"BUILDING FEATURES: {family}")
        print("=" * 100)
        feat_map[family] = build_features_for_family(raw, family, bar_seconds=bar_seconds)

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
    _, trades = run_backtest(plan)

    if spec.print_head:
        print(trades.head())
    print(f"Trades: {len(trades):,}")

    metrics = compute_basic_metrics(trades, initial_capital=initial_capital)

    out = {
        "name": spec.name,
        "group": spec.group,
        "feature_family": spec.feature_family,
        "n_trades": len(trades),
        "metrics": metrics,
    }

    del plan
    del trades
    gc.collect()

    return out


def summarize_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for result in results:
        metrics = result["metrics"] or {}

        rows.append(
            {
                "name": result["name"],
                "group": result["group"],
                "feature_family": result["feature_family"],
                "n_trades": result["n_trades"],
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
        summary = summary.sort_values(
            ["group", "sharpe", "total_return"],
            ascending=[True, False, False],
            na_position="last",
        ).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("FULL-SAMPLE SUMMARY")
    print("=" * 100)
    if summary.empty:
        print("No results.")
    else:
        print(summary.to_string(index=False))

    return summary


def build_regular_experiments(dataset_name: str) -> List[BacktestSpec]:
    if dataset_name == "micro_sp_futures":
        return [
            BacktestSpec(
                name="ema600_adx50_high_conviction",
                strategy_fn=bt.ema_adx_mean_reversion,
                feature_family="regular",
                group="regular",
                strategy_kwargs={
                    "z_col": "price_vs_ema600",
                    "adx_col": "adx_50",
                    "long_threshold": -0.0065,
                    "short_threshold": 0.0065,
                    "adx_max": 18.0,
                    "stop_loss_pct": 0.0032,
                    "take_profit_pct": 0.0058,
                    "max_hold_seconds": 14400.0,
                    "size": 1.0,
                },
            ),
            BacktestSpec(
                name="bb_long_higher_conviction",
                strategy_fn=bt.bollinger_exhaustion_reversal_long,
                feature_family="regular",
                group="regular",
                strategy_kwargs={
                    "bb_col": "bb_pos",
                    "adx_col": "adx_50",
                    "resistance_col": "rolling_high_4h",
                    "support_col": "rolling_low_4h",
                    "upper_threshold": 0.992,
                    "lower_threshold": 0.008,
                    "adx_max": 18.0,
                    "level_tolerance": 0.0010,
                    "stop_loss_pct": 0.0030,
                    "take_profit_pct": 0.0054,
                    "max_hold_seconds": 12600.0,
                    "size": 1.0,
                },
            ),
            BacktestSpec(
                name="macd_signal_cross_trend_core",
                strategy_fn=bt.macd_signal_cross_trend,
                feature_family="regular",
                group="macd",
                strategy_kwargs={
                    "macd_cross_up_col": "macd_cross_up",
                    "macd_cross_down_col": "macd_cross_down",
                    "trend_up_col": "ema100_gt_ema300",
                    "trend_down_col": "ema100_lt_ema300",
                    "stop_loss_pct": 0.0028,
                    "take_profit_pct": 0.0055,
                    "max_hold_seconds": 7200.0,
                    "exit_on_opposite_cross": True,
                    "size": 1.0,
                },
            ),
            BacktestSpec(
                name="macd_rsi_confirmation_core",
                strategy_fn=bt.macd_rsi_confirmation,
                feature_family="regular",
                group="macd",
                strategy_kwargs={
                    "macd_cross_up_col": "macd_cross_up",
                    "macd_cross_down_col": "macd_cross_down",
                    "rsi_col": "rsi_50",
                    "long_rsi_max": 52.0,
                    "short_rsi_min": 48.0,
                    "stop_loss_pct": 0.0025,
                    "take_profit_pct": 0.0048,
                    "max_hold_seconds": 5400.0,
                    "exit_on_opposite_cross": True,
                    "size": 1.0,
                },
            ),
            BacktestSpec(
                name="macd_hist_reversal_core",
                strategy_fn=bt.macd_hist_reversal,
                feature_family="regular",
                group="macd",
                strategy_kwargs={
                    "macd_hist_norm_col": "macd_hist_atr_norm",
                    "macd_hist_slope_col": "macd_hist_slope",
                    "long_threshold": -0.12,
                    "short_threshold": 0.12,
                    "stop_loss_pct": 0.0025,
                    "take_profit_pct": 0.0048,
                    "max_hold_seconds": 7200.0,
                    "exit_at_zero": True,
                    "size": 1.0,
                },
            ),
        ]

    return [
        BacktestSpec(
            name="ema600_adx50_high_conviction",
            strategy_fn=bt.ema_adx_mean_reversion,
            feature_family="regular",
            group="regular",
            strategy_kwargs={
                "z_col": "price_vs_ema600",
                "adx_col": "adx_50",
                "long_threshold": -0.0065,
                "short_threshold": 0.0065,
                "adx_max": 18.0,
                "stop_loss_pct": 0.0032,
                "take_profit_pct": 0.0058,
                "max_hold_seconds": 14400.0,
                "size": 1.0,
            },
        ),
    ]


def build_fib_experiments(dataset_name: str) -> List[BacktestSpec]:
    if dataset_name == "micro_sp_futures":
        return [
            BacktestSpec(
                name="fib_4h_rsi_active",
                strategy_fn=bt.fib_trend_retracement_rsi,
                feature_family="fib",
                group="fib",
                strategy_kwargs={
                    "trend_up_col": "ema100_gt_ema300",
                    "trend_down_col": "ema100_lt_ema300",
                    "fib_prefix": "fib_4h",
                    "range_col": "trend_range_4h",
                    "range_min": 0.0035,
                    "rsi_col": "rsi_50",
                    "long_rsi_max": 55.0,
                    "short_rsi_min": 45.0,
                    "stop_loss_pct": 0.0030,
                    "take_profit_pct": 0.0055,
                    "max_hold_seconds": 21600.0,
                    "exit_on_midpoint": True,
                    "size": 1.0,
                },
            ),
            BacktestSpec(
                name="fib_4h_rsi_balanced",
                strategy_fn=bt.fib_trend_retracement_rsi,
                feature_family="fib",
                group="fib",
                strategy_kwargs={
                    "trend_up_col": "ema100_gt_ema300",
                    "trend_down_col": "ema100_lt_ema300",
                    "fib_prefix": "fib_4h",
                    "range_col": "trend_range_4h",
                    "range_min": 0.0045,
                    "rsi_col": "rsi_50",
                    "long_rsi_max": 52.0,
                    "short_rsi_min": 48.0,
                    "stop_loss_pct": 0.0030,
                    "take_profit_pct": 0.0058,
                    "max_hold_seconds": 21600.0,
                    "exit_on_midpoint": True,
                    "size": 1.0,
                },
            ),
            BacktestSpec(
                name="fib_8h_structure_active",
                strategy_fn=bt.fib_trend_retracement_structure,
                feature_family="fib",
                group="fib",
                strategy_kwargs={
                    "trend_up_col": "ema100_gt_ema300",
                    "trend_down_col": "ema100_lt_ema300",
                    "fib_prefix": "fib_8h",
                    "range_col": "trend_range_8h",
                    "range_min": 0.0045,
                    "support_col": "prev_session_low",
                    "resistance_col": "prev_session_high",
                    "level_tolerance": 0.0025,
                    "stop_loss_pct": 0.0030,
                    "take_profit_pct": 0.0060,
                    "max_hold_seconds": 28800.0,
                    "exit_on_midpoint": True,
                    "size": 1.0,
                },
            ),
            BacktestSpec(
                name="fib_1d_day_active",
                strategy_fn=bt.fib_trend_retracement_day,
                feature_family="fib",
                group="fib",
                strategy_kwargs={
                    "trend_up_col": "ema300_gt_ema600",
                    "trend_down_col": "ema300_lt_ema600",
                    "fib_prefix": "fib_1d",
                    "range_col": "trend_range_1d",
                    "range_min": 0.0065,
                    "adx_col": "adx_50",
                    "adx_min": 18.0,
                    "stop_loss_pct": 0.0035,
                    "take_profit_pct": 0.0065,
                    "max_hold_seconds": 43200.0,
                    "exit_on_midpoint": True,
                    "size": 1.0,
                },
            ),
            BacktestSpec(
                name="macd_fib_4h_confirmation",
                strategy_fn=bt.macd_fib_retracement_confirmation,
                feature_family="fib",
                group="macd_fib",
                strategy_kwargs={
                    "macd_hist_slope_col": "macd_hist_slope",
                    "fib_zone_col": "fib_4h_in_fib_zone_500_618",
                    "trend_up_col": "ema100_gt_ema300",
                    "trend_down_col": "ema100_lt_ema300",
                    "stop_loss_pct": 0.0030,
                    "take_profit_pct": 0.0060,
                    "max_hold_seconds": 21600.0,
                    "exit_on_slope_flip": True,
                    "size": 1.0,
                },
            ),
            BacktestSpec(
                name="macd_fib_8h_confirmation",
                strategy_fn=bt.macd_fib_retracement_confirmation,
                feature_family="fib",
                group="macd_fib",
                strategy_kwargs={
                    "macd_hist_slope_col": "macd_hist_slope",
                    "fib_zone_col": "fib_8h_in_fib_zone_500_618",
                    "trend_up_col": "ema100_gt_ema300",
                    "trend_down_col": "ema100_lt_ema300",
                    "stop_loss_pct": 0.0032,
                    "take_profit_pct": 0.0062,
                    "max_hold_seconds": 28800.0,
                    "exit_on_slope_flip": True,
                    "size": 1.0,
                },
            ),
        ]

    return [
        BacktestSpec(
            name="fib_4h_rsi_active",
            strategy_fn=bt.fib_trend_retracement_rsi,
            feature_family="fib",
            group="fib",
            strategy_kwargs={
                "trend_up_col": "ema100_gt_ema300",
                "trend_down_col": "ema100_lt_ema300",
                "fib_prefix": "fib_4h",
                "range_col": "trend_range_4h",
                "range_min": 0.0060,
                "rsi_col": "rsi_50",
                "long_rsi_max": 48.0,
                "short_rsi_min": 52.0,
                "stop_loss_pct": 0.0032,
                "take_profit_pct": 0.0058,
                "max_hold_seconds": 21600.0,
                "exit_on_midpoint": True,
                "size": 1.0,
            },
        ),
    ]


def build_experiments(group: str, dataset_name: str) -> List[BacktestSpec]:
    regular = build_regular_experiments(dataset_name)
    fib = build_fib_experiments(dataset_name)

    if group == "regular":
        return regular
    if group == "fib":
        return fib
    if group == "all":
        return regular + fib

    raise ValueError(f"Unknown group: {group}")


def filter_experiments(
    experiments: List[BacktestSpec],
    feature_family: Optional[str] = None,
    group: Optional[str] = None,
    names: Optional[list[str]] = None,
) -> List[BacktestSpec]:
    out = experiments

    if feature_family is not None:
        out = [x for x in out if x.feature_family == feature_family]

    if group is not None:
        out = [x for x in out if x.group == group]

    if names is not None:
        wanted = set(names)
        out = [x for x in out if x.name in wanted]

    return out


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


def main():
    initial_capital = 1000.0

    # dataset_name = "micro_currency_futures"
    dataset_name = "micro_sp_futures"

    cfg = get_dataset_config(dataset_name)

    raw = load_raw_data(
        dataset=cfg["dataset"],
        symbols_prefix=cfg["symbols_prefix"],
        include_spreads=False,
        start=cfg["start"],
        end=cfg["end"],
    )

    group_to_run = "all"
    experiments = build_experiments(group=group_to_run, dataset_name=dataset_name)

    # examples:
    # experiments = filter_experiments(experiments, group="fib")
    # experiments = filter_experiments(experiments, names=["fib_4h_rsi_active", "macd_signal_cross_trend_core"])

    feat_map = prepare_feature_sets(raw, experiments, bar_seconds=cfg["bar_seconds"])

    _, full_summary = run_full_sample_suite(
        feat_map=feat_map,
        experiments=experiments,
        initial_capital=initial_capital,
    )

    return {
        "dataset": dataset_name,
        "summary": full_summary,
    }


if __name__ == "__main__":
    main()