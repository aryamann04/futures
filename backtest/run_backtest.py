from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

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
    feature_family: str = "regular"
    group: str = "core"
    print_head: bool = True


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    configs = {
        "micro_currency_futures": {
            "dataset": "micro_currency_futures",
            "symbols_prefix": "M6E",
            "start": "2025-01-01",
            "end": "2026-03-01",
            "bar_seconds": 1,
        },
        "micro_sp_futures": {
            "dataset": "micro_sp_futures",
            "symbols_prefix": "MES",
            "start": "2022-03-27",
            "end": "2026-03-01",
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
            add_session_levels=True,
            add_opening_ranges=True,
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
            add_opening_ranges=True,
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


def build_strategy_library(dataset_name: str) -> List[BacktestSpec]:
    opening_gate = bt.EntryGate(
        trade_start="09:30",
        trade_end="11:30",
        cooldown_bars=180 if dataset_name == "micro_currency_futures" else 3,
        max_trades_per_day=2,
    )
    balanced_gate = bt.EntryGate(
        trade_start=None,
        trade_end=None,
        cooldown_bars=300 if dataset_name == "micro_currency_futures" else 5,
        max_trades_per_day=2,
    )
    one_trade_open_gate = bt.EntryGate(
        trade_start="09:30",
        trade_end="11:30",
        cooldown_bars=180 if dataset_name == "micro_currency_futures" else 8,
        max_trades_per_day=1,
    )

    macd_tp = bt.TradeParams(
        stop_loss_pct=0.0025,
        take_profit_pct=0.0048,
        max_hold_seconds=5400.0,
        size=1.0,
    )
    meanrev_tp = bt.TradeParams(
        stop_loss_pct=0.0018,
        take_profit_pct=0.0038,
        max_hold_seconds=3600.0,
        size=1.0,
    )
    breakout_tp = bt.TradeParams(
        stop_loss_pct=0.0022,
        take_profit_pct=0.0055,
        max_hold_seconds=5400.0,
        size=1.0,
    )
    momentum_tp = bt.TradeParams(
        stop_loss_pct=0.0020,
        take_profit_pct=0.0045,
        max_hold_seconds=3600.0,
        size=1.0,
    )
    high_conviction_tp = bt.TradeParams(
        stop_loss_pct=0.0032,
        take_profit_pct=0.0058,
        max_hold_seconds=14400.0,
        size=1.0,
    )

    return [
        BacktestSpec(
            name="ema600_adx50_high_conviction",
            strategy_fn=bt.ema600_adx50_high_conviction,
            group="baseline",
            strategy_kwargs={
                "trade_params": high_conviction_tp,
                "gate": balanced_gate,
            },
        ),
        BacktestSpec(
            name="macd_ema_rsi_confluence_opening_window",
            strategy_fn=bt.confluence_strategy,
            group="baseline",
            strategy_kwargs={
                "long_conditions": [
                    bt.col_eq("macd_cross_up", 1),
                    bt.col_eq("ema100_gt_ema300", 1),
                    bt.col_lte("rsi_50", 55.0),
                    bt.abs_col_gte("macd_hist_atr_norm", 0.08),
                ],
                "short_conditions": [
                    bt.col_eq("macd_cross_down", 1),
                    bt.col_eq("ema100_lt_ema300", 1),
                    bt.col_gte("rsi_50", 45.0),
                    bt.abs_col_gte("macd_hist_atr_norm", 0.08),
                ],
                "trade_params": macd_tp,
                "gate": opening_gate,
            },
        ),
        BacktestSpec(
            name="opening_range_breakout_15m",
            strategy_fn=bt.opening_range_breakout,
            group="breakout",
            strategy_kwargs={
                "opening_range_minutes": 15,
                "breakout_buffer_pct": 0.00025,
                "confirm_trend": True,
                "confirm_vwap": True,
                "confirm_adx_min": 18.0,
                "trade_params": breakout_tp,
                "gate": one_trade_open_gate,
            },
        ),
        BacktestSpec(
            name="opening_range_breakout_30m",
            strategy_fn=bt.opening_range_breakout,
            group="breakout",
            strategy_kwargs={
                "opening_range_minutes": 30,
                "breakout_buffer_pct": 0.00025,
                "confirm_trend": True,
                "confirm_vwap": True,
                "confirm_adx_min": 18.0,
                "trade_params": breakout_tp,
                "gate": one_trade_open_gate,
            },
        ),
        BacktestSpec(
            name="opening_range_breakout_retest_30m",
            strategy_fn=bt.opening_range_breakout_retest,
            group="breakout",
            strategy_kwargs={
                "opening_range_minutes": 30,
                "retest_tolerance_pct": 0.0008,
                "confirm_trend": True,
                "confirm_adx_min": 18.0,
                "trade_params": breakout_tp,
                "gate": one_trade_open_gate,
            },
        ),
        BacktestSpec(
            name="donchian_breakout_adx_opening_window",
            strategy_fn=bt.donchian_breakout_adx,
            group="momentum_breakout",
            strategy_kwargs={
                "lookback_bars": 60,
                "adx_min": 20.0,
                "rel_volume_min": 1.05,
                "trade_params": breakout_tp,
                "gate": one_trade_open_gate,
            },
        ),
        BacktestSpec(
            name="ema_slope_momentum_pullback_opening_window",
            strategy_fn=bt.ema_slope_momentum_pullback,
            group="momentum_pullback",
            strategy_kwargs={
                "pullback_to_ema": "ema20",
                "trend_fast_col": "ema50",
                "trend_slow_col": "ema200",
                "slope_ema_col": "ema50",
                "slope_lookback": 10,
                "slope_min_pct": 0.00035,
                "rsi_long_min": 52.0,
                "rsi_short_max": 48.0,
                "adx_min": 16.0,
                "pullback_tolerance_pct": 0.0009,
                "trade_params": momentum_tp,
                "gate": opening_gate,
            },
        ),
        BacktestSpec(
            name="vwap_trend_pullback_opening_window",
            strategy_fn=bt.vwap_trend_pullback,
            group="momentum_pullback",
            strategy_kwargs={
                "trend_fast_col": "ema50",
                "trend_slow_col": "ema200",
                "rsi_long_min": 52.0,
                "rsi_short_max": 48.0,
                "adx_min": 16.0,
                "vwap_tolerance_pct": 0.0009,
                "trade_params": momentum_tp,
                "gate": opening_gate,
            },
        ),
        BacktestSpec(
            name="macd_trend_opening_window",
            strategy_fn=bt.macd_signal_cross_trend,
            group="legacy_compare",
            strategy_kwargs={
                "macd_cross_up_col": "macd_cross_up",
                "macd_cross_down_col": "macd_cross_down",
                "trend_up_col": "ema100_gt_ema300",
                "trend_down_col": "ema100_lt_ema300",
                "stop_loss_pct": 0.0025,
                "take_profit_pct": 0.0048,
                "max_hold_seconds": 5400.0,
                "trade_start": opening_gate.trade_start,
                "trade_end": opening_gate.trade_end,
                "cooldown_bars": opening_gate.cooldown_bars,
                "max_trades_per_day": opening_gate.max_trades_per_day,
            },
        ),
        BacktestSpec(
            name="ema_rsi_mean_reversion_opening_window",
            strategy_fn=bt.confluence_strategy,
            group="legacy_compare",
            strategy_kwargs={
                "long_conditions": [
                    bt.crossed_below("price_vs_ema80", -0.0018),
                    bt.crossed_below("rsi_14", 32.0),
                    bt.col_lte("adx_14", 22.0),
                ],
                "short_conditions": [
                    bt.crossed_above("price_vs_ema80", 0.0018),
                    bt.crossed_above("rsi_14", 68.0),
                    bt.col_lte("adx_14", 22.0),
                ],
                "trade_params": meanrev_tp,
                "gate": opening_gate,
            },
        ),
        BacktestSpec(
            name="fib_rsi_trend_opening_window",
            strategy_fn=bt.fib_trend_retracement_rsi,
            feature_family="fib",
            group="legacy_compare",
            strategy_kwargs={
                "fib_prefix": "fib_4h",
                "trend_up_col": "ema100_gt_ema300",
                "trend_down_col": "ema100_lt_ema300",
                "range_col": "trend_range_4h",
                "range_min": 0.0045,
                "rsi_col": "rsi_50",
                "long_rsi_max": 55.0,
                "short_rsi_min": 45.0,
                "stop_loss_pct": 0.0030,
                "take_profit_pct": 0.0060,
                "max_hold_seconds": 21600.0,
                "trade_start": opening_gate.trade_start,
                "trade_end": opening_gate.trade_end,
                "cooldown_bars": opening_gate.cooldown_bars,
                "max_trades_per_day": opening_gate.max_trades_per_day,
            },
        ),
    ]


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
        results.append(run_single_backtest(feat, spec, initial_capital=initial_capital))
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

    experiments = build_strategy_library(dataset_name)

    # examples:
    # experiments = filter_experiments(experiments, group="breakout")
    # experiments = filter_experiments(experiments, group="momentum_pullback")
    # experiments = filter_experiments(experiments, names=["opening_range_breakout_30m"])

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