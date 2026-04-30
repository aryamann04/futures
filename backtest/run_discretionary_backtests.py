from __future__ import annotations

import argparse
from pathlib import Path

from backtest.metrics import format_metrics
from backtest.tui import launch_tui
from backtest.strategy_runner import (
    StrategyRunSpec,
    load_intraday_data,
    run_multi_strategy_research,
    run_multi_walk_forward_research,
    run_parameter_grid,
    run_strategy_research,
    run_walk_forward_research,
)
from features.resample import timeframe_to_timedelta
from strategies import (
    atr_breakout_only,
    confluence_continuation_strategy,
    fvg_pullback_strategy,
    naive_session_breakout,
    opening_range_breakout_strategy,
    random_time_entry,
    session_breakout_strategy,
    sweep_reclaim_strategy,
    vwap_reclaim_only,
)


def _effective_fvg_timeframe(base_timeframe: str, requested: str) -> str:
    if base_timeframe == "native":
        return requested
    return requested if timeframe_to_timedelta(requested) >= timeframe_to_timedelta(base_timeframe) else base_timeframe


STRATEGIES = {
    "session_breakout": session_breakout_strategy,
    "sweep_reclaim": sweep_reclaim_strategy,
    "fvg_pullback": fvg_pullback_strategy,
    "opening_range": opening_range_breakout_strategy,
    "confluence_continuation": confluence_continuation_strategy,
    "naive_session_breakout": naive_session_breakout,
    "vwap_reclaim_only": vwap_reclaim_only,
    "atr_breakout_only": atr_breakout_only,
    "random_time_entry": random_time_entry,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run discretionary-style futures backtests.")
    parser.add_argument("--dataset", default="micro_sp_futures")
    parser.add_argument("--strategy", nargs="+", required=True, choices=sorted(STRATEGIES.keys()))
    parser.add_argument("--symbols-prefix", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default="backtest/outputs/discretionary")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--base-timeframe", default="native", choices=["native", "1m", "5m", "15m", "1h"])
    parser.add_argument("--level-prefix", default="london")
    parser.add_argument("--opening-range-prefix", default="ny_open_range")
    parser.add_argument("--score-threshold", type=int, default=3)
    parser.add_argument("--reclaim-window", type=int, default=3)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--train-period", default="120D")
    parser.add_argument("--test-period", default="30D")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--tui", action="store_true")
    return parser.parse_args()


def _strategy_kwargs(args: argparse.Namespace, strategy_name: str) -> dict:
    if strategy_name == "session_breakout":
        return {"level_prefix": args.level_prefix}
    if strategy_name == "fvg_pullback":
        return {"timeframe": _effective_fvg_timeframe(args.base_timeframe, args.timeframe)}
    if strategy_name == "opening_range":
        return {"opening_range_prefix": args.opening_range_prefix}
    if strategy_name == "confluence_continuation":
        return {"score_threshold": args.score_threshold}
    if strategy_name == "sweep_reclaim":
        return {"reclaim_window": args.reclaim_window}
    return {}


def _build_specs(args: argparse.Namespace) -> list[StrategyRunSpec]:
    return [
        StrategyRunSpec(
            name=strategy_name,
            strategy_fn=STRATEGIES[strategy_name],
            strategy_kwargs=_strategy_kwargs(args, strategy_name),
        )
        for strategy_name in args.strategy
    ]


def main() -> None:
    args = _parse_args()
    if args.tui:
        launch_tui()
        return
    raw = load_intraday_data(
        dataset=args.dataset,
        symbols_prefix=args.symbols_prefix,
        start=args.start,
        end=args.end,
    )
    specs = _build_specs(args)

    if args.grid:
        if len(specs) != 1:
            raise ValueError("--grid currently supports one strategy at a time.")
        spec = specs[0]
        if spec.name == "fvg_pullback":
            summary = run_parameter_grid(
                raw,
                strategy_fn=spec.strategy_fn,
                base_name=spec.name,
                param_grid={"timeframe": ["1m", "5m", "15m"], "target_atr_multiple": [2.0, 2.5, 3.0]},
                max_workers=args.jobs,
                base_timeframe=args.base_timeframe,
            )
        elif spec.name == "sweep_reclaim":
            summary = run_parameter_grid(
                raw,
                strategy_fn=spec.strategy_fn,
                base_name=spec.name,
                param_grid={"reclaim_window": [1, 3, 5], "stop_buffer_atr": [0.25, 0.35, 0.5]},
                max_workers=args.jobs,
                base_timeframe=args.base_timeframe,
            )
        else:
            summary = run_parameter_grid(
                raw,
                strategy_fn=spec.strategy_fn,
                base_name=spec.name,
                param_grid={"max_hold_bars": [45, 90, 120]},
                max_workers=args.jobs,
                base_timeframe=args.base_timeframe,
            )
        out_path = Path(args.output_dir) / spec.name / "grid_results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(summary.head(20).to_string(index=False))
        print(f"\nSaved parameter grid to {out_path}")
        return

    if args.walk_forward:
        if len(specs) == 1:
            result = run_walk_forward_research(
                raw,
                specs[0],
                train_period=args.train_period,
                test_period=args.test_period,
                base_timeframe=args.base_timeframe,
            )
            print(result["results"].to_string(index=False))
        else:
            result = run_multi_walk_forward_research(
                raw,
                specs,
                train_period=args.train_period,
                test_period=args.test_period,
                base_timeframe=args.base_timeframe,
                max_workers=args.jobs,
            )
            print(result["summary"].to_string(index=False))
        return

    if len(specs) == 1:
        result = run_strategy_research(raw, specs[0], output_dir=args.output_dir, base_timeframe=args.base_timeframe)
        print(format_metrics(result["metrics"]))
        if not result["trades"].empty:
            print("\nRecent trades")
            print(result["trades"].tail(10).to_string(index=False))
        if result["report_paths"] is not None:
            print("\nReport files")
            for name, path in result["report_paths"].items():
                print(f"{name}: {path}")
        return

    multi = run_multi_strategy_research(
        raw,
        specs,
        output_dir=args.output_dir,
        max_workers=args.jobs,
        base_timeframe=args.base_timeframe,
    )
    print(multi["summary"].to_string(index=False))
    if multi["report_paths"] is not None:
        print("\nComparison files")
        for name, path in multi["report_paths"].items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
