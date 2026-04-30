from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import compute_extended_metrics, equity_curve
from backtest.validation import ValidationSpec, run_walk_forward_validation
from data.load import futures_data
from features.discretionary import build_discretionary_features
from features.research import _filter_raw_data
from reports.strategy_report import write_multi_strategy_report, write_strategy_report


StrategyFn = Callable[..., pd.DataFrame]
ProgressCallback = Callable[[str, str, float, str], None]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyRunSpec:
    name: str
    strategy_fn: StrategyFn
    strategy_kwargs: dict[str, Any] = field(default_factory=dict)
    commission_per_side: float = 0.75
    contract_specs: Optional[dict[str, Any]] = None


@dataclass
class PreparedResearchData:
    raw: pd.DataFrame
    features: pd.DataFrame
    fvg_events: pd.DataFrame
    sweep_events: pd.DataFrame
    base_timeframe: str = "native"


@lru_cache(maxsize=8)
def _cached_filtered_load(
    dataset: str,
    symbols_tuple: tuple[str, ...],
    symbols_prefix: Optional[str],
    start: Optional[str],
    end: Optional[str],
    chunksize: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    symbols = list(symbols_tuple) if symbols_tuple else None
    columns = ["ts_event", "symbol", "open", "high", "low", "close", "volume"]
    for chunk in futures_data(dataset=dataset, columns=columns, chunksize=chunksize):
        filtered = _filter_raw_data(
            chunk,
            symbols=symbols,
            symbols_prefix=symbols_prefix,
            include_spreads=False,
            start=start,
            end=end,
        )
        if not filtered.empty:
            parts.append(filtered)
    if not parts:
        return pd.DataFrame(columns=[c.lower() for c in columns])
    out = pd.concat(parts, axis=0, ignore_index=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce", downcast="float")
    return out


def load_intraday_data(
    dataset: str,
    *,
    symbols: Optional[list[str]] = None,
    symbols_prefix: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    symbols_tuple = tuple(symbols) if symbols is not None else tuple()
    return _cached_filtered_load(dataset, symbols_tuple, symbols_prefix, start, end, chunksize).copy()


def prepare_research_data(
    raw: pd.DataFrame,
    *,
    base_timeframe: str = "native",
    fvg_timeframes: tuple[str, ...] = ("1min", "5min", "15min", "1h"),
) -> PreparedResearchData:
    features, fvg_events, sweep_events = build_discretionary_features(
        raw,
        base_timeframe=base_timeframe,
        fvg_timeframes=fvg_timeframes,
    )
    if "atr_1min_14_regime" in features.columns and "volatility_regime" not in features.columns:
        features["volatility_regime"] = features["atr_1min_14_regime"]
    return PreparedResearchData(
        raw=raw,
        features=features,
        fvg_events=fvg_events,
        sweep_events=sweep_events,
        base_timeframe=base_timeframe,
    )


def _execute_spec(
    prepared: PreparedResearchData,
    spec: StrategyRunSpec,
    *,
    initial_capital: float,
    output_dir: Optional[str | Path],
    progress_callback: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    if progress_callback is not None:
        progress_callback(spec.name, "features_ready", 0.15, f"base_tf={prepared.base_timeframe}")
    features = prepared.features.copy()
    if progress_callback is not None:
        progress_callback(spec.name, "building_plan", 0.35, "creating signals")
    plan = spec.strategy_fn(features, **spec.strategy_kwargs)
    if progress_callback is not None:
        progress_callback(spec.name, "running_engine", 0.65, "simulating fills")
    bars, trades = run_backtest(
        plan,
        contract_specs=spec.contract_specs,
        commission_per_side=spec.commission_per_side,
    )
    if "volatility_regime" in features.columns and "entry_time" in trades.columns and not trades.empty:
        join_cols = ["ts_event", "volatility_regime"] + (["symbol"] if "symbol" in features.columns and "symbol" in trades.columns else [])
        lookup = features[join_cols].rename(columns={"ts_event": "entry_time"})
        trades = trades.merge(lookup, on=["entry_time"] + (["symbol"] if "symbol" in lookup.columns else []), how="left")
    metrics = compute_extended_metrics(trades, initial_capital=initial_capital, print_summary=False)
    report_paths = None
    if output_dir is not None:
        if progress_callback is not None:
            progress_callback(spec.name, "writing_report", 0.9, "saving outputs")
        report_paths = write_strategy_report(trades, Path(output_dir) / spec.name, initial_capital=initial_capital, metrics=metrics)
    if progress_callback is not None:
        progress_callback(spec.name, "complete", 1.0, f"trades={metrics.get('total_trades', 0)}")
    return {
        "name": spec.name,
        "plan": plan,
        "bars": bars,
        "trades": trades,
        "metrics": metrics,
        "equity_curve": equity_curve(trades, initial_capital=initial_capital),
        "report_paths": report_paths,
    }


def run_strategy_research(
    raw: pd.DataFrame,
    spec: StrategyRunSpec,
    *,
    output_dir: Optional[str | Path] = None,
    initial_capital: float = 10000.0,
    base_timeframe: str = "native",
    fvg_timeframes: tuple[str, ...] = ("1min", "5min", "15min", "1h"),
    progress_callback: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    prepared = prepare_research_data(raw, base_timeframe=base_timeframe, fvg_timeframes=fvg_timeframes)
    result = _execute_spec(
        prepared,
        spec,
        initial_capital=initial_capital,
        output_dir=output_dir,
        progress_callback=progress_callback,
    )
    result["features"] = prepared.features
    result["fvg_events"] = prepared.fvg_events
    result["sweep_events"] = prepared.sweep_events
    return result


def run_multi_strategy_research(
    raw: pd.DataFrame,
    specs: list[StrategyRunSpec],
    *,
    output_dir: Optional[str | Path] = None,
    initial_capital: float = 10000.0,
    max_workers: Optional[int] = None,
    base_timeframe: str = "native",
    fvg_timeframes: tuple[str, ...] = ("1min", "5min", "15min", "1h"),
    progress_callback: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    prepared = prepare_research_data(raw, base_timeframe=base_timeframe, fvg_timeframes=fvg_timeframes)
    if max_workers is None:
        max_workers = max(1, min(len(specs), 8))
    if progress_callback is not None:
        for spec in specs:
            progress_callback(spec.name, "queued", 0.0, "waiting")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _execute_spec,
                prepared,
                spec,
                initial_capital=initial_capital,
                output_dir=output_dir,
                progress_callback=progress_callback,
            ): spec.name
            for spec in specs
        }
        for future in as_completed(futures):
            results.append(future.result())

    results = sorted(results, key=lambda item: item["name"])
    summary = summarize_results(results)
    combined_report_paths = None
    if output_dir is not None:
        combined_report_paths = write_multi_strategy_report(results, output_dir=Path(output_dir) / "comparison")
    return {
        "prepared": prepared,
        "results": results,
        "summary": summary,
        "report_paths": combined_report_paths,
    }


def summarize_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        metrics = result.get("metrics", {})
        rows.append(
            {
                "name": result["name"],
                "total_trades": metrics.get("total_trades"),
                "total_pnl_dollars": metrics.get("total_pnl_dollars"),
                "total_return": metrics.get("total_return"),
                "cagr": metrics.get("cagr"),
                "sharpe": metrics.get("sharpe"),
                "sortino": metrics.get("sortino"),
                "calmar": metrics.get("calmar"),
                "max_drawdown": metrics.get("max_drawdown"),
                "win_rate": metrics.get("win_rate"),
                "profit_factor": metrics.get("profit_factor"),
                "avg_r": metrics.get("avg_r"),
                "trades_per_day": metrics.get("trades_per_day"),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["sortino", "sharpe", "total_pnl_dollars"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    return summary


def run_parameter_grid(
    raw: pd.DataFrame,
    *,
    strategy_fn: StrategyFn,
    base_name: str,
    param_grid: dict[str, list[Any]],
    initial_capital: float = 10000.0,
    max_workers: Optional[int] = None,
    base_timeframe: str = "native",
    fvg_timeframes: tuple[str, ...] = ("1min", "5min", "15min", "1h"),
    progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    keys = sorted(param_grid.keys())
    specs = [
        StrategyRunSpec(
            name=f"{base_name}__" + "__".join(f"{key}={value}" for key, value in zip(keys, values)),
            strategy_fn=strategy_fn,
            strategy_kwargs=dict(zip(keys, values)),
        )
        for values in product(*(param_grid[key] for key in keys))
    ]
    run = run_multi_strategy_research(
        raw,
        specs,
        output_dir=None,
        initial_capital=initial_capital,
        max_workers=max_workers,
        base_timeframe=base_timeframe,
        fvg_timeframes=fvg_timeframes,
        progress_callback=progress_callback,
    )
    summary = run["summary"].copy()
    for key in keys:
        summary[key] = [next(spec.strategy_kwargs[key] for spec in specs if spec.name == name) for name in summary["name"]]
    return summary


def run_walk_forward_research(
    raw: pd.DataFrame,
    spec: StrategyRunSpec,
    *,
    train_period: str = "120D",
    test_period: str = "30D",
    step_period: Optional[str] = None,
    initial_capital: float = 10000.0,
    base_timeframe: str = "native",
    fvg_timeframes: tuple[str, ...] = ("1min", "5min", "15min", "1h"),
) -> dict[str, Any]:
    LOGGER.debug(
        "Running walk-forward research for %s with base_timeframe=%s train_period=%s test_period=%s",
        spec.name,
        base_timeframe,
        train_period,
        test_period,
    )
    prepared = prepare_research_data(raw, base_timeframe=base_timeframe, fvg_timeframes=fvg_timeframes)
    validation_spec = ValidationSpec(
        name=spec.name,
        strategy_fn=spec.strategy_fn,
        strategy_kwargs=spec.strategy_kwargs,
    )
    return run_walk_forward_validation(
        prepared.features,
        spec=validation_spec,
        train_period=train_period,
        test_period=test_period,
        step_period=step_period,
        initial_capital=initial_capital,
        keep_test_trades=True,
        verbose=False,
    )


def run_multi_walk_forward_research(
    raw: pd.DataFrame,
    specs: list[StrategyRunSpec],
    *,
    train_period: str = "120D",
    test_period: str = "30D",
    step_period: Optional[str] = None,
    initial_capital: float = 10000.0,
    base_timeframe: str = "native",
    fvg_timeframes: tuple[str, ...] = ("1min", "5min", "15min", "1h"),
    max_workers: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    if max_workers is None:
        max_workers = max(1, min(len(specs), 8))
    if progress_callback is not None:
        for spec in specs:
            progress_callback(spec.name, "queued", 0.0, "waiting")

    results: list[dict[str, Any]] = []

    def _worker(spec: StrategyRunSpec) -> dict[str, Any]:
        if progress_callback is not None:
            progress_callback(spec.name, "preparing", 0.1, "building feature set")
        out = run_walk_forward_research(
            raw,
            spec,
            train_period=train_period,
            test_period=test_period,
            step_period=step_period,
            initial_capital=initial_capital,
            base_timeframe=base_timeframe,
            fvg_timeframes=fvg_timeframes,
        )
        if progress_callback is not None:
            total_windows = out["aggregate"].get("n_windows", 0)
            progress_callback(spec.name, "complete", 1.0, f"windows={total_windows}")
        return {"name": spec.name, **out}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, spec): spec.name for spec in specs}
        for future in as_completed(futures):
            results.append(future.result())

    results = sorted(results, key=lambda item: item["name"])
    rows = []
    for result in results:
        agg = result["aggregate"]
        rows.append(
            {
                "name": result["name"],
                "n_windows": agg.get("n_windows"),
                "total_trades": agg.get("total_trades"),
                "mean_total_return": agg.get("mean_total_return"),
                "median_total_return": agg.get("median_total_return"),
                "mean_sharpe": agg.get("mean_sharpe"),
                "median_sharpe": agg.get("median_sharpe"),
                "mean_max_drawdown": agg.get("mean_max_drawdown"),
                "mean_win_rate": agg.get("mean_win_rate"),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["mean_sharpe", "mean_total_return"], ascending=[False, False], na_position="last").reset_index(drop=True)
    return {"results": results, "summary": summary}
