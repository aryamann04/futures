from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import compute_extended_metrics


StrategyFn = Callable[..., pd.DataFrame]


@dataclass
class ValidationSpec:
    name: str
    strategy_fn: StrategyFn
    strategy_kwargs: Dict[str, Any]


def _pick_column(columns, candidates, required=True):
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def _coerce_timestamp_like(value, series: pd.Series) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    series_tz = getattr(series.dt, "tz", None)

    if series_tz is not None:
        if ts.tzinfo is None:
            ts = ts.tz_localize(series_tz)
        else:
            ts = ts.tz_convert(series_tz)
    else:
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)

    return ts


def _prepare_df(
    df: pd.DataFrame,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> tuple[pd.DataFrame, str, Optional[str]]:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]

    ts_col = ts_col or _pick_column(out.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    symbol_col = symbol_col or _pick_column(out.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"], required=False)

    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    out = out.dropna(subset=[ts_col]).sort_values(([symbol_col] if symbol_col else []) + [ts_col]).reset_index(drop=True)

    return out, ts_col, symbol_col


def _filter_date_range(
    df: pd.DataFrame,
    ts_col: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    out = df

    if start is not None:
        start = _coerce_timestamp_like(start, out[ts_col])
        out = out[out[ts_col] >= start]

    if end is not None:
        end = _coerce_timestamp_like(end, out[ts_col])
        out = out[out[ts_col] < end]

    return out.reset_index(drop=True)


def _run_strategy_on_slice(
    df_slice: pd.DataFrame,
    spec: ValidationSpec,
    initial_capital: float,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
    keep_bars: bool = False,
    keep_trades: bool = False,
) -> Dict[str, Any]:
    if df_slice.empty:
        return {
            "bars": pd.DataFrame() if keep_bars else None,
            "trades": pd.DataFrame() if keep_trades else None,
            "metrics": {},
        }

    plan = spec.strategy_fn(df_slice, **spec.strategy_kwargs)
    bars, trades = run_backtest(plan, ts_col=ts_col, symbol_col=symbol_col)
    metrics = compute_extended_metrics(trades, initial_capital=initial_capital, print_summary=False)

    return {
        "bars": bars if keep_bars else None,
        "trades": trades if keep_trades else None,
        "metrics": metrics,
    }


def _safe_series_mean(series: pd.Series):
    valid = pd.to_numeric(series, errors="coerce").dropna()
    return valid.mean() if not valid.empty else None


def _safe_series_median(series: pd.Series):
    valid = pd.to_numeric(series, errors="coerce").dropna()
    return valid.median() if not valid.empty else None


def run_out_of_sample_test(
    df: pd.DataFrame,
    spec: ValidationSpec,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
    initial_capital: float = 1000.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
    keep_train_outputs: bool = False,
    keep_test_outputs: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    df, ts_col, symbol_col = _prepare_df(df, ts_col=ts_col, symbol_col=symbol_col)

    train_df = _filter_date_range(df, ts_col=ts_col, start=train_start, end=train_end)
    test_df = _filter_date_range(df, ts_col=ts_col, start=test_start, end=test_end)

    if verbose:
        print("\n" + "=" * 100)
        print(f"OUT-OF-SAMPLE TEST: {spec.name}")
        print("=" * 100)
        print(f"Train rows: {len(train_df):,}")
        if not train_df.empty:
            print(f"Train range: {train_df[ts_col].min()} -> {train_df[ts_col].max()}")
        print(f"Test rows: {len(test_df):,}")
        if not test_df.empty:
            print(f"Test range: {test_df[ts_col].min()} -> {test_df[ts_col].max()}")

    train_result = _run_strategy_on_slice(
        train_df,
        spec=spec,
        initial_capital=initial_capital,
        ts_col=ts_col,
        symbol_col=symbol_col,
        keep_bars=keep_train_outputs,
        keep_trades=keep_train_outputs,
    )

    test_result = _run_strategy_on_slice(
        test_df,
        spec=spec,
        initial_capital=initial_capital,
        ts_col=ts_col,
        symbol_col=symbol_col,
        keep_bars=keep_test_outputs,
        keep_trades=keep_test_outputs,
    )

    summary = pd.DataFrame(
        [
            {
                "segment": "train",
                "n_trades": 0 if train_result["metrics"] == {} else train_result["metrics"].get("total_trades", None),
                "total_return": train_result["metrics"].get("total_return"),
                "cagr": train_result["metrics"].get("cagr"),
                "sharpe": train_result["metrics"].get("sharpe"),
                "max_drawdown": train_result["metrics"].get("max_drawdown"),
                "win_rate": train_result["metrics"].get("win_rate"),
                "avg_trade_pnl": train_result["metrics"].get("avg_trade_pnl"),
            },
            {
                "segment": "test",
                "n_trades": 0 if test_result["metrics"] == {} else test_result["metrics"].get("total_trades", None),
                "total_return": test_result["metrics"].get("total_return"),
                "cagr": test_result["metrics"].get("cagr"),
                "sharpe": test_result["metrics"].get("sharpe"),
                "max_drawdown": test_result["metrics"].get("max_drawdown"),
                "win_rate": test_result["metrics"].get("win_rate"),
                "avg_trade_pnl": test_result["metrics"].get("avg_trade_pnl"),
            },
        ]
    )

    if verbose:
        print("\nSummary")
        print(summary.to_string(index=False))

    return {
        "train_df": train_df if keep_train_outputs else None,
        "test_df": test_df if keep_test_outputs else None,
        "train_result": train_result,
        "test_result": test_result,
        "summary": summary,
    }


def _generate_walk_forward_windows(
    df: pd.DataFrame,
    ts_col: str,
    train_period: str,
    test_period: str,
    step_period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[Dict[str, pd.Timestamp]]:
    base_series = df[ts_col]
    data_start = _coerce_timestamp_like(start, base_series) if start is not None else base_series.min().normalize()
    data_end = _coerce_timestamp_like(end, base_series) if end is not None else base_series.max()

    train_delta = pd.Timedelta(train_period)
    test_delta = pd.Timedelta(test_period)
    step_delta = pd.Timedelta(step_period) if step_period is not None else test_delta

    windows = []
    anchor = data_start

    while True:
        train_start = anchor
        train_end = train_start + train_delta
        test_start = train_end
        test_end = test_start + test_delta

        if test_end > data_end:
            break

        windows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

        anchor = anchor + step_delta

    return windows


def run_walk_forward_validation(
    df: pd.DataFrame,
    spec: ValidationSpec,
    train_period: str = "120D",
    test_period: str = "30D",
    step_period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    initial_capital: float = 1000.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
    keep_test_trades: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    df, ts_col, symbol_col = _prepare_df(df, ts_col=ts_col, symbol_col=symbol_col)

    windows = _generate_walk_forward_windows(
        df=df,
        ts_col=ts_col,
        train_period=train_period,
        test_period=test_period,
        step_period=step_period,
        start=start,
        end=end,
    )

    if verbose:
        print("\n" + "=" * 100)
        print(f"WALK-FORWARD VALIDATION: {spec.name}")
        print("=" * 100)
        print(f"Number of windows: {len(windows)}")

    rows = []
    all_test_trades = []

    for idx, w in enumerate(windows, start=1):
        train_df = _filter_date_range(df, ts_col, w["train_start"], w["train_end"])
        test_df = _filter_date_range(df, ts_col, w["test_start"], w["test_end"])

        test_result = _run_strategy_on_slice(
            test_df,
            spec=spec,
            initial_capital=initial_capital,
            ts_col=ts_col,
            symbol_col=symbol_col,
            keep_bars=False,
            keep_trades=keep_test_trades,
        )

        test_metrics = test_result["metrics"]

        row = {
            "window": idx,
            "train_start": w["train_start"],
            "train_end": w["train_end"],
            "test_start": w["test_start"],
            "test_end": w["test_end"],
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "n_trades": test_metrics.get("total_trades"),
            "total_return": test_metrics.get("total_return"),
            "cagr": test_metrics.get("cagr"),
            "sharpe": test_metrics.get("sharpe"),
            "max_drawdown": test_metrics.get("max_drawdown"),
            "win_rate": test_metrics.get("win_rate"),
            "avg_trade_pnl": test_metrics.get("avg_trade_pnl"),
        }
        rows.append(row)

        if keep_test_trades and test_result["trades"] is not None and not test_result["trades"].empty:
            temp = test_result["trades"].copy()
            temp["window"] = idx
            temp["train_start"] = w["train_start"]
            temp["train_end"] = w["train_end"]
            temp["test_start"] = w["test_start"]
            temp["test_end"] = w["test_end"]
            all_test_trades.append(temp)

    results_df = pd.DataFrame(rows)

    for col in ["total_return", "sharpe", "max_drawdown", "win_rate", "n_trades"]:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    aggregate = {
        "n_windows": len(results_df),
        "mean_total_return": _safe_series_mean(results_df["total_return"]) if not results_df.empty else None,
        "median_total_return": _safe_series_median(results_df["total_return"]) if not results_df.empty else None,
        "mean_sharpe": _safe_series_mean(results_df["sharpe"]) if not results_df.empty else None,
        "median_sharpe": _safe_series_median(results_df["sharpe"]) if not results_df.empty else None,
        "mean_max_drawdown": _safe_series_mean(results_df["max_drawdown"]) if not results_df.empty else None,
        "median_max_drawdown": _safe_series_median(results_df["max_drawdown"]) if not results_df.empty else None,
        "mean_win_rate": _safe_series_mean(results_df["win_rate"]) if not results_df.empty else None,
        "total_trades": int(pd.to_numeric(results_df["n_trades"], errors="coerce").fillna(0).sum()) if not results_df.empty else 0,
    }

    if verbose:
        print("\nWindow-by-window results")
        if results_df.empty:
            print("No windows generated.")
        else:
            print(results_df.to_string(index=False))

    if verbose:
        print("\nAggregate")
        print(pd.Series(aggregate).to_string())

    all_test_trades_df = pd.concat(all_test_trades, axis=0, ignore_index=True) if all_test_trades else pd.DataFrame()

    return {
        "windows": windows,
        "results": results_df,
        "aggregate": aggregate,
        "test_trades": all_test_trades_df if keep_test_trades else None,
    }


def compare_specs_walk_forward(
    df: pd.DataFrame,
    specs: List[ValidationSpec],
    train_period: str = "120D",
    test_period: str = "30D",
    step_period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    initial_capital: float = 1000.0,
    ts_col: Optional[str] = None,
    symbol_col: Optional[str] = None,
) -> pd.DataFrame:
    rows = []

    for spec in specs:
        out = run_walk_forward_validation(
            df=df,
            spec=spec,
            train_period=train_period,
            test_period=test_period,
            step_period=step_period,
            start=start,
            end=end,
            initial_capital=initial_capital,
            ts_col=ts_col,
            symbol_col=symbol_col,
            keep_test_trades=False,
        )

        agg = out["aggregate"]
        rows.append(
            {
                "name": spec.name,
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
        summary = summary.sort_values(["mean_sharpe", "mean_total_return"], ascending=[False, False]).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("WALK-FORWARD COMPARISON")
    print("=" * 100)
    if summary.empty:
        print("No results.")
    else:
        print(summary.to_string(index=False))

    return summary
