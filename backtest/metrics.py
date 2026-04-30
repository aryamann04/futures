from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SECONDS_PER_YEAR = 365 * 24 * 60 * 60


def _validate_trades(trades: pd.DataFrame) -> pd.DataFrame:
    required = ["entry_time", "exit_time", "pnl_dollars"]
    for col in required:
        if col not in trades.columns:
            raise ValueError(f"Missing required column: {col}")

    out = trades.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
    for col in ["pnl_dollars", "ret", "ret_notional", "notional_entry", "pnl_points", "pnl_r", "side", "bars_held"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["entry_time", "exit_time", "pnl_dollars"]).sort_values("exit_time").reset_index(drop=True)
    return out


def equity_curve(trades: pd.DataFrame, initial_capital: float = 1000.0) -> pd.DataFrame:
    trades = _validate_trades(trades)
    if trades.empty:
        return pd.DataFrame(columns=["time", "equity", "pnl_dollars", "returns", "drawdown"])

    pnl = trades["pnl_dollars"].astype(float)
    equity = initial_capital + pnl.cumsum()
    curve = pd.DataFrame(
        {
            "time": pd.concat(
                [
                    pd.Series([trades["entry_time"].iloc[0]]),
                    trades["exit_time"].reset_index(drop=True),
                ],
                ignore_index=True,
            ),
            "equity": pd.concat(
                [
                    pd.Series([float(initial_capital)]),
                    equity.reset_index(drop=True),
                ],
                ignore_index=True,
            ),
            "pnl_dollars": pd.concat([pd.Series([0.0]), pnl.reset_index(drop=True)], ignore_index=True),
        }
    )
    curve["returns"] = curve["equity"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    curve["running_max"] = curve["equity"].cummax()
    curve["drawdown"] = (curve["equity"] - curve["running_max"]) / curve["running_max"]
    return curve.drop(columns=["running_max"])


def _safe_cagr(start_equity: float, end_equity: float, total_seconds: float) -> float:
    if total_seconds <= 0 or start_equity <= 0 or end_equity <= 0:
        return np.nan
    annual_factor = SECONDS_PER_YEAR / total_seconds
    log_return = np.log(end_equity / start_equity)
    if not np.isfinite(annual_factor) or not np.isfinite(log_return) or annual_factor > 1e4:
        return np.nan
    annualized_log_return = log_return * annual_factor
    if annualized_log_return > 700:
        return np.nan
    if annualized_log_return < -700:
        return -1.0
    return float(np.exp(annualized_log_return) - 1.0)


def _annualized_ratio(returns: pd.Series, downside_only: bool = False) -> float:
    valid = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 2:
        return np.nan
    mean = float(valid.mean())
    denom_series = valid[valid < 0] if downside_only else valid
    std = float(denom_series.std(ddof=1))
    if std == 0 or np.isnan(std):
        return np.nan
    return mean / std * np.sqrt(len(valid))


def _max_consecutive(values: pd.Series, positive: bool = True) -> int:
    if values.empty:
        return 0
    mask = values.gt(0) if positive else values.lt(0)
    best = 0
    current = 0
    for flag in mask.fillna(False):
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def compute_basic_metrics(
    trades: pd.DataFrame,
    initial_capital: float = 10000.0,
    *,
    print_summary: bool = False,
) -> dict:
    trades = _validate_trades(trades)
    if trades.empty:
        return {}

    curve = equity_curve(trades, initial_capital=initial_capital)
    start_equity = float(curve["equity"].iloc[0])
    end_equity = float(curve["equity"].iloc[-1])
    total_pnl_dollars = end_equity - start_equity
    total_return = end_equity / start_equity - 1.0 if start_equity != 0 else np.nan
    total_seconds = float((curve["time"].iloc[-1] - curve["time"].iloc[0]).total_seconds())
    cagr = _safe_cagr(start_equity, end_equity, total_seconds)
    sharpe = _annualized_ratio(curve["returns"], downside_only=False)
    max_drawdown = float(curve["drawdown"].min()) if not curve.empty else np.nan
    win_rate = float((trades["pnl_dollars"] > 0).mean())
    avg_trade_pnl = float(trades["pnl_dollars"].mean())
    median_trade_pnl = float(trades["pnl_dollars"].median())
    avg_trade_return = float(trades["ret"].mean()) if "ret" in trades.columns else np.nan
    median_trade_return = float(trades["ret"].median()) if "ret" in trades.columns else np.nan
    avg_ret_notional = float(trades["ret_notional"].mean()) if "ret_notional" in trades.columns else np.nan
    median_ret_notional = float(trades["ret_notional"].median()) if "ret_notional" in trades.columns else np.nan

    metrics = {
        "initial_capital": float(initial_capital),
        "ending_capital": end_equity,
        "total_pnl_dollars": total_pnl_dollars,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "avg_trade_pnl": avg_trade_pnl,
        "median_trade_pnl": median_trade_pnl,
        "avg_return": avg_trade_return,
        "median_return": median_trade_return,
        "avg_ret_notional": avg_ret_notional,
        "median_ret_notional": median_ret_notional,
        "total_trades": int(len(trades)),
    }
    if print_summary:
        print(format_metrics(metrics))
    return metrics


def trade_diagnostics(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    trades = _validate_trades(trades)
    if trades.empty:
        empty = pd.DataFrame()
        return {
            "by_direction": empty,
            "by_hour": empty,
            "by_session": empty,
            "by_setup": empty,
            "holding_profitability": empty,
            "return_distribution": empty,
            "drawdown_clusters": empty,
        }

    work = trades.copy()
    local_exit = work["exit_time"].dt.tz_convert("America/New_York") if getattr(work["exit_time"].dt, "tz", None) is not None else work["exit_time"]
    work["direction"] = work["side"].map({1: "long", -1: "short"}).fillna("unknown")
    work["exit_hour_ny"] = local_exit.dt.hour
    work["holding_minutes"] = (work["exit_time"] - work["entry_time"]).dt.total_seconds() / 60.0
    if "bars_held" in work.columns:
        work["holding_bucket"] = pd.cut(
            work["holding_minutes"],
            bins=[-np.inf, 5, 15, 30, 60, 120, np.inf],
            labels=["<=5m", "5-15m", "15-30m", "30-60m", "60-120m", ">120m"],
        )
    else:
        work["holding_bucket"] = "unknown"

    by_direction = performance_breakdown(work, by="direction")
    by_hour = performance_breakdown(work, by="exit_hour_ny")
    by_session = performance_breakdown(work, by="session_name")
    by_setup = performance_breakdown(work, by="setup")
    by_regime = performance_breakdown(work, by="volatility_regime") if "volatility_regime" in work.columns else pd.DataFrame()
    holding_profitability = (
        work.groupby("holding_bucket", observed=False)
        .agg(
            trades=("pnl_dollars", "size"),
            avg_pnl_dollars=("pnl_dollars", "mean"),
            median_pnl_dollars=("pnl_dollars", "median"),
            win_rate=("pnl_dollars", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    return_distribution = pd.DataFrame(
        {
            "metric": ["p05", "p25", "p50", "p75", "p95", "std", "mean"],
            "pnl_dollars": [
                float(work["pnl_dollars"].quantile(0.05)),
                float(work["pnl_dollars"].quantile(0.25)),
                float(work["pnl_dollars"].quantile(0.50)),
                float(work["pnl_dollars"].quantile(0.75)),
                float(work["pnl_dollars"].quantile(0.95)),
                float(work["pnl_dollars"].std(ddof=1)),
                float(work["pnl_dollars"].mean()),
            ],
        }
    )
    curve = equity_curve(work)
    if curve.empty:
        drawdown_clusters = pd.DataFrame()
    else:
        underwater = curve["drawdown"] < 0
        cluster_ids = underwater.ne(underwater.shift(fill_value=False)).cumsum()
        drawdown_clusters = (
            curve.loc[underwater]
            .assign(cluster_id=cluster_ids[underwater].to_numpy())
            .groupby("cluster_id", observed=False)
            .agg(
                start_time=("time", "min"),
                end_time=("time", "max"),
                min_drawdown=("drawdown", "min"),
                points=("time", "size"),
            )
            .reset_index(drop=True)
            .sort_values("min_drawdown")
        )

    diagnostics = {
        "by_direction": by_direction,
        "by_hour": by_hour,
        "by_session": by_session,
        "by_setup": by_setup,
        "by_regime": by_regime,
        "holding_profitability": holding_profitability,
        "return_distribution": return_distribution,
        "drawdown_clusters": drawdown_clusters,
    }
    return diagnostics


def compute_extended_metrics(
    trades: pd.DataFrame,
    initial_capital: float = 10000.0,
    *,
    print_summary: bool = False,
) -> dict:
    trades = _validate_trades(trades)
    if trades.empty:
        return {}

    basic = compute_basic_metrics(trades, initial_capital=initial_capital, print_summary=False)
    winners = trades.loc[trades["pnl_dollars"] > 0, "pnl_dollars"]
    losers = trades.loc[trades["pnl_dollars"] < 0, "pnl_dollars"]
    gross_profit = float(winners.sum()) if not winners.empty else 0.0
    gross_loss = float(losers.abs().sum()) if not losers.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
    expectancy = float(trades["pnl_dollars"].mean())
    avg_winner = float(winners.mean()) if not winners.empty else np.nan
    avg_loser = float(losers.mean()) if not losers.empty else np.nan
    avg_r = float(trades["pnl_r"].mean()) if "pnl_r" in trades.columns else np.nan
    median_r = float(trades["pnl_r"].median()) if "pnl_r" in trades.columns else np.nan
    avg_holding_minutes = float((trades["exit_time"] - trades["entry_time"]).dt.total_seconds().mean() / 60.0)
    trade_days = (
        trades["entry_time"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d").nunique()
        if getattr(trades["entry_time"].dt, "tz", None) is not None
        else trades["entry_time"].dt.strftime("%Y-%m-%d").nunique()
    )
    trades_per_day = len(trades) / trade_days if trade_days else np.nan
    curve = equity_curve(trades, initial_capital=initial_capital)
    sortino = _annualized_ratio(curve["returns"], downside_only=True)
    cagr = basic.get("cagr", np.nan)
    max_drawdown_abs = abs(float(basic.get("max_drawdown", np.nan)))
    calmar = cagr / max_drawdown_abs if np.isfinite(cagr) and np.isfinite(max_drawdown_abs) and max_drawdown_abs > 0 else np.nan

    basic.update(
        {
            "avg_winner": avg_winner,
            "avg_loser": avg_loser,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "avg_r": avg_r,
            "median_r": median_r,
            "sortino": sortino,
            "calmar": calmar,
            "max_consecutive_wins": _max_consecutive(trades["pnl_dollars"], positive=True),
            "max_consecutive_losses": _max_consecutive(trades["pnl_dollars"], positive=False),
            "average_holding_minutes": avg_holding_minutes,
            "trades_per_day": trades_per_day,
        }
    )
    if print_summary:
        print(format_metrics(basic))
    return basic


def performance_breakdown(trades: pd.DataFrame, by: str) -> pd.DataFrame:
    trades = _validate_trades(trades)
    if trades.empty or by not in trades.columns:
        return pd.DataFrame()
    agg_map = {
        "trades": ("pnl_dollars", "size"),
        "total_pnl_dollars": ("pnl_dollars", "sum"),
        "avg_pnl_dollars": ("pnl_dollars", "mean"),
        "median_pnl_dollars": ("pnl_dollars", "median"),
        "win_rate": ("pnl_dollars", lambda s: float((s > 0).mean())),
    }
    if "pnl_r" in trades.columns:
        agg_map["avg_r"] = ("pnl_r", "mean")
    return (
        trades.groupby(by, observed=False)
        .agg(**agg_map)
        .reset_index()
        .sort_values("total_pnl_dollars", ascending=False)
    )


def format_metrics(metrics: dict) -> str:
    if not metrics:
        return "No trades."
    ordered_keys = [
        "total_trades",
        "total_pnl_dollars",
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "expectancy",
        "avg_r",
        "median_r",
        "average_holding_minutes",
        "trades_per_day",
    ]
    lines = ["BACKTEST METRICS"]
    for key in ordered_keys:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, (float, np.floating)):
                lines.append(f"{key:>24}: {value:,.6f}" if np.isfinite(value) else f"{key:>24}: nan")
            else:
                lines.append(f"{key:>24}: {value}")
    return "\n".join(lines)


def print_equity_curve(trades: pd.DataFrame, initial_capital: float = 10000.0) -> pd.DataFrame:
    curve = equity_curve(trades, initial_capital=initial_capital)
    print(f"\nEquity Curve (${initial_capital:,.2f} initial):\n")
    print(curve.tail(20).to_string(index=False))
    return curve


def plot_equity_curve(
    trades: pd.DataFrame,
    initial_capital: float = 10000.0,
    *,
    ax=None,
    label: Optional[str] = None,
    title: Optional[str] = None,
):
    curve = equity_curve(trades, initial_capital=initial_capital)
    if curve.empty:
        return curve
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curve["time"], curve["equity"], label=label or "strategy")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity ($)")
    ax.set_title(title or f"Equity Curve (Starting Capital = ${initial_capital:,.2f})")
    if label is not None:
        ax.legend()
    return curve


def plot_drawdown_curve(trades: pd.DataFrame, initial_capital: float = 10000.0, *, ax=None, title: str = "Drawdown Curve"):
    curve = equity_curve(trades, initial_capital=initial_capital)
    if curve.empty:
        return curve
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(curve["time"], curve["drawdown"], color="crimson")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown")
    return curve


def plot_trade_return_distribution(trades: pd.DataFrame, *, ax=None, title: str = "Trade Return Distribution"):
    trades = _validate_trades(trades)
    if trades.empty:
        return trades
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(trades["pnl_dollars"], bins=40, color="steelblue", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Trade PnL ($)")
    ax.set_ylabel("Frequency")
    return trades


def plot_equity_comparison(
    curves: dict[str, pd.DataFrame],
    *,
    ax=None,
    title: str = "Equity Curve Comparison",
):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))
    for name, curve in curves.items():
        if curve.empty:
            continue
        ax.plot(curve["time"], curve["equity"], label=name)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    return curves
