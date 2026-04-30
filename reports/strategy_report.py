from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtest.metrics import (
    compute_extended_metrics,
    equity_curve,
    performance_breakdown,
    plot_drawdown_curve,
    plot_equity_comparison,
    plot_equity_curve,
    plot_trade_return_distribution,
    trade_diagnostics,
)


def _save_current_fig(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def write_strategy_report(
    trades: pd.DataFrame,
    output_dir: str | Path,
    summary_name: str = "summary.csv",
    trades_name: str = "trades.csv",
    equity_name: str = "equity_curve.csv",
    initial_capital: float = 10000.0,
    metrics: Optional[dict] = None,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = metrics or compute_extended_metrics(trades, initial_capital=initial_capital, print_summary=False)
    summary = pd.DataFrame([metrics])
    equity = equity_curve(trades, initial_capital=initial_capital)
    setup_breakdown = performance_breakdown(trades, by="setup")
    symbol_breakdown = performance_breakdown(trades, by="symbol")
    session_breakdown = performance_breakdown(trades, by="session_name")
    diagnostics = trade_diagnostics(trades)

    summary_path = out_dir / summary_name
    trades_path = out_dir / trades_name
    equity_path = out_dir / equity_name
    setup_path = out_dir / "performance_by_setup.csv"
    symbol_path = out_dir / "performance_by_symbol.csv"
    session_path = out_dir / "performance_by_session.csv"
    direction_path = out_dir / "performance_by_direction.csv"
    hour_path = out_dir / "performance_by_hour.csv"
    holding_path = out_dir / "holding_profitability.csv"
    distribution_path = out_dir / "return_distribution.csv"
    drawdown_cluster_path = out_dir / "drawdown_clusters.csv"

    summary.to_csv(summary_path, index=False)
    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path, index=False)
    setup_breakdown.to_csv(setup_path, index=False)
    symbol_breakdown.to_csv(symbol_path, index=False)
    session_breakdown.to_csv(session_path, index=False)
    diagnostics["by_direction"].to_csv(direction_path, index=False)
    diagnostics["by_hour"].to_csv(hour_path, index=False)
    diagnostics["holding_profitability"].to_csv(holding_path, index=False)
    diagnostics["return_distribution"].to_csv(distribution_path, index=False)
    diagnostics["drawdown_clusters"].to_csv(drawdown_cluster_path, index=False)
    if not diagnostics["by_regime"].empty:
        diagnostics["by_regime"].to_csv(out_dir / "performance_by_regime.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    plot_equity_curve(trades, initial_capital=initial_capital, ax=axes[0], title="Equity Curve")
    plot_drawdown_curve(trades, initial_capital=initial_capital, ax=axes[1], title="Drawdown Curve")
    plot_trade_return_distribution(trades, ax=axes[2], title="Trade PnL Distribution")
    plots_path = _save_current_fig(out_dir / "strategy_plots.png")

    return {
        "summary": summary_path,
        "trades": trades_path,
        "equity": equity_path,
        "setup_breakdown": setup_path,
        "symbol_breakdown": symbol_path,
        "session_breakdown": session_path,
        "direction_breakdown": direction_path,
        "hour_breakdown": hour_path,
        "holding_profitability": holding_path,
        "return_distribution": distribution_path,
        "drawdown_clusters": drawdown_cluster_path,
        "plots": plots_path,
    }


def write_multi_strategy_report(results: list[dict], output_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{"name": result["name"], **result["metrics"]} for result in results])
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    curves = {result["name"]: result["equity_curve"] for result in results}
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_equity_comparison(curves, ax=ax)
    comparison_plot = _save_current_fig(out_dir / "equity_comparison.png")

    fig, axes = plt.subplots(max(1, len(results)), 1, figsize=(12, max(4, len(results) * 3.5)))
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        plot_equity_curve(result["trades"], ax=ax, label=result["name"], title=f"{result['name']} Equity Curve")
    individual_plot = _save_current_fig(out_dir / "individual_equity_curves.png")

    return {
        "summary": summary_path,
        "comparison_plot": comparison_plot,
        "individual_plots": individual_plot,
    }
