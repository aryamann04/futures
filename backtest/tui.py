from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from backtest.metrics import format_metrics
from backtest.strategy_runner import (
    StrategyRunSpec,
    load_intraday_data,
    run_multi_strategy_research,
    run_multi_walk_forward_research,
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


console = Console()
LOGGER = logging.getLogger(__name__)

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


@dataclass
class TuiConfig:
    dataset: str
    strategies: list[str]
    symbols_prefix: Optional[str]
    start: str
    end: str
    base_timeframe: str
    fvg_timeframe: str
    jobs: Optional[int]
    output_dir: str
    walk_forward: bool
    train_period: str
    test_period: str


def _strategy_kwargs(config: TuiConfig, strategy_name: str) -> dict[str, Any]:
    if strategy_name == "session_breakout":
        return {"level_prefix": "london"}
    if strategy_name == "fvg_pullback":
        tf = config.fvg_timeframe
        if config.base_timeframe != "native" and timeframe_to_timedelta(tf) < timeframe_to_timedelta(config.base_timeframe):
            tf = config.base_timeframe
        return {"timeframe": tf}
    if strategy_name == "opening_range":
        return {"opening_range_prefix": "ny_open_range"}
    if strategy_name == "confluence_continuation":
        return {"score_threshold": 3}
    if strategy_name == "sweep_reclaim":
        return {"reclaim_window": 3}
    return {}


def _sparkline(values, width: int = 40) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    values = list(values)
    if not values:
        return ""
    if len(values) > width:
        step = max(1, len(values) // width)
        values = values[::step][:width]
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return blocks[0] * len(values)
    return "".join(blocks[min(len(blocks) - 1, int((v - lo) / (hi - lo) * (len(blocks) - 1)))] for v in values)


def _collect_config() -> TuiConfig:
    console.print(Panel("Backtest TUI\nSet the run config once, then launch parallel backtests from the terminal.", title="Parameter Panel"))
    dataset = Prompt.ask("Dataset", default="micro_sp_futures")
    strategy_input = Prompt.ask("Strategies (comma-separated or 'all')", default="all")
    strategies = sorted(STRATEGIES.keys()) if strategy_input.strip().lower() == "all" else [s.strip() for s in strategy_input.split(",") if s.strip()]
    symbols_prefix = Prompt.ask("Symbols prefix", default="MES")
    start = Prompt.ask("Start date", default="2024-01-01")
    end = Prompt.ask("End date", default="2026-02-28")
    base_timeframe = Prompt.ask("Base timeframe", default="15m", choices=["native", "1m", "5m", "15m", "1h"])
    fvg_timeframe = Prompt.ask("FVG timeframe", default=base_timeframe if base_timeframe != "native" else "15m", choices=["1m", "5m", "15m", "1h"])
    jobs_input = Prompt.ask("Parallel jobs (blank = auto)", default="")
    output_dir = Prompt.ask("Output directory", default="backtest/outputs/discretionary_tui")
    walk_forward = Confirm.ask("Run walk-forward validation?", default=False)
    train_period = "120D"
    test_period = "30D"
    if walk_forward:
        train_period = Prompt.ask("Train period", default="120D")
        test_period = Prompt.ask("Test period", default="30D")

    jobs = int(jobs_input) if jobs_input.strip() else None
    return TuiConfig(
        dataset=dataset,
        strategies=strategies,
        symbols_prefix=symbols_prefix or None,
        start=start,
        end=end,
        base_timeframe=base_timeframe,
        fvg_timeframe=fvg_timeframe,
        jobs=jobs,
        output_dir=output_dir,
        walk_forward=walk_forward,
        train_period=train_period,
        test_period=test_period,
    )


def _render_params(config: TuiConfig) -> Panel:
    table = Table.grid(padding=(0, 2))
    for key, value in [
        ("dataset", config.dataset),
        ("strategies", ", ".join(config.strategies)),
        ("symbols", config.symbols_prefix or "all"),
        ("date range", f"{config.start} -> {config.end}"),
        ("base timeframe", config.base_timeframe),
        ("fvg timeframe", config.fvg_timeframe),
        ("walk-forward", str(config.walk_forward)),
        ("jobs", str(config.jobs) if config.jobs is not None else "auto"),
    ]:
        table.add_row(f"[bold]{key}[/bold]", str(value))
    return Panel(table, title="Parameters")


def _render_progress(states: dict[str, dict[str, Any]]) -> Panel:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Strategy")
    table.add_column("Stage")
    table.add_column("Progress")
    table.add_column("Detail")
    for name in sorted(states):
        state = states[name]
        pct = int(state.get("progress", 0.0) * 100)
        table.add_row(name, state.get("stage", "queued"), f"{pct:3d}%", state.get("detail", ""))
    return Panel(table, title="Live Progress")


def _render_results(results: list[dict[str, Any]]) -> Panel:
    if not results:
        return Panel("Waiting for results...", title="Results")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Strategy")
    table.add_column("Trades", justify="right")
    table.add_column("PnL $", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Curve")
    for result in results:
        metrics = result.get("metrics", {})
        curve = result.get("equity_curve")
        spark = _sparkline(curve["equity"].tolist(), width=28) if curve is not None and not curve.empty else ""
        table.add_row(
            result["name"],
            str(metrics.get("total_trades", 0)),
            f"{metrics.get('total_pnl_dollars', float('nan')):,.2f}" if metrics else "nan",
            f"{metrics.get('sharpe', float('nan')):.2f}" if metrics else "nan",
            f"{metrics.get('max_drawdown', float('nan')):.2%}" if metrics else "nan",
            spark,
        )
    return Panel(table, title="Results")


def _render_comparison_chart(results: list[dict[str, Any]]) -> Panel:
    if not results:
        return Panel("No comparison chart yet.", title="Comparative View")
    lines = []
    for result in results:
        curve = result.get("equity_curve")
        spark = _sparkline(curve["equity"].tolist(), width=48) if curve is not None and not curve.empty else ""
        lines.append(f"{result['name']:<28} {spark}")
    return Panel("\n".join(lines), title="Comparative Equity")


def _build_specs(config: TuiConfig) -> list[StrategyRunSpec]:
    return [
        StrategyRunSpec(
            name=name,
            strategy_fn=STRATEGIES[name],
            strategy_kwargs=_strategy_kwargs(config, name),
        )
        for name in config.strategies
    ]


def _setup_debug_logger(output_dir: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "tui_debug.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    has_file_handler = any(isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == str(log_path) for handler in logger.handlers)
    if not has_file_handler:
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    return log_path


def launch_tui() -> None:
    config = _collect_config()
    log_path = _setup_debug_logger(config.output_dir)
    LOGGER.info("Launching TUI with config=%s", config)
    specs = _build_specs(config)
    raw = load_intraday_data(
        dataset=config.dataset,
        symbols_prefix=config.symbols_prefix,
        start=config.start,
        end=config.end,
    )

    states = {spec.name: {"stage": "queued", "progress": 0.0, "detail": "waiting"} for spec in specs}
    results: list[dict[str, Any]] = []
    result_box: dict[str, Any] = {"error": None}
    progress_events: "queue.Queue[tuple[str, str, float, str]]" = queue.Queue()

    def progress_callback(name: str, stage: str, progress: float, detail: str) -> None:
        progress_events.put((name, stage, progress, detail))

    def worker() -> None:
        try:
            if config.walk_forward:
                if len(specs) == 1:
                    states[specs[0].name] = {"stage": "walk_forward", "progress": 0.2, "detail": "running windows"}
                    result_box["walk_forward"] = run_walk_forward_research(
                        raw,
                        specs[0],
                        train_period=config.train_period,
                        test_period=config.test_period,
                        base_timeframe=config.base_timeframe,
                    )
                    states[specs[0].name] = {"stage": "complete", "progress": 1.0, "detail": "walk-forward complete"}
                else:
                    result_box["walk_forward_multi"] = run_multi_walk_forward_research(
                        raw,
                        specs,
                        train_period=config.train_period,
                        test_period=config.test_period,
                        base_timeframe=config.base_timeframe,
                        max_workers=config.jobs,
                        progress_callback=progress_callback,
                    )
            elif len(specs) == 1:
                result = run_strategy_research(
                    raw,
                    specs[0],
                    output_dir=config.output_dir,
                    base_timeframe=config.base_timeframe,
                    progress_callback=progress_callback,
                )
                results.append(result)
            else:
                multi = run_multi_strategy_research(
                    raw,
                    specs,
                    output_dir=config.output_dir,
                    max_workers=config.jobs,
                    base_timeframe=config.base_timeframe,
                    progress_callback=progress_callback,
                )
                results.extend(multi["results"])
                result_box["multi"] = multi
        except Exception as exc:
            LOGGER.exception("TUI worker failed")
            result_box["error"] = RuntimeError(f"{exc}\nDebug log: {log_path}")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    layout = Layout()
    layout.split_column(
        Layout(name="top", size=11),
        Layout(name="middle", size=12),
        Layout(name="bottom"),
    )
    layout["top"].split_row(Layout(name="params"), Layout(name="compare"))
    layout["middle"].split_row(Layout(name="progress"), Layout(name="results"))

    with Live(layout, console=console, refresh_per_second=8, screen=True):
        while thread.is_alive() or not progress_events.empty():
            while True:
                try:
                    name, stage, progress, detail = progress_events.get_nowait()
                except queue.Empty:
                    break
                states[name] = {"stage": stage, "progress": progress, "detail": detail}

            layout["params"].update(_render_params(config))
            layout["progress"].update(_render_progress(states))
            layout["results"].update(_render_results(results))
            layout["compare"].update(_render_comparison_chart(results))
            if result_box.get("error") is not None:
                break
            time.sleep(0.1)

        layout["params"].update(_render_params(config))
        layout["progress"].update(_render_progress(states))
        if config.walk_forward and "walk_forward" in result_box:
            wf = result_box["walk_forward"]
            layout["results"].update(Panel(wf["results"].to_string(index=False), title="Walk-Forward Results"))
            layout["compare"].update(Panel(str(wf["aggregate"]), title="Aggregate"))
        elif config.walk_forward and "walk_forward_multi" in result_box:
            wf_multi = result_box["walk_forward_multi"]
            layout["results"].update(Panel(wf_multi["summary"].to_string(index=False), title="Walk-Forward Summary"))
            summary_lines = []
            for item in wf_multi["results"]:
                agg = item["aggregate"]
                summary_lines.append(f"{item['name']}: windows={agg.get('n_windows', 0)} trades={agg.get('total_trades', 0)} mean_sharpe={agg.get('mean_sharpe')}")
            layout["compare"].update(Panel("\n".join(summary_lines), title="Walk-Forward Aggregate"))
        else:
            layout["results"].update(_render_results(results))
            layout["compare"].update(_render_comparison_chart(results))
        time.sleep(0.2)

    if result_box.get("error") is not None:
        raise result_box["error"]

    if results:
        console.print("\n[bold]Final Summary[/bold]")
        for result in results:
            console.print(Panel(format_metrics(result["metrics"]), title=result["name"]))
            if result.get("report_paths"):
                console.print("Reports:")
                for name, path in result["report_paths"].items():
                    console.print(f"  {name}: {path}")
    elif "walk_forward" in result_box:
        console.print("\n[bold]Walk-Forward Summary[/bold]")
        console.print(result_box["walk_forward"]["results"].to_string(index=False))
    elif "walk_forward_multi" in result_box:
        console.print("\n[bold]Walk-Forward Summary[/bold]")
        console.print(result_box["walk_forward_multi"]["summary"].to_string(index=False))
