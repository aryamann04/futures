# futures-backtest

Event-driven backtesting framework for CME micro futures strategies, with a full feature engineering pipeline and ML-based signal research.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

## Overview

This project backtests systematic intraday strategies on CME micro futures — primarily Micro EUR/USD (M6E) and Micro E-mini S&P 500 (MES) — using tick-level and second-bar data sourced from [Databento](https://databento.com). The core engine simulates bar-by-bar execution with slippage, commissions, stop/take-profit fills, and end-of-day flattening. A separate feature pipeline computes 200+ technical indicators (trend, momentum, volatility, session levels, Fibonacci retracements, rolling ranges) which feed both rule-based strategies and a LightGBM-based signal research workflow. Walk-forward validation and a Rich-powered TUI are included for out-of-sample testing and interactive exploration.

## Features

- **Backtest engine** (`backtest/engine.py`): bar-by-bar simulation supporting long/short entries, stop-loss/take-profit (absolute or %-based), max-hold-bars/seconds exits, end-of-day flattening, slippage ticks, and configurable commissions. Multi-symbol dispatch with per-symbol contract specs (multiplier, tick size, tick value).
- **Strategy DSL** (`backtest/strategies.py`): composable `Condition` functions (`crossed_above`, `col_gt`, `entered_band`, etc.) with `and_`/`or_`/`not_` combinators. `build_plan` and `confluence_strategy` produce signal DataFrames consumed by the engine. `EntryGate` enforces NY session windows, cooldown bars, and daily trade caps.
- **Implemented strategies**: EMA/ADX high-conviction, MACD+EMA+RSI confluence, opening range breakout (15m/30m) with VWAP and ADX confirmation, opening range retest, Donchian channel breakout, EMA-slope momentum pullback, VWAP trend pullback, Fibonacci golden-zone retracement, rolling-range bounce/breakout, session-level reversal, 15m OR breakout, 60-min EMA pullback.
- **Discretionary strategies** (`strategies/`): sweep-and-reclaim, FVG pullback, session breakout, confluence continuation, and several baselines — run via the TUI or strategy runner.
- **Feature pipeline** (`features/build_features.py`): per-symbol computation of SMAs/EMAs (20 windows), MACD, RSI (6 windows), ATR/NATR, ADX, Bollinger Bands, Stochastic, Williams %R, OBV, relative volume, session highs/lows, 5m/15m opening ranges, rolling high/low ranges (30m–1d), Fibonacci levels (236/382/500/618/786) anchored to 4h/8h/1d swings, and fair value gaps. All features are optionally shifted by 1 bar to prevent lookahead.
- **ML analysis** (`features/ml_analysis.py`): LightGBM classifiers and regressors trained on 60s/300s/600s forward-return horizons; gain-based and permutation importance aggregated by feature category and lookback window. Includes an OOS ML-signal backtest vs. buy-and-hold.
- **Metrics** (`backtest/metrics.py`): equity curve, CAGR, Sharpe, Sortino, Calmar, max drawdown, profit factor, win rate, avg/median R, drawdown clusters, and performance breakdowns by direction, hour, session, setup, and holding-time bucket.
- **Walk-forward validation** (`backtest/validation.py`): rolling train/test windows with configurable periods and step size.
- **Rich TUI** (`backtest/tui.py`): interactive terminal UI for configuring and running multi-strategy backtests in parallel, with live progress bars and sparkline equity curves.
- **HTML reports** (`reports/strategy_report.py`): per-strategy and multi-strategy HTML output with equity curve plots, trade diagnostics, and breakdown tables.

## Project Structure

```
futures/
├── data/
│   ├── config.py               # WRDS credentials, base paths
│   ├── load.py                 # CSV loader for Databento datasets
│   └── micro-*-databento/      # Raw data (gitignored)
├── features/
│   ├── build_features.py       # Main feature engineering pipeline
│   ├── discretionary.py        # Features for discretionary strategies
│   ├── ml_analysis.py          # LightGBM feature importance + ML backtest
│   ├── research.py             # Data filtering utilities
│   ├── resample.py             # Timeframe resampling
│   ├── atr.py, fvg.py, vwap.py, structure.py, sweeps.py, ...
│   └── session_levels.py, volume_profile.py, confluence.py
├── backtest/
│   ├── engine.py               # Core bar-by-bar simulation engine
│   ├── strategies.py           # Strategy DSL + implemented strategies
│   ├── metrics.py              # Performance metrics and plot helpers
│   ├── strategy_runner.py      # High-level runner with parallelism + reports
│   ├── validation.py           # Walk-forward validation
│   ├── run_backtest.py         # Script: feature families + strategy library
│   ├── run_experiments.py      # Script: walk-forward experiments (M6E)
│   ├── run_discretionary_backtests.py
│   └── tui.py                  # Rich interactive TUI
├── strategies/                 # Discretionary strategy implementations
│   ├── session_breakout.py
│   ├── sweep_reclaim.py
│   ├── fvg_pullback.py
│   ├── opening_range.py
│   ├── confluence_continuation.py
│   └── baselines.py
├── eda/                        # Exploratory notebooks/scripts
├── reports/                    # HTML report generation
└── tests/                      # pytest test suite
```

## Setup

**Prerequisites**: Python 3.10+, TA-Lib C library (`brew install ta-lib` on macOS).

```bash
pip install pandas numpy matplotlib ta-lib lightgbm scikit-learn rich python-dotenv
```

Create a `.env` file in the project root (required only for WRDS data access):

```
WRDS_USERNAME=your_username
```

Place Databento CSV exports in:
```
data/micro-currency-futures-databento/data.csv
data/micro-sp-futures-databento/data.csv
```

## Usage

**Run the full strategy library on MES data:**

```bash
cd /path/to/futures
python -m backtest.run_backtest
```

**Run walk-forward experiments on M6E:**

```bash
python -m backtest.run_experiments
```

**Launch the interactive TUI:**

```bash
python -m backtest.tui
```

**Run ML feature importance analysis:**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python -m features.ml_analysis
```

**Use the engine and strategies directly:**

```python
from data.load import futures_data
from features.build_features import build_features
from backtest.strategies import opening_range_breakout, TradeParams
from backtest.engine import run_backtest
from backtest.metrics import compute_extended_metrics

raw = futures_data(dataset="micro_sp_futures")
feat = build_features(raw, add_session_levels=True, add_rolling_ranges=True, bar_seconds=60)

plan = opening_range_breakout(
    feat,
    opening_range_minutes=30,
    trade_params=TradeParams(stop_loss_pct=0.0022, take_profit_pct=0.0055, max_hold_seconds=5400),
)

_, trades = run_backtest(plan)
metrics = compute_extended_metrics(trades, initial_capital=10_000, print_summary=True)
```

**Compose a custom strategy with the DSL:**

```python
from backtest.strategies import confluence_strategy, crossed_above, col_gte, TradeParams, EntryGate

plan = confluence_strategy(
    feat,
    long_conditions=[crossed_above("rsi_14", 30), col_gte("adx_14", 20)],
    short_conditions=[crossed_above("rsi_14", 70), col_gte("adx_14", 20)],
    trade_params=TradeParams(stop_loss_pct=0.002, take_profit_pct=0.004),
    gate=EntryGate(trade_start="09:30", trade_end="12:00", cooldown_bars=60, max_trades_per_day=2),
)
_, trades = run_backtest(plan)
```
