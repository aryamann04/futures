from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SECONDS_PER_YEAR = 365 * 24 * 60 * 60


def _validate_trades(trades: pd.DataFrame) -> pd.DataFrame:
    required = ["entry_time", "exit_time", "pnl_dollars"]
    for col in required:
        if col not in trades.columns:
            raise ValueError(f"Missing required column: {col}")

    out = trades.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
    out["pnl_dollars"] = pd.to_numeric(out["pnl_dollars"], errors="coerce")

    optional_numeric = ["ret", "ret_notional", "notional_entry", "pnl_points"]
    for col in optional_numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["entry_time", "exit_time", "pnl_dollars"])
    out = out.sort_values("exit_time").reset_index(drop=True)
    return out


def equity_curve(trades: pd.DataFrame, initial_capital: float = 1000.0) -> pd.DataFrame:
    trades = _validate_trades(trades)

    if len(trades) == 0:
        return pd.DataFrame(columns=["time", "equity", "pnl_dollars", "returns", "drawdown"])

    equity = [float(initial_capital)]
    times = [trades["entry_time"].iloc[0]]
    pnl_series = [0.0]

    current_equity = float(initial_capital)

    for _, row in trades.iterrows():
        current_equity = current_equity + float(row["pnl_dollars"])
        equity.append(current_equity)
        times.append(row["exit_time"])
        pnl_series.append(float(row["pnl_dollars"]))

    curve = pd.DataFrame({
        "time": times,
        "equity": equity,
        "pnl_dollars": pnl_series,
    })

    curve["returns"] = curve["equity"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    curve["running_max"] = curve["equity"].cummax()
    curve["drawdown"] = (curve["equity"] - curve["running_max"]) / curve["running_max"]
    curve = curve.drop(columns=["running_max"])

    return curve


def compute_basic_metrics(trades: pd.DataFrame, initial_capital: float = 10000.0):
    trades = _validate_trades(trades)

    if len(trades) == 0:
        print("No trades.")
        return {}

    curve = equity_curve(trades, initial_capital=initial_capital)

    start_equity = float(curve["equity"].iloc[0])
    end_equity = float(curve["equity"].iloc[-1])

    total_pnl_dollars = end_equity - start_equity
    total_return = end_equity / start_equity - 1.0 if start_equity != 0 else np.nan

    start = curve["time"].iloc[0]
    end = curve["time"].iloc[-1]
    total_seconds = (end - start).total_seconds()

    if total_seconds <= 0 or start_equity <= 0 or end_equity <= 0:
        cagr = np.nan
    else:
        annual_factor = SECONDS_PER_YEAR / total_seconds
        log_return = np.log(end_equity / start_equity)

        if not np.isfinite(annual_factor) or not np.isfinite(log_return):
            cagr = np.nan
        elif annual_factor > 1e4:
            cagr = np.nan
        else:
            annualized_log_return = log_return * annual_factor

            if annualized_log_return > 700:
                cagr = np.nan
            elif annualized_log_return < -700:
                cagr = -1.0
            else:
                cagr = np.exp(annualized_log_return) - 1.0

    returns = curve["returns"]
    vol = returns.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        sharpe = np.nan
    else:
        sharpe = (returns.mean() / vol) * np.sqrt(len(returns))

    max_drawdown = float(curve["drawdown"].min())

    win_rate = float((trades["pnl_dollars"] > 0).mean())
    avg_trade_pnl = float(trades["pnl_dollars"].mean())
    median_trade_pnl = float(trades["pnl_dollars"].median())

    avg_trade_return = float(trades["ret"].mean()) if "ret" in trades.columns else np.nan
    median_trade_return = float(trades["ret"].median()) if "ret" in trades.columns else np.nan

    avg_ret_notional = float(trades["ret_notional"].mean()) if "ret_notional" in trades.columns else np.nan
    median_ret_notional = float(trades["ret_notional"].median()) if "ret_notional" in trades.columns else np.nan

    print("\n" + "=" * 60)
    print("BACKTEST METRICS")
    print("=" * 60)
    print(f"Initial Capital     : {initial_capital:,.2f}")
    print(f"Ending Capital      : {end_equity:,.2f}")
    print(f"Total PnL ($)       : {total_pnl_dollars:,.2f}")
    print(f"Total Return        : {total_return:.4f}")
    print(f"CAGR                : {cagr:.4f}" if np.isfinite(cagr) else "CAGR                : nan")
    print(f"Sharpe              : {sharpe:.4f}" if np.isfinite(sharpe) else "Sharpe              : nan")
    print(f"Max Drawdown        : {max_drawdown:.4f}")
    print(f"Total Trades        : {len(trades):,}")
    print(f"Win Rate            : {win_rate:.4f}")
    print(f"Avg Trade PnL ($)   : {avg_trade_pnl:,.2f}")
    print(f"Median Trade PnL ($): {median_trade_pnl:,.2f}")
    print(f"Avg Trade Return    : {avg_trade_return:.6f}")
    print(f"Median Trade Return : {median_trade_return:.6f}")
    print(f"Avg Ret on Notional : {avg_ret_notional:.6f}")
    print(f"Median Ret Notional : {median_ret_notional:.6f}")
    print("=" * 60)

    return {
        "initial_capital": initial_capital,
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
        "total_trades": len(trades),
    }


def print_equity_curve(trades: pd.DataFrame, initial_capital: float = 10000.0):
    curve = equity_curve(trades, initial_capital=initial_capital)

    print(f"\nEquity Curve (${initial_capital:,.2f} initial):\n")
    print(curve.tail(20).to_string(index=False))

    return curve


def plot_equity_curve(trades: pd.DataFrame, initial_capital: float = 10000.0):
    curve = equity_curve(trades, initial_capital=initial_capital)

    plt.figure()
    plt.plot(curve["time"], curve["equity"])
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.title(f"Equity Curve (Starting Capital = ${initial_capital:,.2f})")
    plt.show()

    return curve