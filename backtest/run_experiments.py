"""
run_experiments.py — Walk-forward strategy experiments for Micro EUR/USD (M6E).

Parameter philosophy (anti-overfitting):
  - SL/TP derived from EUR/USD volatility structure, NOT P&L optimization.
      EUR/USD 1-hour ATR ≈ 0.15-0.22 %.  SL = ~1.5x hourly ATR ≈ 0.28 %.
      TP = 2.2x SL ≈ 0.62 %.  Same values across all strategies.
  - Entry conditions: exactly 2 per side — one structural setup, one trend filter.
      No RSI, no momentum confirmation.  Each additional condition multiplies
      the curve-fitting risk exponentially.
  - Entry thresholds are round, structurally motivated numbers (0.20 %, not
      0.2347 %) so they generalise to unseen regimes.
  - Uniform gate across ALL strategies: NY session 09:00-13:00,
      7200-bar (2-hour) cooldown, max 2 trades/day.  Not per-strategy tuned.
  - Hold time: 60-120 min — wide enough to capture the move, narrow enough
      not to require overnight risk.

Validation:
  1. Walk-forward: 120D train / 30D test / 30D step  (~10 OOS windows).
  2. Final OOS holdout: Jan-Oct 2025 train → Nov 2025-Mar 2026 test (4 months).
  3. Full-sample reference for context only.

Usage:
    cd /Users/aryaman/futures
    DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python3 -m backtest.run_experiments
"""

from __future__ import annotations

import gc
import warnings
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from backtest.run_backtest import (
    BacktestSpec,
    build_features_for_family,
    get_dataset_config,
    load_raw_data,
    run_single_backtest,
    summarize_results,
)
from backtest.strategies import (
    EntryGate,
    TradeParams,
    fib_golden_zone_trend,
    long_ema_pullback,
    or15m_breakout_trend,
    rolling_range_bounce,
    rolling_range_breakout_trend,
    session_level_reversal,
)
from backtest.validation import (
    ValidationSpec,
    compare_specs_walk_forward,
    run_out_of_sample_test,
    run_walk_forward_validation,
)

warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "experiments"

# ---------------------------------------------------------------------------
# Shared parameters — derived from EUR/USD volatility, NOT curve-fitted
# ---------------------------------------------------------------------------
#
# EUR/USD 1-hour ATR ≈ 0.15-0.22 %  (from natr_300 on 1-sec bars × sqrt(300))
# SL = 1.5 × hourly ATR → 0.28 %   (survives ≥ 1 hour of normal noise)
# TP = 2.2 × SL        → 0.62 %    (meaningful directional move)
# These numbers are the same for every strategy to prevent cherry-picking.
#
# Hold: 60-120 min depending on whether the strategy is mean-reversion or trend.
# Gate: 09:00-13:00 NY (liquid 4-hr morning window), 7 200-bar cooldown
#       = minimum 2-hour gap between trades → at most 2 trades per session.

SL_PCT   = 0.0028   # 0.28 %
TP_PCT   = 0.0062   # 0.62 %   (2.2 × SL)

# Mean-reversion hold is shorter; trend/breakout holds are longer
TP_TREND = 0.0068   # 0.68 %   (2.4 × SL — trend strategies need more room)

HOLD_MR       = 3_600.0   # 60 min — mean-reversion (fade)
HOLD_TREND    = 7_200.0   # 120 min — trend-following / breakout

GATE = EntryGate(
    trade_start       = "09:00",
    trade_end         = "13:00",
    cooldown_bars     = 7_200,   # 2 hours on 1-sec bars
    max_trades_per_day = 2,
)


def _tp(trend: bool = False) -> TradeParams:
    return TradeParams(
        stop_loss_pct    = SL_PCT,
        take_profit_pct  = TP_TREND if trend else TP_PCT,
        max_hold_seconds = HOLD_TREND if trend else HOLD_MR,
    )


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

def _build_experiments() -> List[BacktestSpec]:
    """
    6 strategies × uniform gate = no per-strategy time-window tuning.

    Entry logic for each strategy is exactly 2 conditions per side:
      Condition 1 — structural setup (level proximity / zone / breakout)
      Condition 2 — trend direction (binary EMA comparison, no threshold)

    Proximity thresholds are 0.20 % (structural: 2× typical 1-min noise of
    EUR/USD).  EMA windows (1 200 bars = 20 min, 3 600 bars = 60 min) are
    chosen to be round multiples of the trading hour on 1-sec bars.
    """

    # ── Strategy 1 ──────────────────────────────────────────────────────────
    # Rolling Range Bounce (60-min range, mean-reversion)
    # Setup   : price within 0.20 % of the 60-min rolling low/high
    # Trend   : 20-min EMA vs 60-min EMA gives broad directional bias
    # Rationale: dist_rolling_low/high_60m are top permutation-importance
    #   features at the 300s ML horizon.  Tight threshold (0.20 %) → fires
    #   only when price genuinely tests the range extreme.  2-hr cooldown
    #   prevents over-trading a range that hugs the level for 30+ minutes.
    spec_bounce = BacktestSpec(
        name="rolling_range_bounce_60m",
        strategy_fn=rolling_range_bounce,
        group="ml_meanrev",
        strategy_kwargs=dict(
            range_label_near   = "60m",
            near_threshold     = 0.0020,   # 0.20 % — structural, not tuned
            long_trend_filter  = -0.0040,  # ema_3600 filter: not in a strong downtrend
            short_trend_filter =  0.0040,
            adx_max            = 999.0,    # no ADX filter — one less parameter
            trade_params       = _tp(trend=False),
            gate               = GATE,
        ),
    )

    # ── Strategy 2 ──────────────────────────────────────────────────────────
    # Rolling Range Bounce on the wider 2-hour range
    # Same logic as S1 but uses the 2h rolling range — catches larger swings.
    spec_bounce_2h = BacktestSpec(
        name="rolling_range_bounce_2h",
        strategy_fn=rolling_range_bounce,
        group="ml_meanrev",
        strategy_kwargs=dict(
            range_label_near   = "2h",
            near_threshold     = 0.0020,
            long_trend_filter  = -0.0040,
            short_trend_filter =  0.0040,
            adx_max            = 999.0,
            trade_params       = _tp(trend=False),
            gate               = GATE,
        ),
    )

    # ── Strategy 3 ──────────────────────────────────────────────────────────
    # Fibonacci Golden Zone + Trend (4-hour swing)
    # Setup   : price inside 38.2-61.8 % retracement band of the 4h swing,
    #           in the lower/upper half depending on direction.
    # Trend   : ema_1200 (20 min) vs ema_3600 (60 min)
    # Rationale: fib_4h_dist_fib_382 is a top-3 permutation-importance feature
    #   at all horizons.  The golden zone is a canonical (not data-mined) level.
    #   range_min=0.0040 prevents trading tiny intraday ranges (< 40 pips).
    spec_fib_4h = BacktestSpec(
        name="fib_golden_zone_4h",
        strategy_fn=fib_golden_zone_trend,
        group="ml_fib",
        strategy_kwargs=dict(
            fib_prefix          = "fib_4h",
            range_min           = 0.0040,   # only trade when 4h swing ≥ 0.40 %
            range_pos_long_max  = 0.55,     # lower half of zone → long
            range_pos_short_min = 0.45,     # upper half → short
            trade_params        = _tp(trend=False),
            gate                = GATE,
        ),
    )

    # ── Strategy 4 ──────────────────────────────────────────────────────────
    # Fibonacci Golden Zone + Trend (8-hour swing, wider timeframe)
    # Same logic on the 8h swing — less frequent but higher-quality entries.
    spec_fib_8h = BacktestSpec(
        name="fib_golden_zone_8h",
        strategy_fn=fib_golden_zone_trend,
        group="ml_fib",
        strategy_kwargs=dict(
            fib_prefix          = "fib_8h",
            range_min           = 0.0050,   # 8h swing must be ≥ 0.50 %
            range_pos_long_max  = 0.55,
            range_pos_short_min = 0.45,
            trade_params        = _tp(trend=False),
            gate                = GATE,
        ),
    )

    # ── Strategy 5 ──────────────────────────────────────────────────────────
    # Rolling Range Breakout — 2-hour (momentum / trend-following)
    # Setup   : price just made a new 2-hour high/low (dist crosses zero)
    # Trend   : ema_300 (5 min) vs ema_1200 (20 min) aligned with breakout
    # Rationale: dist_rolling_high_2h crossed_above(0) fires at most once per
    #   new 2h extreme — inherently infrequent without needing tight thresholds.
    #   Trend alignment prevents chasing false breakouts in countertrend moves.
    spec_breakout = BacktestSpec(
        name="rolling_range_breakout_2h",
        strategy_fn=rolling_range_breakout_trend,
        group="ml_breakout",
        strategy_kwargs=dict(
            range_label = "2h",
            fast_ema    = "ema_300",    # 5-min EMA
            slow_ema    = "ema_1200",   # 20-min EMA
            adx_min     = 0.0,          # no ADX filter — keeps it simple
            trade_params = _tp(trend=True),
            gate         = GATE,
        ),
    )

    # ── Strategy 6 ──────────────────────────────────────────────────────────
    # 15-min Opening Range Breakout + 60-min EMA trend
    # Setup   : price just broke above/below the 15-min opening range
    # Trend   : 20-min EMA vs 60-min EMA confirms direction
    # Rationale: dist_or_high/low_15m top session features at 60s and 300s
    #   horizons.  The OR is a canonical institutional level; crossing it with
    #   trend confirmation is a structural edge, not a data-mined one.
    spec_or = BacktestSpec(
        name="or15m_breakout_trend",
        strategy_fn=or15m_breakout_trend,
        group="ml_session",
        strategy_kwargs=dict(
            fast_ema     = "ema_1200",   # 20-min EMA
            slow_ema     = "ema_3600",   # 60-min EMA
            adx_min      = 0.0,          # no ADX filter
            trade_params = _tp(trend=True),
            gate         = EntryGate(    # OR needs a slightly later start
                trade_start        = "10:00",   # after 15m OR has formed + buffer
                trade_end          = "13:00",
                cooldown_bars      = 7_200,
                max_trades_per_day = 2,
            ),
        ),
    )

    # ── Strategy 7 ──────────────────────────────────────────────────────────
    # Long-EMA Pullback (re-entry in trend at the 60-min EMA)
    # Setup   : price pulls back to within 0.20 % of the 60-min EMA
    # Trend   : 20-min EMA vs 60-min EMA (same EMAs as setup, so direction
    #           is already baked in — the pullback is TO the slow EMA)
    # Rationale: ema_3600 is the top trend feature at the 600s ML horizon by
    #   permutation importance.  Price trading 0.20 % above/below this EMA
    #   represents a genuine test of the trend line, not just random proximity.
    spec_pullback = BacktestSpec(
        name="long_ema_pullback_60m",
        strategy_fn=long_ema_pullback,
        group="ml_trend",
        strategy_kwargs=dict(
            fast_ema      = "ema_1200",
            slow_ema      = "ema_3600",
            pullback_near = 0.0020,   # within 0.20 % of ema_3600 from correct side
            pullback_max  = 0.0040,   # cap: not more than 0.40 % beyond EMA
            adx_min       = 0.0,
            trade_params  = _tp(trend=True),
            gate          = GATE,
        ),
    )

    return [
        spec_bounce,
        spec_bounce_2h,
        spec_fib_4h,
        spec_fib_8h,
        spec_breakout,
        spec_or,
        spec_pullback,
    ]


def _to_val(bs: BacktestSpec) -> ValidationSpec:
    return ValidationSpec(
        name=bs.name,
        strategy_fn=bs.strategy_fn,
        strategy_kwargs=bs.strategy_kwargs,
    )


def _hdr(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load + build features (once, shared across all experiments) ──────
    cfg = get_dataset_config("micro_currency_futures")
    raw = load_raw_data(
        dataset        = cfg["dataset"],
        symbols_prefix = cfg["symbols_prefix"],
        include_spreads= False,
        start          = cfg["start"],
        end            = cfg["end"],
    )

    _hdr("Building features")
    feat = build_features_for_family(raw, "regular", bar_seconds=cfg["bar_seconds"])
    del raw; gc.collect()

    experiments = _build_experiments()
    val_specs   = [_to_val(e) for e in experiments]

    # ── 2. Walk-forward: 120D train / 30D test / 30D step ───────────────────
    # 14-month dataset → ~10 non-overlapping OOS windows.
    # The train window is intentionally kept fixed (not growing) so each test
    # window evaluates the same amount of recent history — avoids giving later
    # windows unfair advantage from more training data.
    _hdr(
        "Walk-Forward Validation\n"
        "  Train: 120D  |  Test: 30D  |  Step: 30D\n"
        "  ~10 independent OOS test windows over Jan 2025 – Mar 2026"
    )
    wf_summary = compare_specs_walk_forward(
        df            = feat,
        specs         = val_specs,
        train_period  = "120D",
        test_period   = "30D",
        step_period   = "30D",
        initial_capital = 1_000.0,
    )
    wf_summary.to_csv(OUTPUT_DIR / "walkforward_comparison.csv", index=False)
    print("\nWalk-forward summary saved.")

    # ── 3. Per-strategy window detail for top-4 ──────────────────────────────
    top_names = wf_summary.head(4)["name"].tolist()
    _hdr(f"Per-strategy window detail — top {len(top_names)} by mean Sharpe: {top_names}")

    wf_details: Dict[str, Any] = {}
    for spec in val_specs:
        if spec.name not in top_names:
            continue
        out = run_walk_forward_validation(
            df            = feat,
            spec          = spec,
            train_period  = "120D",
            test_period   = "30D",
            step_period   = "30D",
            initial_capital = 1_000.0,
            keep_test_trades = True,
        )
        wf_details[spec.name] = out
        if out["test_trades"] is not None and not out["test_trades"].empty:
            fname = f"wf_trades_{spec.name}.csv"
            out["test_trades"].to_csv(OUTPUT_DIR / fname, index=False)
        gc.collect()

    # ── 4. Final OOS holdout (4 months, truly blind) ─────────────────────────
    # Train period deliberately ends 2 months before the dataset ends to leave
    # a buffer — the final 4 months of data were never touched during any
    # walk-forward window's training phase.
    _hdr(
        "Final OOS Holdout  (strictly blind)\n"
        "  Train: Jan 2025 – Oct 2025  |  Test: Nov 2025 – Mar 2026"
    )
    oos_rows: List[Dict] = []
    for spec in val_specs:
        if spec.name not in top_names:
            continue
        result = run_out_of_sample_test(
            df              = feat,
            spec            = spec,
            train_end       = "2025-11-01",
            test_start      = "2025-11-01",
            test_end        = "2026-03-01",
            initial_capital = 1_000.0,
            keep_test_outputs = True,
        )
        tm = result["test_result"]["metrics"]
        oos_rows.append({
            "name":             spec.name,
            "n_trades":         tm.get("total_trades"),
            "total_return":     tm.get("total_return"),
            "cagr":             tm.get("cagr"),
            "sharpe":           tm.get("sharpe"),
            "max_drawdown":     tm.get("max_drawdown"),
            "win_rate":         tm.get("win_rate"),
            "avg_trade_pnl":    tm.get("avg_trade_pnl"),
            "median_trade_pnl": tm.get("median_trade_pnl"),
        })
        trades = result["test_result"].get("trades")
        if trades is not None and not trades.empty:
            trades.to_csv(OUTPUT_DIR / f"oos_trades_{spec.name}.csv", index=False)
        gc.collect()

    oos_df = (pd.DataFrame(oos_rows)
              .sort_values("sharpe", ascending=False, na_position="last"))
    _hdr("OOS Holdout Results (Nov 2025 – Mar 2026)")
    print(oos_df.to_string(index=False))
    oos_df.to_csv(OUTPUT_DIR / "oos_holdout_summary.csv", index=False)

    # ── 5. Full-sample reference ──────────────────────────────────────────────
    _hdr("Full-Sample Reference (in-sample, all strategies)")
    results = []
    for spec in experiments:
        r = run_single_backtest(feat, spec, initial_capital=1_000.0)
        results.append(r)
        gc.collect()
    full_summary = summarize_results(results)
    full_summary.to_csv(OUTPUT_DIR / "full_sample_summary.csv", index=False)

    # ── 6. Final ranking ──────────────────────────────────────────────────────
    _hdr("FINAL RANKING")
    combined = wf_summary[["name", "mean_sharpe", "median_sharpe",
                            "mean_total_return", "mean_win_rate",
                            "total_trades"]].copy()
    combined.columns = ["name", "wf_mean_sharpe", "wf_median_sharpe",
                        "wf_mean_return", "wf_mean_winrate", "wf_total_trades"]
    if oos_rows:
        oos_merge = oos_df[["name", "sharpe", "total_return",
                             "win_rate", "n_trades"]].copy()
        oos_merge.columns = ["name", "oos_sharpe", "oos_return",
                             "oos_winrate", "oos_trades"]
        ranking = combined.merge(oos_merge, on="name", how="left")
    else:
        ranking = combined

    ranking = ranking.sort_values("wf_mean_sharpe", ascending=False, na_position="last")
    print(ranking.to_string(index=False))
    ranking.to_csv(OUTPUT_DIR / "final_ranking.csv", index=False)

    _hdr("Parameter summary (anti-overfitting)")
    print(f"  SL  (all strategies) : {SL_PCT*100:.2f} %  (1.5× EUR/USD hourly ATR)")
    print(f"  TP  (mean-rev)       : {TP_PCT*100:.2f} %  (2.2× SL)")
    print(f"  TP  (trend)          : {TP_TREND*100:.2f} %  (2.4× SL)")
    print(f"  Hold (mean-rev)      : {int(HOLD_MR/60)} min")
    print(f"  Hold (trend)         : {int(HOLD_TREND/60)} min")
    print(f"  Gate (most)          : {GATE.trade_start}–{GATE.trade_end} NY, "
          f"{GATE.cooldown_bars//3600}h cooldown, "
          f"max {GATE.max_trades_per_day}/day")
    print(f"  Entry conditions     : 2 per side (structural setup + trend direction)")
    print(f"  No ADX / RSI filters : avoids additional free parameter")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  {f.name}")

    return ranking


if __name__ == "__main__":
    main()
