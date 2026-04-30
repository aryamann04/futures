"""
ML-based feature importance analysis for Micro EUR/USD Futures (M6E).

Trains LightGBM classifiers and regressors to predict forward returns at
multiple horizons (60s, 300s, 600s), then ranks features by gain-based and
permutation importance.  Aggregates results by feature category and timeframe
window to surface which types of signals and lookback periods drive returns.

Also backtests a simple ML-signal strategy on the held-out OOS period and
compares it against a basic buy-and-hold baseline.

Memory safety:
  - Caps process address space at 6 GB via resource.setrlimit.
  - Stride-samples the training set so the feature matrix stays ≤ MAX_SAMPLE_ROWS.
  - Downcasts all numeric columns to float32 immediately after building features.
  - Deletes every large intermediate object with del + gc.collect().

Usage:
    cd /Users/aryaman/futures
    DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python3 -m features.ml_analysis
"""

from __future__ import annotations

import gc
import re
import resource
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from backtest.engine import run_backtest
from backtest.metrics import compute_basic_metrics
from data.load import micro_futures_data
from features.build_features import add_targets, build_features
from features.research import _detect_feature_columns, _filter_raw_data, _pick_column

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---------------------------------------------------------------------------
# Memory cap  (6 GB — adjust to ~70 % of your actual RAM)
# ---------------------------------------------------------------------------
_GB = 1024 ** 3
try:
    resource.setrlimit(resource.RLIMIT_AS, (6 * _GB, 6 * _GB))
    print("Memory cap set to 6 GB.")
except Exception as e:
    print(f"Warning: could not set memory cap ({e}).")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_DIR  = Path(__file__).resolve().parent / "outputs" / "ml_analysis"
MAX_SAMPLE_ROWS = 350_000    # max rows used for ML training (stride-sampled from train set)
TRAIN_FRAC      = 0.80       # first 80 % → train, last 20 % → OOS test
OOS_COOLDOWN_BARS   = 300    # min 1-sec bars between ML trades (= 5 min)
OOS_MAX_PER_DAY     = 3
OOS_SL_PCT          = 0.0025
OOS_TP_PCT          = 0.0048
OOS_MAX_HOLD_SEC    = 300.0
LONG_THRESHOLD      = 0.55   # classifier prob threshold for long entry
SHORT_THRESHOLD     = 0.45   # classifier prob threshold for short entry

# ---------------------------------------------------------------------------
# Feature category tagging
# ---------------------------------------------------------------------------
_CATEGORY_RULES: List[Tuple[str, List[str]]] = [
    ("basic_returns",  ["ret_", "logret_", "hl_spread", "oc_spread"]),
    ("trend",          ["ema_", "sma_", "price_vs_ema", "ema_slope_", "macd",
                        "ema100_gt", "ema100_lt", "ema300_gt", "ema300_lt",
                        "ema100_300", "ema300_600"]),
    ("momentum",       ["rsi_", "mom_", "roc_", "stoch_", "willr_"]),
    ("volatility",     ["atr_", "natr_", "bb_", "adx_", "_atr_norm", "_atr_dist"]),
    ("volume",         ["obv", "vol_sma_", "rel_volume_", "volume_spike_"]),
    ("session",        ["hour_utc", "minute_of_day", "is_london_ny", "is_us_morning",
                        "prev_session_", "dist_prev_session_",
                        "opening_range_", "dist_or_", "date_utc"]),
    ("rolling_ranges", ["rolling_high_", "rolling_low_", "dist_rolling_",
                        "_range_pos", "trend_range_"]),
    ("fibonacci",      ["fib_", "dist_fib_", "in_fib_zone_"]),
]


def _tag_category(feature: str) -> str:
    f = feature.lower()
    for category, patterns in _CATEGORY_RULES:
        for p in patterns:
            if p in f:
                return category
    return "other"


def _extract_window(feature: str) -> Optional[int]:
    """Pull trailing numeric window, e.g. ema_300 → 300."""
    m = re.search(r"_(\d+)(?:s|m)?$", feature)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)$", feature)
    if m:
        return int(m.group(1))
    return None


def _window_group(w: Optional[int]) -> str:
    if w is None:
        return "no_window"
    if w <= 20:
        return "short (<=20s)"
    if w <= 100:
        return "medium (21-100s)"
    if w <= 600:
        return "long (101-600s)"
    return "very_long (>600s)"


# ---------------------------------------------------------------------------
# Downcast helpers
# ---------------------------------------------------------------------------
def _downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all float columns to float32, int to int32. In-place."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_and_build(
    symbols_prefix: str,
    start: str,
    end: str,
    bar_seconds: int = 1,
    active_hours_utc: Tuple[int, int] = (7, 17),
) -> pd.DataFrame:
    """
    Load raw OHLCV → build full feature set → add targets → filter trading hours.
    Returns a DataFrame with both features and OHLCV (needed for OOS backtest).
    All floats are downcast to float32.
    """
    print("\n[1/5] Loading raw data ...")
    raw = micro_futures_data(
        columns=["ts_event", "symbol", "open", "high", "low", "close", "volume"]
    )
    raw = _filter_raw_data(
        raw,
        symbols_prefix=symbols_prefix,
        include_spreads=False,
        start=start,
        end=end,
    )
    print(f"      Rows after date/symbol filter : {len(raw):,}")

    if raw.empty:
        raise ValueError("No data remaining after symbol/date filters.")

    print("[2/5] Building features ...")
    df = build_features(
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
    del raw
    gc.collect()

    df = _downcast_df(df)
    df.columns = [c.lower() for c in df.columns]
    print(f"      Shape after feature build     : {df.shape}")

    # Apply trading-hour filter AFTER building features so rolling windows
    # at the edges of each session don't blow up.
    ts_col = _pick_column(df.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    hour = df[ts_col].dt.hour
    df = df[(hour >= active_hours_utc[0]) & (hour < active_hours_utc[1])].reset_index(drop=True)
    print(f"      Rows after trading-hour filter: {len(df):,}")

    return df


# ---------------------------------------------------------------------------
# Feature matrix preparation
# ---------------------------------------------------------------------------
def _prepare_matrix(
    df: pd.DataFrame,
    horizons: List[int],
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Add forward-return targets; keep only feature + target columns.
    Only drops rows where *targets* are NaN — LightGBM handles feature NaN natively.
    """
    print("[3/5] Adding targets and preparing ML matrix ...")
    df = add_targets(df, horizons=horizons)
    df = _downcast_df(df)
    df.columns = [c.lower() for c in df.columns]

    target_ret_cols = [f"target_fwd_ret_{h}s" for h in horizons if f"target_fwd_ret_{h}s" in df.columns]
    dir_dict: dict = {}
    target_dir_cols: List[str] = []
    for h in horizons:
        rc = f"target_fwd_ret_{h}s"
        dc = f"target_dir_{h}s"
        if rc in df.columns:
            dir_dict[dc] = (df[rc] > 0).astype("int8")
            target_dir_cols.append(dc)

    df = pd.concat([df, pd.DataFrame(dir_dict, index=df.index)], axis=1)

    keep = [c for c in (feature_cols + target_ret_cols + target_dir_cols) if c in df.columns]
    df = df[keep].copy()

    before = len(df)
    df = df.dropna(subset=target_ret_cols + target_dir_cols)
    print(f"      Rows with valid targets        : {len(df):,}  (dropped {before - len(df):,} NaN-target rows)")
    return df


# ---------------------------------------------------------------------------
# Train / OOS split
# ---------------------------------------------------------------------------
def _time_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC):
    split = int(len(df) * train_frac)
    return df.iloc[:split], df.iloc[split:]


def _stride_sample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Return a stride-sampled subset ≤ max_rows, preserving time order."""
    if len(df) <= max_rows:
        return df
    step = max(1, len(df) // max_rows)
    return df.iloc[::step].head(max_rows).copy()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
_LGB_BASE = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_samples=50,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


def _norm(arr: np.ndarray) -> np.ndarray:
    t = arr.sum()
    return arr / t if t > 0 else arr


def _train_horizon(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    n_perm_repeats: int = 10,
) -> dict:
    """Train classifier + regressor for one horizon, return importance arrays."""
    dir_col = f"target_dir_{horizon}s"
    ret_col = f"target_fwd_ret_{horizon}s"

    # stride-sample training set
    tr = _stride_sample(train_df, MAX_SAMPLE_ROWS)
    te = test_df  # keep full OOS set for evaluation

    X_tr  = tr[feature_cols].values.astype("float32")
    y_d_tr = tr[dir_col].values.astype("int32")
    y_r_tr = tr[ret_col].values.astype("float32")

    X_te  = te[feature_cols].values.astype("float32")
    y_d_te = te[dir_col].values.astype("int32")
    y_r_te = te[ret_col].values.astype("float32")

    print(f"\n  Train sample: {len(tr):,} rows  |  OOS test: {len(te):,} rows")

    # ---- Classifier ----
    print(f"  Training classifier  (horizon={horizon}s) ...")
    clf = lgb.LGBMClassifier(**_LGB_BASE)
    clf.fit(X_tr, y_d_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    auc  = roc_auc_score(y_d_te, prob)
    acc  = accuracy_score(y_d_te, (prob >= 0.5).astype(int))
    ll   = log_loss(y_d_te, prob)
    print(f"    Classifier  AUC={auc:.4f}  Acc={acc:.4f}  LogLoss={ll:.4f}")
    clf_imp = _norm(clf.feature_importances_.astype(float))

    # ---- Regressor ----
    print(f"  Training regressor   (horizon={horizon}s) ...")
    reg = lgb.LGBMRegressor(objective="regression_l1", **_LGB_BASE)
    reg.fit(X_tr, y_r_tr)
    pred_r  = reg.predict(X_te).astype("float32")
    mae     = float(np.mean(np.abs(pred_r - y_r_te)))
    dir_acc = float(np.mean(np.sign(pred_r) == np.sign(y_r_te)))
    print(f"    Regressor   MAE={mae:.6f}  DirAcc={dir_acc:.4f}")
    reg_imp = _norm(reg.feature_importances_.astype(float))

    combined_imp = (clf_imp + reg_imp) / 2.0

    # ---- Permutation importance ----
    # Use n_jobs=1 to avoid forking large arrays; sub-sample test set if large
    perm_te_X = X_te[:50_000] if len(X_te) > 50_000 else X_te
    perm_te_y = y_d_te[:50_000] if len(y_d_te) > 50_000 else y_d_te
    print(f"  Permutation importance (n_repeats={n_perm_repeats}, rows={len(perm_te_X):,}) ...")
    perm = permutation_importance(
        clf, perm_te_X, perm_te_y,
        n_repeats=n_perm_repeats,
        random_state=42,
        scoring="roc_auc",
        n_jobs=1,
    )
    perm_imp = _norm(np.maximum(perm.importances_mean, 0))

    # Keep the 300s classifier for OOS signal generation
    trained_clf = clf if horizon == 300 else None

    del clf, reg, X_tr, X_te, y_d_tr, y_r_tr, pred_r
    gc.collect()

    return {
        "horizon":             horizon,
        "clf_auc":             auc,
        "clf_acc":             acc,
        "clf_logloss":         ll,
        "reg_mae":             mae,
        "reg_dir_acc":         dir_acc,
        "clf_gain_importance": clf_imp,
        "reg_gain_importance": reg_imp,
        "combined_gain":       combined_imp,
        "perm_importance":     perm_imp,
        "trained_clf":         trained_clf,  # only non-None for horizon==300
    }


# ---------------------------------------------------------------------------
# OOS strategy backtest
# ---------------------------------------------------------------------------
def _apply_cooldown(
    signals: np.ndarray,
    cooldown_bars: int = OOS_COOLDOWN_BARS,
    max_per_day: int   = OOS_MAX_PER_DAY,
    dates: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Zero out signals that violate cooldown or daily trade limit."""
    out = signals.copy()
    last_bar  = -cooldown_bars
    day_counts: dict = {}

    for i in range(len(out)):
        if out[i] == 0:
            continue
        if i - last_bar < cooldown_bars:
            out[i] = 0
            continue
        if dates is not None:
            d = dates[i]
            if day_counts.get(d, 0) >= max_per_day:
                out[i] = 0
                continue
            day_counts[d] = day_counts.get(d, 0) + 1
        last_bar = i

    return out


def _build_ml_plan(
    oos_full_df: pd.DataFrame,
    clf,
    feature_cols: List[str],
    horizon: int = 300,
) -> pd.DataFrame:
    """
    Apply the trained classifier to the full OOS period and build a backtest plan.
    Signals: +1 (long) when prob > LONG_THRESHOLD, -1 (short) when prob < SHORT_THRESHOLD.
    """
    ts_col  = _pick_column(oos_full_df.columns, ["ts_event", "timestamp", "datetime"])
    sym_col = _pick_column(oos_full_df.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"])

    available_feats = [f for f in feature_cols if f in oos_full_df.columns]
    X_oos = oos_full_df[available_feats].values.astype("float32")

    print(f"  Generating ML signals on OOS period ({len(X_oos):,} bars) ...")
    prob = clf.predict_proba(X_oos)[:, 1].astype("float32")

    raw_signal = np.where(prob > LONG_THRESHOLD, 1,
                 np.where(prob < SHORT_THRESHOLD, -1, 0)).astype("int8")

    dates = oos_full_df[ts_col].dt.date.values if ts_col in oos_full_df.columns else None
    signal = _apply_cooldown(raw_signal, dates=dates)

    plan = oos_full_df[[ts_col, sym_col, "open", "high", "low", "close"]].copy()
    plan["entry_signal"]    = signal
    plan["stop_loss_pct"]   = OOS_SL_PCT
    plan["take_profit_pct"] = OOS_TP_PCT
    plan["max_hold_seconds"] = OOS_MAX_HOLD_SEC

    n_signals = int((signal != 0).sum())
    print(f"  Signals generated: {n_signals:,}  "
          f"(long={int((signal == 1).sum()):,}, short={int((signal == -1).sum()):,})")
    return plan


def _run_oos_backtest(
    oos_full_df: pd.DataFrame,
    clf,
    feature_cols: List[str],
    initial_capital: float = 1_000.0,
) -> dict:
    """Generate ML signal on OOS period → run through engine → compute metrics."""
    print("\n[5/5] OOS Strategy Backtest ...")
    plan = _build_ml_plan(oos_full_df, clf, feature_cols)

    _, trades = run_backtest(plan)
    n_trades = len(trades)
    print(f"  OOS trades executed: {n_trades:,}")

    if n_trades == 0:
        print("  No trades — thresholds too tight or no signal on OOS period.")
        return {"n_trades": 0}

    metrics = compute_basic_metrics(trades, initial_capital=initial_capital)
    return {"n_trades": n_trades, "metrics": metrics, "trades": trades}


# ---------------------------------------------------------------------------
# Importance aggregation helpers
# ---------------------------------------------------------------------------
def _imp_df(feature_cols: List[str], results: List[dict], key: str, label: str) -> pd.DataFrame:
    rows = []
    for r in results:
        imp = r[key]
        for feat, score in zip(feature_cols, imp):
            rows.append({
                "horizon_s":      r["horizon"],
                "feature":        feat,
                "category":       _tag_category(feat),
                "window":         _extract_window(feat),
                "window_group":   _window_group(_extract_window(feat)),
                "importance":     float(score),
                "importance_type": label,
            })
    return pd.DataFrame(rows)


def _cat_summary(imp: pd.DataFrame) -> pd.DataFrame:
    return (
        imp.groupby(["importance_type", "horizon_s", "category"], as_index=False)["importance"]
        .sum()
        .sort_values(["importance_type", "horizon_s", "importance"], ascending=[True, True, False])
    )


def _win_summary(imp: pd.DataFrame) -> pd.DataFrame:
    sub = imp[imp["window"].notna()].copy()
    return (
        sub.groupby(["importance_type", "horizon_s", "window_group"], as_index=False)["importance"]
        .sum()
        .sort_values(["importance_type", "horizon_s", "importance"], ascending=[True, True, False])
    )


def _hdr(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _top(df: pd.DataFrame, horizon: int, imp_type: str, n: int = 30):
    sub = df[(df["horizon_s"] == horizon) & (df["importance_type"] == imp_type)]
    print(sub.sort_values("importance", ascending=False).head(n)
            [["feature", "category", "window_group", "importance"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_ml_analysis(
    symbols_prefix:    str              = "M6E",
    start:             str              = "2025-03-01",
    end:               str              = "2026-03-01",
    horizons:          Iterable[int]    = (60, 300, 600),
    bar_seconds:       int              = 1,
    active_hours_utc:  Tuple[int, int]  = (7, 17),
    train_frac:        float            = TRAIN_FRAC,
    n_perm_repeats:    int              = 10,
    top_n:             int              = 30,
    initial_capital:   float            = 1_000.0,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    horizons = list(horizons)

    # ------------------------------------------------------------------
    # 1. Load raw data + build features (full year, then hour-filtered)
    # ------------------------------------------------------------------
    full_df = _load_and_build(
        symbols_prefix=symbols_prefix,
        start=start, end=end,
        bar_seconds=bar_seconds,
        active_hours_utc=active_hours_utc,
    )

    sym_col = _pick_column(full_df.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"])
    ts_col  = _pick_column(full_df.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    feature_cols = _detect_feature_columns(full_df, sym_col, ts_col)
    print(f"\n      Feature columns detected       : {len(feature_cols)}")

    # ------------------------------------------------------------------
    # 2. Time-split BEFORE adding targets (avoids target leakage between splits)
    # ------------------------------------------------------------------
    split_idx  = int(len(full_df) * train_frac)
    train_full = full_df.iloc[:split_idx].copy()   # has OHLCV + features (no targets yet)
    oos_full   = full_df.iloc[split_idx:].copy()   # kept in full for OOS backtest
    del full_df; gc.collect()

    print(f"      Train period rows              : {len(train_full):,}")
    print(f"      OOS   period rows              : {len(oos_full):,}")

    # ------------------------------------------------------------------
    # 3. Build ML matrix (add targets, keep feature+target cols only)
    # ------------------------------------------------------------------
    # Train matrix
    train_matrix = _prepare_matrix(train_full.copy(), horizons, feature_cols)
    feature_cols  = [c for c in feature_cols if c in train_matrix.columns]

    # OOS matrix (targets computed on OOS OHLCV, no leakage from train)
    oos_matrix = _prepare_matrix(oos_full.copy(), horizons, feature_cols)

    print(f"      Train matrix shape             : {train_matrix.shape}")
    print(f"      OOS   matrix shape             : {oos_matrix.shape}")
    print(f"      Final feature count            : {len(feature_cols)}")

    # ------------------------------------------------------------------
    # 4. Train models per horizon
    # ------------------------------------------------------------------
    print("\n[4/5] Training LightGBM models ...")
    all_results   = []
    clf_300       = None   # save 300s classifier for OOS backtest

    for h in horizons:
        _hdr(f"HORIZON: {h}s  ({h // 60}min {h % 60}s)")
        result = _train_horizon(
            train_df=train_matrix,
            test_df=oos_matrix,
            feature_cols=feature_cols,
            horizon=h,
            n_perm_repeats=n_perm_repeats,
        )
        if result["trained_clf"] is not None:
            clf_300 = result["trained_clf"]
        # Drop the classifier object from the result dict to save memory
        result["trained_clf"] = None
        all_results.append(result)
        gc.collect()

    # ------------------------------------------------------------------
    # 5. OOS strategy backtest using 300s classifier
    # ------------------------------------------------------------------
    oos_result: dict = {}
    if clf_300 is not None and not oos_full.empty:
        oos_result = _run_oos_backtest(
            oos_full_df=oos_full,
            clf=clf_300,
            feature_cols=feature_cols,
            initial_capital=initial_capital,
        )
        del clf_300
        gc.collect()

    del train_matrix, oos_matrix, train_full
    gc.collect()

    # ------------------------------------------------------------------
    # 6. Build importance DataFrames
    # ------------------------------------------------------------------
    gain_df = _imp_df(feature_cols, all_results, "combined_gain",   "gain")
    perm_df = _imp_df(feature_cols, all_results, "perm_importance", "permutation")
    all_imp = pd.concat([gain_df, perm_df], axis=0, ignore_index=True)

    cat_sum = _cat_summary(all_imp)
    win_sum = _win_summary(all_imp)

    metrics_rows = [{
        "horizon_s":   r["horizon"],
        "clf_auc":     r["clf_auc"],
        "clf_acc":     r["clf_acc"],
        "clf_logloss": r["clf_logloss"],
        "reg_mae":     r["reg_mae"],
        "reg_dir_acc": r["reg_dir_acc"],
    } for r in all_results]
    metrics_df = pd.DataFrame(metrics_rows)

    # ------------------------------------------------------------------
    # 7. Print results
    # ------------------------------------------------------------------
    _hdr("MODEL METRICS")
    print(metrics_df.to_string(index=False))

    for h in horizons:
        _hdr(f"TOP {top_n} FEATURES BY GAIN IMPORTANCE  |  horizon={h}s")
        _top(all_imp, h, "gain", top_n)
        _hdr(f"TOP {top_n} FEATURES BY PERMUTATION IMPORTANCE  |  horizon={h}s")
        _top(all_imp, h, "permutation", top_n)

    _hdr("CATEGORY IMPORTANCE SUMMARY — GAIN")
    sub = cat_sum[cat_sum["importance_type"] == "gain"]
    print(sub[["horizon_s", "category", "importance"]].to_string(index=False))

    _hdr("CATEGORY IMPORTANCE SUMMARY — PERMUTATION")
    sub = cat_sum[cat_sum["importance_type"] == "permutation"]
    print(sub[["horizon_s", "category", "importance"]].to_string(index=False))

    _hdr("TIMEFRAME WINDOW IMPORTANCE — GAIN")
    sub = win_sum[win_sum["importance_type"] == "gain"]
    print(sub[["horizon_s", "window_group", "importance"]].to_string(index=False))

    _hdr("TIMEFRAME WINDOW IMPORTANCE — PERMUTATION")
    sub = win_sum[win_sum["importance_type"] == "permutation"]
    print(sub[["horizon_s", "window_group", "importance"]].to_string(index=False))

    _hdr("OOS STRATEGY BACKTEST  (300s horizon ML signal)")
    if not oos_result or oos_result.get("n_trades", 0) == 0:
        print("No OOS trades — check LONG_THRESHOLD / SHORT_THRESHOLD settings.")
    else:
        m = oos_result.get("metrics", {})
        print(f"  OOS trades        : {oos_result['n_trades']:,}")
        for k in ["total_return", "cagr", "sharpe", "max_drawdown",
                  "win_rate", "avg_trade_pnl", "median_trade_pnl"]:
            if k in m:
                print(f"  {k:<22}: {m[k]}")

    # ------------------------------------------------------------------
    # 8. Save CSVs
    # ------------------------------------------------------------------
    gain_out = gain_df.sort_values(["horizon_s", "importance"], ascending=[True, False])
    perm_out = perm_df.sort_values(["horizon_s", "importance"], ascending=[True, False])

    gain_out.to_csv(OUTPUT_DIR / "feature_importance_gain.csv",        index=False)
    perm_out.to_csv(OUTPUT_DIR / "feature_importance_permutation.csv", index=False)
    cat_sum.to_csv( OUTPUT_DIR / "category_importance_summary.csv",    index=False)
    win_sum.to_csv( OUTPUT_DIR / "timeframe_importance_summary.csv",   index=False)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv",                index=False)

    if oos_result.get("n_trades", 0) > 0 and "trades" in oos_result:
        oos_result["trades"].to_csv(OUTPUT_DIR / "oos_trades.csv", index=False)
        oos_metrics_df = pd.DataFrame([oos_result.get("metrics", {})])
        oos_metrics_df["n_trades"] = oos_result["n_trades"]
        oos_metrics_df.to_csv(OUTPUT_DIR / "oos_strategy_metrics.csv", index=False)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  {f.name}")

    return {
        "metrics":          metrics_df,
        "gain_importance":  gain_out,
        "perm_importance":  perm_out,
        "category_summary": cat_sum,
        "window_summary":   win_sum,
        "oos_result":       oos_result,
    }


if __name__ == "__main__":
    run_ml_analysis(
        symbols_prefix   = "M6E",
        start            = "2025-03-01",
        end              = "2026-03-01",
        horizons         = [60, 300, 600],
        top_n            = 30,
        initial_capital  = 1_000.0,
    )
