from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from data.load import micro_futures_data
from features.build_features import add_targets, build_features


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "research"


def _pick_column(columns, candidates, required=True):
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def _safe_qcut(series: pd.Series, q: int):
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if len(valid) < q * 20 or valid.nunique() < q:
        return pd.Series(index=series.index, dtype="object")
    try:
        return pd.qcut(s, q=q, duplicates="drop")
    except Exception:
        return pd.Series(index=series.index, dtype="object")


def _ols_stats(x: pd.Series, y: pd.Series):
    data = pd.concat([x, y], axis=1).dropna()
    if len(data) < 100:
        return np.nan, np.nan, np.nan, np.nan
    X = sm.add_constant(data.iloc[:, 0])
    model = sm.OLS(data.iloc[:, 1], X).fit()
    return (
        float(model.params.iloc[1]),
        float(model.tvalues.iloc[1]),
        float(model.rsquared),
        int(len(data)),
    )


def _bucket_summary(df: pd.DataFrame, feature: str, target: str, q: int = 10):
    buckets = _safe_qcut(df[feature], q=q)
    data = pd.DataFrame({"bucket": buckets, target: df[target]}).dropna()
    if data.empty:
        return pd.DataFrame()
    out = (
        data.groupby("bucket", observed=False)[target]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    out["feature"] = feature
    out["target"] = target
    cols = ["feature", "target", "bucket", "count", "mean", "median", "std"]
    return out[cols]


def _interaction_table(df: pd.DataFrame, f1: str, f2: str, target: str, q: int = 5):
    b1 = _safe_qcut(df[f1], q=q)
    b2 = _safe_qcut(df[f2], q=q)
    data = pd.DataFrame({"b1": b1, "b2": b2, target: df[target]}).dropna()
    if data.empty:
        return pd.DataFrame()
    pivot = pd.pivot_table(
        data,
        values=target,
        index="b1",
        columns="b2",
        aggfunc="mean",
        observed=False,
    )
    pivot.index.name = f1
    pivot.columns.name = f2
    return pivot


def _corr_stats(x: pd.Series, y: pd.Series):
    data = pd.concat([x, y], axis=1).dropna()
    if len(data) < 100:
        return np.nan, np.nan, 0
    pearson = float(data.iloc[:, 0].corr(data.iloc[:, 1], method="pearson"))
    spearman = float(data.iloc[:, 0].corr(data.iloc[:, 1], method="spearman"))
    return pearson, spearman, int(len(data))


def _detect_feature_columns(df: pd.DataFrame, symbol_col: str, ts_col: str):
    excluded = {
        symbol_col,
        ts_col,
        "open",
        "open_",
        "high",
        "high_",
        "low",
        "low_",
        "close",
        "close_",
        "settlement",
        "volume",
    }
    feature_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if col.startswith("target_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


def _print_header(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _print_top(df: pd.DataFrame, sort_cols, ascending, n=20):
    if df.empty:
        print("No results.")
        return
    print(df.sort_values(sort_cols, ascending=ascending).head(n).to_string(index=False))


def _normalize_symbols(symbols):
    if symbols is None:
        return None
    if isinstance(symbols, str):
        return [symbols]
    return list(symbols)


def _coerce_timestamp_like(value, series: pd.Series):
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


def _filter_raw_data(raw: pd.DataFrame, symbols=None, symbols_prefix=None, include_spreads=False, start=None, end=None):
    raw = raw.copy()
    raw.columns = [c.lower() for c in raw.columns]

    symbol_col = _pick_column(raw.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"])
    ts_col = _pick_column(raw.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])

    raw[ts_col] = pd.to_datetime(raw[ts_col], errors="coerce")
    raw = raw.dropna(subset=[ts_col])

    symbols = _normalize_symbols(symbols)
    if symbols is not None:
        raw = raw[raw[symbol_col].astype(str).isin([str(x) for x in symbols])]
    
    if symbols_prefix is not None:
        raw = raw[raw[symbol_col].astype(str).str.startswith(symbols_prefix)]
    
    if not include_spreads:
        raw = raw[~raw[symbol_col].astype(str).str.contains("-")]

    if start is not None:
        start = _coerce_timestamp_like(start, raw[ts_col])
        raw = raw[raw[ts_col] >= start]

    if end is not None:
        end = _coerce_timestamp_like(end, raw[ts_col]) + pd.Timedelta(days=1)
        raw = raw[raw[ts_col] < end]

    raw = raw.sort_values([symbol_col, ts_col]).reset_index(drop=True)
    return raw


def _make_run_tag(symbols=None, start=None, end=None):
    parts = []
    symbols = _normalize_symbols(symbols)

    if symbols:
        symbol_part = "-".join(str(s).replace("/", "_").replace(".", "_") for s in symbols[:5])
        if len(symbols) > 5:
            symbol_part += "-more"
        parts.append(symbol_part)

    if start is not None:
        parts.append(f"from_{pd.Timestamp(start).strftime('%Y%m%d')}")

    if end is not None:
        parts.append(f"to_{pd.Timestamp(end).strftime('%Y%m%d')}")

    if not parts:
        return "all"

    return "__".join(parts)


def run_research(
    horizons=(60, 300, 600),
    interaction_features=("rsi_14", "price_vs_ema20", "bb_pos", "adx_14", "rel_volume_20"),
    bucket_q=10,
    interaction_q=5,
    symbols=None,
    symbols_prefix=None,
    include_spreads=False,
    start=None,
    end=None,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = micro_futures_data(
        columns=["ts_event", "symbol", "open", "high", "low", "close", "volume"]
    )
    raw = _filter_raw_data(raw, symbols=symbols, symbols_prefix=symbols_prefix, include_spreads=include_spreads, start=start, end=end)

    if raw.empty:
        print("No rows found for the requested symbol/date filter.")
        return

    raw.columns = [c.lower() for c in raw.columns]
    raw_symbol_col = _pick_column(raw.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"])
    raw_ts_col = _pick_column(raw.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])

    print(f"Filtered rows: {len(raw):,}")
    print(f"Symbols: {sorted(raw[raw_symbol_col].astype(str).unique().tolist())}")
    print(f"Time range: {raw[raw_ts_col].min()} -> {raw[raw_ts_col].max()}")

    df = build_features(raw)
    df = add_targets(df, horizons=horizons)

    df.columns = [c.lower() for c in df.columns]
    symbol_col = _pick_column(df.columns, ["symbol", "raw_symbol", "instrument_id", "ticker"])
    ts_col = _pick_column(df.columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    feature_cols = _detect_feature_columns(df, symbol_col, ts_col)
    target_cols = [f"target_fwd_ret_{h}s" for h in horizons if f"target_fwd_ret_{h}s" in df.columns]

    all_corr_rows = []
    all_bucket_rows = []
    all_ols_rows = []

    for symbol, g in df.groupby(symbol_col, sort=False):
        _print_header(f"SYMBOL: {symbol}")
        print(f"Rows: {len(g):,}")
        print(f"Time range: {g[ts_col].min()} -> {g[ts_col].max()}")

        symbol_corr_rows = []
        symbol_ols_rows = []

        for target in target_cols:
            for feature in feature_cols:
                pearson, spearman, n_obs = _corr_stats(g[feature], g[target])
                beta, tstat, r2, n_ols = _ols_stats(g[feature], g[target])

                corr_row = {
                    "symbol": symbol,
                    "target": target,
                    "feature": feature,
                    "pearson_corr": pearson,
                    "spearman_corr": spearman,
                    "n_obs": n_obs,
                }
                ols_row = {
                    "symbol": symbol,
                    "target": target,
                    "feature": feature,
                    "beta": beta,
                    "tstat": tstat,
                    "r2": r2,
                    "n_obs": n_ols,
                }

                symbol_corr_rows.append(corr_row)
                symbol_ols_rows.append(ols_row)
                all_corr_rows.append(corr_row)
                all_ols_rows.append(ols_row)

                bucket_df = _bucket_summary(g, feature, target, q=bucket_q)
                if not bucket_df.empty:
                    bucket_df.insert(0, "symbol", symbol)
                    all_bucket_rows.append(bucket_df)

        symbol_corr_df = pd.DataFrame(symbol_corr_rows)
        symbol_ols_df = pd.DataFrame(symbol_ols_rows)

        print("\nTop features by absolute Pearson correlation")
        _print_top(
            symbol_corr_df.assign(abs_pearson=lambda x: x["pearson_corr"].abs()),
            sort_cols=["target", "abs_pearson"],
            ascending=[True, False],
            n=20,
        )

        print("\nTop features by absolute Spearman correlation")
        _print_top(
            symbol_corr_df.assign(abs_spearman=lambda x: x["spearman_corr"].abs()),
            sort_cols=["target", "abs_spearman"],
            ascending=[True, False],
            n=20,
        )

        print("\nTop features by absolute OLS t-stat")
        _print_top(
            symbol_ols_df.assign(abs_tstat=lambda x: x["tstat"].abs()),
            sort_cols=["target", "abs_tstat"],
            ascending=[True, False],
            n=20,
        )

        available_interactions = [f for f in interaction_features if f in g.columns]
        for target in target_cols:
            _print_header(f"INTERACTIONS | {symbol} | {target}")
            for i in range(len(available_interactions)):
                for j in range(i + 1, len(available_interactions)):
                    f1 = available_interactions[i]
                    f2 = available_interactions[j]
                    table = _interaction_table(g, f1, f2, target, q=interaction_q)
                    if table.empty:
                        continue
                    print(f"\n{f1} x {f2}")
                    print(table.round(8).to_string())

    corr_df = pd.DataFrame(all_corr_rows)
    ols_df = pd.DataFrame(all_ols_rows)
    bucket_df = pd.concat(all_bucket_rows, axis=0, ignore_index=True) if all_bucket_rows else pd.DataFrame()

    overall_corr = (
        corr_df.groupby(["target", "feature"], as_index=False)
        .agg(
            mean_pearson_corr=("pearson_corr", "mean"),
            mean_spearman_corr=("spearman_corr", "mean"),
            median_pearson_corr=("pearson_corr", "median"),
            median_spearman_corr=("spearman_corr", "median"),
            total_obs=("n_obs", "sum"),
            n_symbols=("symbol", "nunique"),
        )
    )
    overall_corr["abs_mean_pearson_corr"] = overall_corr["mean_pearson_corr"].abs()
    overall_corr["abs_mean_spearman_corr"] = overall_corr["mean_spearman_corr"].abs()

    overall_ols = (
        ols_df.groupby(["target", "feature"], as_index=False)
        .agg(
            mean_beta=("beta", "mean"),
            median_beta=("beta", "median"),
            mean_tstat=("tstat", "mean"),
            median_tstat=("tstat", "median"),
            mean_r2=("r2", "mean"),
            total_obs=("n_obs", "sum"),
            n_symbols=("symbol", "nunique"),
        )
    )
    overall_ols["abs_mean_tstat"] = overall_ols["mean_tstat"].abs()

    run_tag = _make_run_tag(symbols=symbols, start=start, end=end)

    corr_df.to_csv(OUTPUT_DIR / f"feature_correlations_by_symbol__{run_tag}.csv", index=False)
    ols_df.to_csv(OUTPUT_DIR / f"feature_ols_by_symbol__{run_tag}.csv", index=False)
    overall_corr.to_csv(OUTPUT_DIR / f"feature_correlations_overall__{run_tag}.csv", index=False)
    overall_ols.to_csv(OUTPUT_DIR / f"feature_ols_overall__{run_tag}.csv", index=False)
    if not bucket_df.empty:
        bucket_df.to_csv(OUTPUT_DIR / f"feature_bucket_results__{run_tag}.csv", index=False)

    _print_header("OVERALL TOP FEATURES BY ABS MEAN PEARSON CORRELATION")
    _print_top(
        overall_corr,
        sort_cols=["target", "abs_mean_pearson_corr"],
        ascending=[True, False],
        n=30,
    )

    _print_header("OVERALL TOP FEATURES BY ABS MEAN SPEARMAN CORRELATION")
    _print_top(
        overall_corr,
        sort_cols=["target", "abs_mean_spearman_corr"],
        ascending=[True, False],
        n=30,
    )

    _print_header("OVERALL TOP FEATURES BY ABS MEAN T-STAT")
    _print_top(
        overall_ols,
        sort_cols=["target", "abs_mean_tstat"],
        ascending=[True, False],
        n=30,
    )

    print(f"\nSaved research outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_research(
        symbols_prefix="M6E",
        start="2025-03-01",
        end="2026-03-01",
    )