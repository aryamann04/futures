import json
from pathlib import Path
import pandas as pd

from data.load import load_conditions, load_manifest, load_metadata, micro_futures_data

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

def _jsonify(x):
    if pd.isna(x):
        return None
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            return str(x)
    return x


def _write_json(path: Path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonify)


def _pick_column(columns, candidates, required=True):
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def _init_state():
    return {
        "n_obs": 0,
        "min_ts": None,
        "max_ts": None,
        "n_missing_ts": 0,
        "total_volume": 0.0,
        "nonzero_volume_obs": 0,
        "zero_volume_obs": 0,
        "missing_volume_obs": 0,
        "open_min": None,
        "open_max": None,
        "high_min": None,
        "high_max": None,
        "low_min": None,
        "low_max": None,
        "close_min": None,
        "close_max": None,
        "n_missing_open": 0,
        "n_missing_high": 0,
        "n_missing_low": 0,
        "n_missing_close": 0,
        "dates": set(),
    }


def _update_min(curr, val):
    if val is None or pd.isna(val):
        return curr
    if curr is None:
        return val
    return min(curr, val)


def _update_max(curr, val):
    if val is None or pd.isna(val):
        return curr
    if curr is None:
        return val
    return max(curr, val)


def _update_group_state(state, group, ts_col, volume_col, open_col, high_col, low_col, close_col):
    state["n_obs"] += len(group)

    ts = group[ts_col]
    state["n_missing_ts"] += int(ts.isna().sum())
    if ts.notna().any():
        state["min_ts"] = _update_min(state["min_ts"], ts.min())
        state["max_ts"] = _update_max(state["max_ts"], ts.max())
        state["dates"].update(pd.Series(ts.dropna().dt.date.unique()).tolist())

    vol = group[volume_col] if volume_col is not None else pd.Series(index=group.index, dtype=float)
    if volume_col is not None:
        state["missing_volume_obs"] += int(vol.isna().sum())
        nz = vol.fillna(0) > 0
        state["nonzero_volume_obs"] += int(nz.sum())
        state["zero_volume_obs"] += int((vol.fillna(0) == 0).sum())
        state["total_volume"] += float(vol.fillna(0).sum())

    for col_name, min_key, max_key, miss_key in [
        (open_col, "open_min", "open_max", "n_missing_open"),
        (high_col, "high_min", "high_max", "n_missing_high"),
        (low_col, "low_min", "low_max", "n_missing_low"),
        (close_col, "close_min", "close_max", "n_missing_close"),
    ]:
        if col_name is None:
            continue
        s = group[col_name]
        state[miss_key] += int(s.isna().sum())
        if s.notna().any():
            state[min_key] = _update_min(state[min_key], s.min())
            state[max_key] = _update_max(state[max_key], s.max())


def _finalize_summary(summary):
    rows = []
    for symbol, state in summary.items():
        n_dates = len(state["dates"])
        n_obs = state["n_obs"]
        rows.append(
            {
                "symbol": symbol,
                "n_obs": n_obs,
                "start_ts": state["min_ts"],
                "end_ts": state["max_ts"],
                "n_dates": n_dates,
                "avg_obs_per_date": n_obs / n_dates if n_dates else None,
                "total_volume": state["total_volume"],
                "avg_volume_per_obs": state["total_volume"] / n_obs if n_obs else None,
                "nonzero_volume_obs": state["nonzero_volume_obs"],
                "zero_volume_obs": state["zero_volume_obs"],
                "pct_zero_volume_obs": state["zero_volume_obs"] / n_obs if n_obs else None,
                "missing_ts_obs": state["n_missing_ts"],
                "missing_volume_obs": state["missing_volume_obs"],
                "missing_open_obs": state["n_missing_open"],
                "missing_high_obs": state["n_missing_high"],
                "missing_low_obs": state["n_missing_low"],
                "missing_close_obs": state["n_missing_close"],
                "open_min": state["open_min"],
                "open_max": state["open_max"],
                "high_min": state["high_min"],
                "high_max": state["high_max"],
                "low_min": state["low_min"],
                "low_max": state["low_max"],
                "close_min": state["close_min"],
                "close_max": state["close_max"],
            }
        )
    df = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    return df


def _build_contract_summary(chunksize=1_000_000):
    first_chunk = next(micro_futures_data(chunksize=5))
    columns = first_chunk.columns.tolist()

    symbol_col = _pick_column(columns, ["symbol", "raw_symbol", "instrument_id", "ticker"])
    ts_col = _pick_column(columns, ["ts_event", "timestamp", "datetime", "date", "marketdate"])
    volume_col = _pick_column(columns, ["volume"], required=False)
    open_col = _pick_column(columns, ["open", "open_"], required=False)
    high_col = _pick_column(columns, ["high", "high_"], required=False)
    low_col = _pick_column(columns, ["low", "low_"], required=False)
    close_col = _pick_column(columns, ["close", "close_", "settlement"], required=False)

    summary = {}

    for chunk in micro_futures_data(chunksize=chunksize):
        if ts_col in chunk.columns:
            chunk[ts_col] = pd.to_datetime(chunk[ts_col], errors="coerce")

        for symbol, group in chunk.groupby(symbol_col, dropna=False):
            key = str(symbol)
            if key not in summary:
                summary[key] = _init_state()
            _update_group_state(summary[key], group, ts_col, volume_col, open_col, high_col, low_col, close_col)

    contract_df = _finalize_summary(summary)

    context = {
        "columns": columns,
        "symbol_col": symbol_col,
        "ts_col": ts_col,
        "volume_col": volume_col,
        "open_col": open_col,
        "high_col": high_col,
        "low_col": low_col,
        "close_col": close_col,
    }
    return contract_df, context


def _build_dataset_overview(contract_df, context):
    metadata = load_metadata()
    manifest = load_manifest()
    conditions = load_conditions()

    file_info = None
    for f in manifest.get("files", []):
        name = f.get("filename", "")
        if name.endswith(".csv"):
            file_info = f
            break

    condition_counts = {}
    if not conditions.empty and "condition" in conditions.columns:
        condition_counts = conditions["condition"].value_counts(dropna=False).to_dict()

    overview = {
        "dataset": metadata.get("query", {}).get("dataset"),
        "schema": metadata.get("query", {}).get("schema"),
        "start": metadata.get("query", {}).get("start"),
        "end": metadata.get("query", {}).get("end"),
        "symbols_requested": metadata.get("query", {}).get("symbols"),
        "customizations": metadata.get("customizations", {}),
        "csv_file": file_info,
        "detected_columns": context["columns"],
        "detected_symbol_col": context["symbol_col"],
        "detected_ts_col": context["ts_col"],
        "detected_volume_col": context["volume_col"],
        "detected_open_col": context["open_col"],
        "detected_high_col": context["high_col"],
        "detected_low_col": context["low_col"],
        "detected_close_col": context["close_col"],
        "n_contracts_found": int(contract_df["symbol"].nunique()),
        "contracts_found": contract_df["symbol"].tolist(),
        "total_obs": int(contract_df["n_obs"].sum()),
        "overall_start_ts": contract_df["start_ts"].min(),
        "overall_end_ts": contract_df["end_ts"].max(),
        "total_volume": float(contract_df["total_volume"].sum()),
        "condition_counts": condition_counts,
    }
    return overview


def _build_condition_summary():
    conditions = load_conditions()
    if conditions.empty:
        return {}

    if "date" in conditions.columns:
        conditions["date"] = pd.to_datetime(conditions["date"], errors="coerce")

    summary = {
        "n_rows": int(len(conditions)),
        "start_date": conditions["date"].min() if "date" in conditions.columns else None,
        "end_date": conditions["date"].max() if "date" in conditions.columns else None,
        "condition_counts": conditions["condition"].value_counts(dropna=False).to_dict() if "condition" in conditions.columns else {},
    }

    if "condition" in conditions.columns:
        degraded = conditions.loc[conditions["condition"] != "available"].copy()
        if not degraded.empty:
            summary["non_available_dates"] = degraded[["date", "condition"]].assign(
                date=lambda x: x["date"].astype(str)
            ).to_dict(orient="records")
        else:
            summary["non_available_dates"] = []

    return summary


def _write_markdown_summary(contract_df, path: Path):
    display_df = contract_df.copy()
    for col in ["start_ts", "end_ts"]:
        if col in display_df.columns:
            display_df[col] = pd.to_datetime(display_df[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in ["pct_zero_volume_obs"]:
        if col in display_df.columns:
            display_df[col] = (100 * display_df[col]).round(2)
    for col in ["total_volume", "avg_volume_per_obs"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)

    with open(path, "w") as f:
        f.write("# Contract Summary\n\n")
        f.write(display_df.to_markdown(index=False))
        f.write("\n")


def run_eda(chunksize=1_000_000):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    contract_df, context = _build_contract_summary(chunksize=chunksize)
    overview = _build_dataset_overview(contract_df, context)
    condition_summary = _build_condition_summary()

    contract_df.to_csv(OUTPUT_DIR / "contract_summary.csv", index=False)
    _write_markdown_summary(contract_df, OUTPUT_DIR / "contract_summary.md")
    _write_json(OUTPUT_DIR / "dataset_overview.json", overview)
    _write_json(OUTPUT_DIR / "condition_summary.json", condition_summary)

    print("\nContract summary\n")
    print(contract_df.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()