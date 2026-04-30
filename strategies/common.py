from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def prepare_strategy_df(df: pd.DataFrame, ts_col: str = "ts_event", symbol_col: Optional[str] = "symbol") -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    if ts_col not in out.columns:
        raise ValueError(f"Missing timestamp column: {ts_col}")
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
    sort_cols = ([symbol_col] if symbol_col and symbol_col in out.columns else []) + [ts_col]
    return out.dropna(subset=[ts_col]).sort_values(sort_cols).reset_index(drop=True)


def empty_plan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["entry_signal"] = 0
    out["exit_signal"] = 0
    out["stop_loss"] = np.nan
    out["take_profit"] = np.nan
    out["max_hold_bars"] = np.nan
    out["size"] = 1.0
    out["setup"] = ""
    out["session_name"] = ""
    out["flatten_eod"] = 1
    return out


def finalize_plan(plan: pd.DataFrame) -> pd.DataFrame:
    plan["entry_signal"] = pd.to_numeric(plan["entry_signal"], errors="coerce").fillna(0).astype("int8")
    plan["exit_signal"] = pd.to_numeric(plan["exit_signal"], errors="coerce").fillna(0).astype("int8")
    return plan

