from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd


def detect_sweeps_and_reclaims(
    df: pd.DataFrame,
    level_columns: Sequence[str],
    reclaim_bars: int = 3,
    structure_bull_col: Optional[str] = None,
    structure_bear_col: Optional[str] = None,
) -> pd.DataFrame:
    events: list[dict[str, object]] = []
    for level_col in level_columns:
        if level_col not in df.columns:
            continue
        for i in range(len(df)):
            level = df.iloc[i][level_col]
            if pd.isna(level):
                continue
            level = float(level)
            for j in range(i, min(i + reclaim_bars + 1, len(df))):
                row = df.iloc[j]
                if row["low"] < level and row["close"] > level:
                    events.append(
                        {
                            "level_name": level_col,
                            "level_price": level,
                            "sweep_time": row["ts_event"] if "ts_event" in df.columns else row.iloc[0],
                            "reclaim_time": row["ts_event"] if "ts_event" in df.columns else row.iloc[0],
                            "direction": "bullish",
                            "bars_to_reclaim": j - i,
                            "max_excursion": level - float(row["low"]),
                            "structure_confirmed": bool(row.get(structure_bull_col, 0)) if structure_bull_col else False,
                            "fvg_after_sweep": False,
                        }
                    )
                    break
                if row["high"] > level and row["close"] < level:
                    events.append(
                        {
                            "level_name": level_col,
                            "level_price": level,
                            "sweep_time": row["ts_event"] if "ts_event" in df.columns else row.iloc[0],
                            "reclaim_time": row["ts_event"] if "ts_event" in df.columns else row.iloc[0],
                            "direction": "bearish",
                            "bars_to_reclaim": j - i,
                            "max_excursion": float(row["high"]) - level,
                            "structure_confirmed": bool(row.get(structure_bear_col, 0)) if structure_bear_col else False,
                            "fvg_after_sweep": False,
                        }
                    )
                    break
    return pd.DataFrame(events)
