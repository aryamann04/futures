import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent / "micro-futures-databento"


def _read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def load_metadata():
    return _read_json(BASE_DIR / "metadata.json")


def load_manifest():
    return _read_json(BASE_DIR / "manifest.json")


def load_conditions():
    return pd.DataFrame(_read_json(BASE_DIR / "condition.json"))


def micro_futures_data(columns=None, chunksize=None):
    path = BASE_DIR / "data.csv"
    header = pd.read_csv(path, nrows=0)
    raw_columns = header.columns.tolist()
    lower_map = {c.lower(): c for c in raw_columns}

    usecols = None
    if columns is not None:
        usecols = [lower_map[c.lower()] for c in columns if c.lower() in lower_map]

    parse_dates = [c for c in raw_columns if c.lower() in {"ts_event", "ts_recv", "timestamp", "datetime"}]

    if chunksize is not None:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, chunksize=chunksize)

    df = pd.read_csv(path, usecols=usecols, parse_dates=parse_dates)
    df.columns = [c.lower() for c in df.columns]
    return df


if __name__ == "__main__":
    print(load_metadata())
    print(load_manifest())
    print(load_conditions().head())
    print(micro_futures_data(chunksize=5).__next__())