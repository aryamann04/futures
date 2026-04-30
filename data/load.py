import json
from pathlib import Path

import pandas as pd


DATA_ROOT = Path(__file__).resolve().parent

DATASET_DIRS = {
    "micro_currency_futures": "micro-currency-futures-databento",
    "micro_sp_futures": "micro-sp-futures-databento",
}

DEFAULT_DTYPES = {
    "rtype": "int16",
    "publisher_id": "int16",
    "instrument_id": "int32",
    "open": "float32",
    "high": "float32",
    "low": "float32",
    "close": "float32",
    "volume": "float32",
    "symbol": "string",
}


def _resolve_base_dir(dataset: str) -> Path:
    if dataset not in DATASET_DIRS:
        valid = ", ".join(DATASET_DIRS.keys())
        raise ValueError(f"Unknown dataset '{dataset}'. Valid options: {valid}")
    return DATA_ROOT / DATASET_DIRS[dataset]


def _read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def dataset_csv_path(dataset: str = "micro_currency_futures") -> Path:
    return _resolve_base_dir(dataset) / "data.csv"


def load_metadata(dataset: str = "micro_currency_futures"):
    base_dir = _resolve_base_dir(dataset)
    return _read_json(base_dir / "metadata.json")


def load_manifest(dataset: str = "micro_currency_futures"):
    base_dir = _resolve_base_dir(dataset)
    return _read_json(base_dir / "manifest.json")


def load_conditions(dataset: str = "micro_currency_futures"):
    base_dir = _resolve_base_dir(dataset)
    return pd.DataFrame(_read_json(base_dir / "condition.json"))


def futures_data(
    dataset: str = "micro_currency_futures",
    columns=None,
    chunksize=None,
    dtype_map=None,
):
    path = dataset_csv_path(dataset)

    header = pd.read_csv(path, nrows=0)
    raw_columns = header.columns.tolist()
    lower_map = {c.lower(): c for c in raw_columns}

    usecols = None
    if columns is not None:
        usecols = [lower_map[c.lower()] for c in columns if c.lower() in lower_map]

    parse_dates = [c for c in raw_columns if c.lower() in {"ts_event", "ts_recv", "timestamp", "datetime"}]
    dtypes = {}
    if dtype_map is None:
        dtype_map = DEFAULT_DTYPES
    for raw_col in raw_columns:
        dtype = dtype_map.get(raw_col.lower())
        if dtype is not None and raw_col not in parse_dates:
            dtypes[raw_col] = dtype

    if chunksize is not None:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, chunksize=chunksize, dtype=dtypes)

    df = pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, dtype=dtypes)
    df.columns = [c.lower() for c in df.columns]
    return df


def micro_futures_data(
    columns=None,
    chunksize=None,
    dataset: str = "micro_currency_futures",
):
    return futures_data(
        dataset=dataset,
        columns=columns,
        chunksize=chunksize,
    )


if __name__ == "__main__":
    for dataset in ["micro_currency_futures", "micro_sp_futures"]:
        print(f"\nDATASET: {dataset}")
        print(load_metadata(dataset))
        print(load_manifest(dataset))
        print(load_conditions(dataset).head())
        print(next(futures_data(dataset=dataset, chunksize=5)))
