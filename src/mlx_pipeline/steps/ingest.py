"""Node 1 — ingest: load the raw dataset, validate its schema, and drop exact
duplicates (duplicates leak across splits and inflate every metric)."""

import json
from pathlib import Path

import mlflow
from zenml import step

REQUIRED_KEYS = {
    "chat": {"messages"},
    "image-text": {"image", "prompt", "completion"},
    "pairs": {"text", "positive"},
}


def load_records(source: str, fmt: str) -> list[dict]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {source}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"dataset is empty: {source}")
    required = REQUIRED_KEYS[fmt]
    for i, rec in enumerate(records):
        missing = required - rec.keys()
        if missing:
            raise ValueError(f"record {i} is missing keys {sorted(missing)} (format={fmt})")
    return records


def dedupe(records: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    unique = []
    for rec in records:
        key = json.dumps(rec, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique, len(records) - len(unique)


@step
def ingest(source: str, fmt: str, dedup: bool = True) -> list[dict]:
    records = load_records(source, fmt)
    n_dropped = 0
    if dedup:
        records, n_dropped = dedupe(records)
    mlflow.log_params({"dataset_source": source, "dataset_records": len(records),
                       "dataset_duplicates_dropped": n_dropped})
    if n_dropped:
        print(f"ingest: dropped {n_dropped} exact-duplicate records "
              f"({len(records)} remain) — duplicates leak across splits")
    return records
