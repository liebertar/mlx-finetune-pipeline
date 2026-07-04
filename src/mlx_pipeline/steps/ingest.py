"""Node 1 — ingest: load the raw dataset and validate its schema."""

import json
from pathlib import Path

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


@step
def ingest(source: str, fmt: str) -> list[dict]:
    return load_records(source, fmt)
