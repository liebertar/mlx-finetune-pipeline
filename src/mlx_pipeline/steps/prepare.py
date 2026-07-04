"""Node 2 — prepare: deterministic split and materialization to the layout
mlx-lm / mlx-vlm expect (a directory with train.jsonl / valid.jsonl / test.jsonl)."""

import json
import random
from pathlib import Path

from zenml import step


def split_records(
    records: list[dict], valid_split: float, test_split: float, seed: int = 42
) -> dict[str, list[dict]]:
    if valid_split + test_split >= 1.0:
        raise ValueError("valid_split + test_split must be < 1.0")
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_valid = max(1, int(n * valid_split))
    n_test = max(1, int(n * test_split))
    if n_valid + n_test >= n:
        raise ValueError(f"dataset too small to split: {n} records")
    return {
        "valid": shuffled[:n_valid],
        "test": shuffled[n_valid : n_valid + n_test],
        "train": shuffled[n_valid + n_test :],
    }


def write_splits(splits: dict[str, list[dict]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in splits.items():
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_dir


@step
def prepare(records: list[dict], valid_split: float, test_split: float, work_dir: str) -> str:
    splits = split_records(records, valid_split, test_split)
    return str(write_splits(splits, Path(work_dir) / "dataset"))
