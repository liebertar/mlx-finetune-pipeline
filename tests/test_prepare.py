import json

import pytest

from mlx_pipeline.steps.prepare import split_records, write_splits

RECORDS = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(20)]


def test_split_sizes():
    splits = split_records(RECORDS, valid_split=0.1, test_split=0.1)
    assert len(splits["valid"]) == 2
    assert len(splits["test"]) == 2
    assert len(splits["train"]) == 16


def test_split_is_deterministic():
    a = split_records(RECORDS, 0.1, 0.1)
    b = split_records(RECORDS, 0.1, 0.1)
    assert a == b


def test_split_has_no_overlap_or_loss():
    splits = split_records(RECORDS, 0.1, 0.1)
    combined = splits["train"] + splits["valid"] + splits["test"]
    assert sorted(json.dumps(r) for r in combined) == sorted(json.dumps(r) for r in RECORDS)


def test_rejects_too_small_dataset():
    with pytest.raises(ValueError, match="too small"):
        split_records(RECORDS[:2], 0.5, 0.4)


def test_writes_mlx_layout(tmp_path):
    splits = split_records(RECORDS, 0.1, 0.1)
    out = write_splits(splits, tmp_path / "dataset")
    for name in ("train", "valid", "test"):
        assert (out / f"{name}.jsonl").exists()
