import json

import pytest

from mlx_pipeline.steps.ingest import load_records


def test_loads_valid_chat_dataset():
    records = load_records("data/samples/llm_chat.jsonl", "chat")
    assert len(records) == 60
    assert all("messages" in r for r in records)


def test_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        load_records("data/samples/does_not_exist.jsonl", "chat")


def test_rejects_wrong_schema(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"prompt": "no messages key"}) + "\n")
    with pytest.raises(ValueError, match="missing keys"):
        load_records(str(bad), "chat")


def test_rejects_empty_file(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_records(str(empty), "chat")
