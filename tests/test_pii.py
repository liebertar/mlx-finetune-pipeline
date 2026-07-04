import json
import subprocess
import sys
from pathlib import Path

from mlx_pipeline.pii_eval import mask_text, normalize, parse_entities, score

GOLD = [
    [{"type": "사람 이름", "text": "김지원"}, {"type": "전화번호", "text": "010-1234-5678"}],
    [{"type": "주소", "text": "서울시 강남구 테헤란로 123"}],
    [],
]


def test_score_perfect():
    result = score(GOLD, GOLD)
    assert result["precision"] == 1.0 and result["recall"] == 1.0 and result["f1"] == 1.0
    assert result["per_type"]["사람 이름"]["support"] == 1


def test_score_miss_and_false_positive():
    preds = [
        [{"type": "사람 이름", "text": "김지원"}],                      # phone missed (FN)
        [{"type": "주소", "text": "서울시 강남구 테헤란로 123"},
         {"type": "회사명", "text": "가짜회사"}],                        # FP
        [],
    ]
    result = score(GOLD, preds)
    assert result["recall"] < 1.0 and result["precision"] < 1.0
    assert result["per_type"]["전화번호"]["recall"] == 0.0


def test_wrong_type_is_both_fp_and_fn():
    preds = [[{"type": "회사명", "text": "김지원"},
              {"type": "전화번호", "text": "010-1234-5678"}], [], []]
    result = score(GOLD, preds)
    assert result["per_type"]["사람 이름"]["recall"] == 0.0


def test_parse_entities_tolerates_chatter():
    out = 'PII를 찾았습니다: {"entities": [{"type": "사람 이름", "text": "박도윤"}]} 이상입니다.'
    assert parse_entities(out) == [{"type": "사람 이름", "text": "박도윤"}]


def test_parse_entities_garbage_returns_empty():
    assert parse_entities("죄송합니다, 잘 모르겠습니다.") == []
    assert parse_entities('{"entities": "oops"}') == []


def test_normalize_collapses_whitespace():
    assert normalize("서울시  강남구\n테헤란로 123") == "서울시 강남구 테헤란로 123"


def test_mask_text_replaces_longest_first():
    text = "김지원 대리의 번호는 010-1234-5678입니다."
    entities = [{"type": "사람 이름", "text": "김지원"}, {"type": "전화번호", "text": "010-1234-5678"}]
    assert mask_text(text, entities) == "[사람 이름] 대리의 번호는 [전화번호]입니다."


def test_generator_gold_is_extractable(tmp_path):
    out = tmp_path / "pii.jsonl"
    subprocess.run(
        [sys.executable, "scripts/generate_pii_dataset.py", "--n", "50",
         "--seed", "3", "--out", str(out)],
        check=True, capture_output=True,
    )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 50
    difficulties = set()
    for rec in records:
        text = rec["messages"][0]["content"].split("텍스트: ")[-1]
        gold = json.loads(rec["messages"][1]["content"])["entities"]
        for e in gold:  # every gold entity appears verbatim exactly once
            assert text.count(e["text"]) == 1, (e, text)
        difficulties.add(rec["meta"]["difficulty"])
    assert difficulties == {"easy", "hard"}


def test_josa_resolution():
    sys.path.insert(0, str(Path("scripts").resolve()))
    from generate_pii_dataset import _resolve_josa

    assert _resolve_josa("김지원[이가] 왔다") == "김지원이 왔다"
    assert _resolve_josa("박도유[이가] 왔다") == "박도유가 왔다"
    assert _resolve_josa("휴먼물산[으로로] 이직") == "휴먼물산으로 이직"
    assert _resolve_josa("서울[으로로] 이동") == "서울로 이동"       # ㄹ 받침 → 로
    assert _resolve_josa("가온테크[은는] 회사다") == "가온테크는 회사다"
