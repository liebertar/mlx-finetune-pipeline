"""Score a GLiNER model against the same fixed gold set the pipeline uses —
for comparison only. Runs in .venv-gliner (PyTorch), not the MLX venv.

Usage: .venv-gliner/bin/python scripts/score_gliner.py \
           --model <gliner-model-id> \
           --data outputs/pii_masking/dataset --threshold 0.5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlx_pipeline.pii_eval import load_test_records, normalize, score  # noqa: E402

LABELS = [
    "사람 이름", "주민등록번호", "외국인등록번호", "여권번호", "운전면허번호",
    "전화번호", "이메일 주소", "계좌번호", "신용카드번호", "사업자등록번호",
    "주소", "우편번호", "회사명", "직책", "사번",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="dataset dir with test.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    from gliner import GLiNER

    model = GLiNER.from_pretrained(args.model, token=args.token)
    records = load_test_records(args.data, args.max_examples)

    golds, preds = [], []
    for i, rec in enumerate(records):
        text = rec["prompt"].split("텍스트: ")[-1]
        entities = model.predict_entities(text, LABELS, threshold=args.threshold)
        golds.append(rec["gold"])
        preds.append([{"type": e["label"], "text": normalize(e["text"])} for e in entities])
        if (i + 1) % 50 == 0:
            print(f"gliner: {i + 1}/{len(records)}", file=sys.stderr, flush=True)

    result = score(golds, preds)
    # difficulty breakdown, same as the pipeline evaluator
    result["by_difficulty"] = {}
    for level in ("easy", "hard"):
        idx = [i for i, r in enumerate(records) if r["difficulty"] == level]
        if idx:
            sub = score([golds[i] for i in idx], [preds[i] for i in idx])
            sub.pop("per_type")
            result["by_difficulty"][level] = sub
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
