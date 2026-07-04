"""정답지(골드셋) v2 빌더.

큰 후보 풀을 생성한 뒤 **템플릿 골격당 최대 cap개**만 골라 문형 다양성을
강제하고, 사람이 검토할 수 있는 flat 포맷으로 저장한다:
  {"id": 1, "text": "...", "entities": [{"type","text"}...], "difficulty": "easy|hard"}

- 골격(skeleton) = 텍스트에서 정답 엔티티와 숫자를 마스킹한 문자열.
  같은 골격이 cap번 이상 뽑히지 않으므로 복붙 시험지가 불가능하다.
- 시드는 학습 풀과 다른 값을 사용해 값 중복도 구조적으로 차단.

Usage: python scripts/build_goldset.py --n 300 --seed 9100 \
           --out data/samples/korean_pii_goldset.jsonl
"""

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlx_pipeline.pii_eval import parse_entities  # noqa: E402


def skeleton(text: str, entities: list[dict]) -> str:
    for e in sorted(entities, key=lambda e: -len(e["text"])):
        text = text.replace(e["text"], f"<{e['type']}>")
    return re.sub(r"\d+", "#", text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--cap", type=int, default=2, help="max examples per sentence skeleton")
    parser.add_argument("--pool", type=int, default=4000)
    parser.add_argument("--oot", action="store_true", help="build from held-out templates")
    parser.add_argument("--out", default="data/samples/korean_pii_goldset.jsonl")
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        pool_path = tmp.name
    cmd = [sys.executable, "scripts/generate_pii_dataset.py", "--n", str(args.pool),
           "--seed", str(args.seed), "--out", pool_path]
    if args.oot:
        cmd.append("--oot")
    subprocess.run(cmd, check=True, capture_output=True)

    candidates = []
    for line in Path(pool_path).read_text().splitlines():
        rec = json.loads(line)
        text = rec["messages"][0]["content"].split("텍스트: ", 1)[1]
        gold = parse_entities(rec["messages"][1]["content"])
        candidates.append({
            "text": text,
            "entities": gold,
            "difficulty": rec.get("meta", {}).get("difficulty", "easy"),
            "skeleton": skeleton(text, gold),
        })

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    skeleton_counts: Counter = Counter()
    selected = []
    # two passes: strict cap first; relax only if the pool can't fill n
    for cap in (args.cap, args.cap * 3, 10**9):
        for cand in candidates:
            if len(selected) >= args.n:
                break
            if skeleton_counts[cand["skeleton"]] >= cap or cand in selected:
                continue
            skeleton_counts[cand["skeleton"]] += 1
            selected.append(cand)
        if len(selected) >= args.n:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, rec in enumerate(selected, 1):
            f.write(json.dumps({
                "id": i, "text": rec["text"], "entities": rec["entities"],
                "difficulty": rec["difficulty"],
            }, ensure_ascii=False) + "\n")

    type_counts: Counter = Counter()
    for rec in selected:
        for e in rec["entities"]:
            type_counts[e["type"]] += 1
    n_hard = sum(1 for r in selected if r["difficulty"] == "hard")
    print(f"goldset: {len(selected)} examples | unique skeletons: {len(skeleton_counts)} "
          f"| max repeats: {max(skeleton_counts.values())} | hard: {n_hard}")
    print("type support:", json.dumps(dict(type_counts.most_common()), ensure_ascii=False))


if __name__ == "__main__":
    main()
