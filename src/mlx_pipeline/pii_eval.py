"""PII extraction evaluation: generate on the held-out test split, parse the
predicted entity JSON, and score precision/recall/F1 — micro, macro, and
per-type — against the injected gold entities.

Matching rule: a prediction is a true positive iff (type, normalized text)
exactly matches a gold entity of the same example. Whitespace is collapsed;
everything else is strict — masking a wrong span is a real failure.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

TYPES = [
    "사람 이름", "주민등록번호", "외국인등록번호", "여권번호", "운전면허번호",
    "전화번호", "이메일 주소", "계좌번호", "신용카드번호", "사업자등록번호",
    "주소", "우편번호", "회사명", "직책", "사번",
]

# The single canonical task instruction. The frozen gold sets store only
# (text, entities); this instruction is applied at eval time so every try
# scores against the identical prompt.
INSTRUCTION = (
    "다음 텍스트에서 개인정보(PII)를 모두 찾아 JSON으로 추출하세요. "
    "유형: " + ", ".join(TYPES) + ". "
    '출력 형식: {"entities": [{"type": "...", "text": "..."}]} '
    "(개인정보가 없으면 빈 배열)\n\n텍스트: "
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def parse_entities(generated: str) -> list[dict]:
    """Extract the first JSON object from model output; tolerate chatter around it."""
    match = JSON_RE.search(generated)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    entities = data.get("entities", [])
    if not isinstance(entities, list):
        return []
    return [
        {"type": str(e.get("type", "")), "text": normalize(str(e.get("text", "")))}
        for e in entities
        if isinstance(e, dict) and e.get("text")
    ]


PASSPORT_RE = re.compile(r"^[MS][0-9]{8}$")
EMP_ID_RE = re.compile(r"^[A-Z][0-9]{6}$")
RRN_RE = re.compile(r"^\d{6}-([1-8])\d{6}$")


def refine_entities(entities: list[dict]) -> list[dict]:
    """결정론적 형식 규칙 레이어 (모델 출력 후처리, 제품 구성요소).

    한국 개인정보 번호 체계는 형식이 유형을 결정한다 — 모델이 문맥 단서
    없이 헷갈린 유형을 규격으로 교정한다:
      영문+8자리 -> 여권번호 / 영문+6자리 -> 사번
      주민/외국인등록번호는 성별 자리(1-4 vs 5-8)가 결정
      "~팀"으로 끝나는 직책은 부서명 오탐 -> 제거
    """
    refined = []
    for e in entities:
        etype, text = e["type"], e["text"]
        if etype in ("사번", "여권번호"):
            if PASSPORT_RE.match(text):
                etype = "여권번호"
            elif EMP_ID_RE.match(text):
                etype = "사번"
        elif etype in ("주민등록번호", "외국인등록번호"):
            match = RRN_RE.match(text)
            if match:
                etype = "주민등록번호" if match.group(1) in "1234" else "외국인등록번호"
        elif etype == "직책" and text.endswith("팀"):
            continue
        refined.append({"type": etype, "text": text})
    return refined


def score(golds: list[list[dict]], preds: list[list[dict]]) -> dict:
    """Per-type and aggregate P/R/F1 over parallel lists of entity lists."""
    tp: dict = defaultdict(int)
    fp: dict = defaultdict(int)
    fn: dict = defaultdict(int)
    support: dict = defaultdict(int)

    for gold, pred in zip(golds, preds):
        gold_set = {(e["type"], normalize(e["text"])) for e in gold}
        pred_set = {(e["type"], normalize(e["text"])) for e in pred}
        for etype, _ in gold_set:
            support[etype] += 1
        for item in pred_set & gold_set:
            tp[item[0]] += 1
        for item in pred_set - gold_set:
            fp[item[0]] += 1
        for item in gold_set - pred_set:
            fn[item[0]] += 1

    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    per_type = {}
    for etype in sorted(support, key=lambda t: -support[t]):
        p, r, f = prf(tp[etype], fp[etype], fn[etype])
        per_type[etype] = {"precision": p, "recall": r, "f1": f, "support": support[etype]}

    micro_p, micro_r, micro_f = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    macro_f = (
        sum(v["f1"] for v in per_type.values()) / len(per_type) if per_type else 0.0
    )
    return {
        "precision": micro_p,
        "recall": micro_r,
        "f1": micro_f,
        "macro_f1": macro_f,
        "per_type": per_type,
        "n_examples": len(golds),
    }


def load_test_records(dataset_dir: str, max_examples: int | None = None) -> list[dict]:
    path = Path(dataset_dir)
    if path.suffix != ".jsonl":            # a dataset dir -> its test split
        path = path / "test.jsonl"
    lines = path.read_text().splitlines()
    if max_examples:
        lines = lines[:max_examples]
    records = []
    for line in lines:
        rec = json.loads(line)
        if "text" in rec:  # flat answer-key format: {text, entities, difficulty}
            records.append({
                "prompt": INSTRUCTION + rec["text"],
                "gold": [{"type": e["type"], "text": normalize(e["text"])}
                         for e in rec["entities"]],
                "difficulty": rec.get("difficulty", "easy"),
            })
            continue
        prompt = next(m["content"] for m in rec["messages"] if m["role"] == "user")
        gold_raw = next(m["content"] for m in rec["messages"] if m["role"] == "assistant")
        records.append({
            "prompt": prompt,
            "gold": parse_entities(gold_raw),
            "difficulty": rec.get("meta", {}).get("difficulty", "easy"),
        })
    return records


def run_pii_eval(
    model_id: str,
    dataset_dir: str,
    adapter_path: str | None = None,
    max_examples: int | None = None,
    max_tokens: int = 300,
    progress_every: int = 50,
) -> dict:
    """Load the model once, generate for every test example, score the lot."""
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = (
        load(model_id, adapter_path=adapter_path) if adapter_path else load(model_id)
    )
    sampler = make_sampler(temp=0.0)  # deterministic

    records = load_test_records(dataset_dir, max_examples)
    golds, preds, samples = [], [], []
    for i, rec in enumerate(records):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": rec["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                       sampler=sampler, verbose=False)
        golds.append(rec["gold"])
        preds.append(refine_entities(parse_entities(out)))
        if len(samples) < 5:
            samples.append({"prompt": rec["prompt"], "generated": out, "gold": rec["gold"]})
        if progress_every and (i + 1) % progress_every == 0:
            print(f"pii eval: {i + 1}/{len(records)}", flush=True)

    result = score(golds, preds)
    result["samples"] = samples
    # ambiguous-case breakdown: how does the model hold up on the hard subset?
    result["by_difficulty"] = {}
    for level in ("easy", "hard"):
        idx = [i for i, r in enumerate(records) if r["difficulty"] == level]
        if idx:
            sub = score([golds[i] for i in idx], [preds[i] for i in idx])
            sub.pop("per_type")
            result["by_difficulty"][level] = sub
    return result


def mask_text(text: str, entities: list[dict]) -> str:
    """The masking product: replace each extracted entity with [유형]."""
    for e in sorted(entities, key=lambda e: -len(e["text"])):
        text = text.replace(e["text"], f"[{e['type']}]")
    return text
