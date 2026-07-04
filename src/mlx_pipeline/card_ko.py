"""한국어 모델 카드 렌더러 (PII 추출/마스킹 모델용).

publish 노드가 pii_report.json / sweep.json / 원장(ledger) / 차트를 모아
완전한 실험 리포트 형태의 카드를 생성한다. 파이프라인 산출물만 기록한다.
"""

import json
from pathlib import Path

CARD = """---
library_name: mlx
base_model: {base_model}
language:
- ko
pipeline_tag: text-generation
tags:
  - mlx
  - korean
  - pii
  - privacy
  - ner
---

# {repo_id}

한국어 텍스트에서 **개인정보(PII) 15종을 JSON으로 추출**하고, 후처리 한 줄로
마스킹까지 수행하는 모델입니다. `{base_model}` 를 LoRA 방식으로
파인튜닝했으며, Apple Silicon(MLX)에서 그대로 추론할 수 있습니다.

**지원 유형(15종):** 사람 이름, 주민등록번호, 외국인등록번호, 여권번호,
운전면허번호, 전화번호, 이메일 주소, 계좌번호, 신용카드번호, 사업자등록번호,
주소, 우편번호, 회사명, 직책, 사번

## 성능

**실전 지표는 미학습 문형(OOT) 열입니다** — 학습에 없던 문장 형태에서의
성능. 골드셋 열은 문형이 학습 분포와 같아(값만 교체) 상한선에 가깝습니다.

| 지표 | **미학습 문형 OOT ({n_oot}건)** | 골드셋 ({n_test}건, 문형=학습 분포) | 베이스 zero-shot |
|---|---|---|---|
| **Precision (micro)** | **{o_p:.3f}** | {t_p:.3f} | {b_p:.3f} |
| **Recall (micro)** | **{o_r:.3f}** | {t_r:.3f} | {b_r:.3f} |
| **F1 (micro)** | **{o_f:.3f}** | {t_f:.3f} | {b_f:.3f} |
| **F1 (macro)** | **{o_mf:.3f}** | {t_mf:.3f} | {b_mf:.3f} |

난이도별 (골드셋에 포함된 애매/혼동 유발 케이스 분리 채점):

{difficulty_table}

유형별 F1 — 파인튜닝 vs 베이스:

![유형별 F1](pii_per_type.png)

<details><summary>유형별 상세 (P / R / F1 / support)</summary>

{per_type_table}

</details>

## 평가 방법과 정답지(골드셋)

- 골드셋은 **시드 고정으로 생성된 불변 세트**입니다. 모든 실험(try)이 같은
  시험지로 채점되며, 데이터셋 저장소에 공개되어 있습니다: {goldset_link}
- 채점 기준: (유형, 문자열) **완전 일치**. 조사(은/는/이/가)까지 잘라내지
  못하면 오답 처리됩니다.
- 골드셋의 약 15%는 **의도적으로 애매한 케이스**입니다 — 송장번호 등 PII
  유사 숫자(마스킹하면 안 됨), 한 문장 속 이름 2개, 국제 전화 형식 등.

## 학습 데이터

외부 유출 없는 **합성 데이터**로 학습했습니다. 템플릿(업무 메일·CS 상담·
인사기록·계약·배송 등 {n_templates}종) × 한국어 개인정보 생성기(성명 풀,
번호 체계별 포맷)로 생성하며, 조사 자동 결합(받침 규칙)으로 자연스러운
문장을 만듭니다. 전체 {n_total}건 (train {n_train} / valid {n_valid} /
test {n_test}), 유형별 분포:

{data_table}

## 학습 설정

{params_table}

## 하이퍼파라미터 탐색

동일 데이터·동일 골드셋에서 조합을 비교해 최종 조합을 선정했습니다:

{sweep_table}

![조합별 검증 손실](sweep_comparison.png)

## Try별 진행 이력

골드셋을 고정한 채 파라미터/데이터를 바꿔가며 진행한 실험 이력입니다
(시간순이 아니라 try 순):

![Try별 성능 추이](try_progression.png)

## 학습 곡선

![학습 곡선](loss_curve.png)

## 사용법

```python
from mlx_lm import load, generate
import json, re

model, tokenizer = load("{repo_id}")

TYPES = "사람 이름, 주민등록번호, 외국인등록번호, 여권번호, 운전면허번호, 전화번호, 이메일 주소, 계좌번호, 신용카드번호, 사업자등록번호, 주소, 우편번호, 회사명, 직책, 사번"

def extract_pii(text: str) -> list[dict]:
    prompt = tokenizer.apply_chat_template([{{"role": "user", "content":
        f'다음 텍스트에서 개인정보(PII)를 모두 찾아 JSON으로 추출하세요. '
        f'유형: {{TYPES}}. 출력 형식: {{{{"entities": [{{{{"type": "...", "text": "..."}}}}]}}}} '
        f'(개인정보가 없으면 빈 배열)\\n\\n텍스트: {{text}}'}}],
        tokenize=False, add_generation_prompt=True)
    out = generate(model, tokenizer, prompt=prompt, max_tokens=300)
    match = re.search(r"\\{{.*\\}}", out, re.DOTALL)
    return json.loads(match.group(0))["entities"] if match else []

def refine(entities):
    # 형식 규칙 레이어: 한국 번호 체계는 규격이 유형을 결정합니다
    out = []
    for e in entities:
        t, x = e["type"], e["text"]
        if re.match(r"^[MS][0-9]{{8}}$", x): t = "여권번호"
        elif re.match(r"^[A-Z][0-9]{{6}}$", x) and t == "여권번호": t = "사번"
        m = re.match(r"^\\d{{6}}-([1-8])\\d{{6}}$", x)
        if m and t in ("주민등록번호", "외국인등록번호"):
            t = "주민등록번호" if m.group(1) in "1234" else "외국인등록번호"
        if t == "직책" and x.endswith("팀"): continue
        out.append({{"type": t, "text": x}})
    return out

def mask_pii(text: str) -> str:
    for e in sorted(refine(extract_pii(text)), key=lambda e: -len(e["text"])):
        text = text.replace(e["text"], f"[{{e['type']}}]")
    return text

print(mask_pii("오지원은 010-1234-5678로 연락 가능하며 주소는 서울시 강남구 테헤란로 123입니다."))
# [사람 이름]은 [전화번호]로 연락 가능하며 주소는 [주소]입니다.
```

## 추출/마스킹 예시 (골드셋에서 발췌)

{samples_section}

## 한계

- 합성 데이터 기반이므로 실제 업무 문서의 분포와 다를 수 있습니다. 실데이터
  골드셋 확장을 권장합니다.
- 15종 외의 개인정보 유형(건강정보 등)은 학습 범위 밖입니다.
- 생성형 추출 특성상 출력 JSON 파싱 실패 가능성이 있으며, 파싱 실패 시
  빈 결과로 처리하는 방어 코드를 권장합니다.

---
*이 카드의 수치·차트·표는 [mlx-finetune-pipeline](https://github.com/liebertar/mlx-finetune-pipeline)
파이프라인이 학습→평가→게이트→배포 과정에서 자동 생성한 것입니다.*
"""


def _table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render_korean_pii_card(
    repo_id: str, base_model: str, work_dir: str, dataset_dir: str,
    train_cfg: dict, goldset_repo: str | None,
) -> str:
    work = Path(work_dir)
    report = json.loads((work / "pii_report.json").read_text())
    tuned, base = report["tuned"], report.get("baseline", {})

    def g(d, k):
        return d.get(k, 0.0)

    per_type_rows = [
        (t, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}", v["support"])
        for t, v in tuned["per_type"].items()
    ]
    diff_rows = []
    for level, name in (("easy", "일반"), ("hard", "애매/혼동 케이스")):
        sub = tuned.get("by_difficulty", {}).get(level)
        if sub:
            diff_rows.append((name, sub["n_examples"], f"{sub['precision']:.3f}",
                              f"{sub['recall']:.3f}", f"{sub['f1']:.3f}"))
    if report.get("oot"):
        oot = report["oot"]
        diff_rows.append(("**미학습 문형(OOT)** — 학습에 없던 문장 형태", oot["n_examples"],
                          f"{oot['precision']:.3f}", f"{oot['recall']:.3f}", f"{oot['f1']:.3f}"))

    splits = {}
    counts: dict = {}
    for split in ("train", "valid", "test"):
        lines = (Path(dataset_dir) / f"{split}.jsonl").read_text().splitlines()
        splits[split] = len(lines)
        for line in lines:
            rec = json.loads(line)
            gold = json.loads(next(m["content"] for m in rec["messages"]
                                   if m["role"] == "assistant"))
            for e in gold["entities"]:
                counts[e["type"]] = counts.get(e["type"], 0) + 1
    data_rows = sorted(counts.items(), key=lambda kv: -kv[1])

    sweep_table = "_스윕 결과 없음_"
    if (work / "sweep.json").exists():
        sweep = json.loads((work / "sweep.json").read_text())
        sweep_table = _table(
            ["조합", "rank", "learning rate", "batch", "F1", ""],
            [(s["label"], s.get("rank", "-"), s.get("learning_rate", "-"),
              s.get("batch_size", "-"),
              f"{s.get('f1', float('nan')):.3f}" if s.get("f1") is not None else "-",
              "**선정**" if s.get("selected") else "") for s in sweep],
        )

    samples_section = ""
    if (work / "sample_generations.md").exists():
        samples_section = (work / "sample_generations.md").read_text()

    goldset_link = (f"[`{goldset_repo}`](https://huggingface.co/datasets/{goldset_repo})"
                    if goldset_repo else "_(비공개)_")

    oot = report.get("oot", {})
    return CARD.format(
        repo_id=repo_id, base_model=base_model,
        n_total=sum(splits.values()), n_train=splits["train"],
        n_valid=splits["valid"],
        n_test=tuned.get("n_examples", splits["test"]),  # frozen gold set size
        n_oot=oot.get("n_examples", 0),
        o_p=g(oot, "precision"), o_r=g(oot, "recall"),
        o_f=g(oot, "f1"), o_mf=g(oot, "macro_f1"),
        t_p=g(tuned, "precision"), t_r=g(tuned, "recall"),
        t_f=g(tuned, "f1"), t_mf=g(tuned, "macro_f1"),
        b_p=g(base, "precision"), b_r=g(base, "recall"),
        b_f=g(base, "f1"), b_mf=g(base, "macro_f1"),
        d_p=g(tuned, "precision") - g(base, "precision"),
        d_r=g(tuned, "recall") - g(base, "recall"),
        d_f=g(tuned, "f1") - g(base, "f1"),
        d_mf=g(tuned, "macro_f1") - g(base, "macro_f1"),
        difficulty_table=_table(["구분", "n", "P", "R", "F1"], diff_rows),
        per_type_table=_table(["유형", "P", "R", "F1", "support"], per_type_rows),
        data_table=_table(["유형", "개수"], data_rows),
        params_table=_table(["파라미터", "값"], sorted(train_cfg.items())),
        sweep_table=sweep_table,
        samples_section=samples_section,
        goldset_link=goldset_link,
        n_templates=35,  # 28 standard + 7 ambiguous templates in the generator
    )
