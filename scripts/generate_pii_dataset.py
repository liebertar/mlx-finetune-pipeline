"""Synthetic Korean PII dataset generator.

Builds chat-format SFT examples: user turn = extraction instruction + text with
injected PII, assistant turn = JSON entity list. Because entities are injected,
gold labels are exact — precision/recall on the held-out split is rigorous.

Entity taxonomy (15 types) matches common Korean PII categories.

Usage: python scripts/generate_pii_dataset.py --n 1300 --seed 7 \
           --out data/samples/korean_pii.jsonl
"""

import argparse
import json
import random
from pathlib import Path

TYPES = [
    "사람 이름", "주민등록번호", "외국인등록번호", "여권번호", "운전면허번호",
    "전화번호", "이메일 주소", "계좌번호", "신용카드번호", "사업자등록번호",
    "주소", "우편번호", "회사명", "직책", "사번",
]

INSTRUCTION = (
    "다음 텍스트에서 개인정보(PII)를 모두 찾아 JSON으로 추출하세요. "
    "유형: " + ", ".join(TYPES) + ". "
    '출력 형식: {"entities": [{"type": "...", "text": "..."}]} '
    "(개인정보가 없으면 빈 배열)\n\n텍스트: "
)

SURNAMES = "김 이 박 최 정 강 조 윤 장 임 한 오 서 신 권 황 안 송 전 홍 유 고 문 양 손 배 백 허 남 심".split()
GIVEN = ("지원 민준 서연 도윤 하은 시우 지우 준서 예은 수아 지호 유진 현우 세아 태윤 "
         "나연 건우 다은 재원 소율 은우 채원 시현 예준 가은 준혁 서진 하람 규민 온유").split()
DOMAINS = ["naver.com", "gmail.com", "daum.net", "kakao.com", "hanmail.net", "outlook.kr"]
CITIES = ["서울시 강남구", "서울시 마포구", "성남시 분당구", "수원시 영통구", "부산시 해운대구",
          "대전시 유성구", "인천시 연수구", "광주시 서구", "대구시 수성구", "고양시 일산동구"]
ROADS = ["테헤란로", "판교역로", "월드컵북로", "센텀중앙로", "대학로", "광평로", "송도과학로",
         "동성로", "정자일로", "백석로"]
COMPANY_HEADS = ["한빛", "누리", "다온", "세모비", "코어", "블루밍", "그린", "온새미로", "미르", "아라",
                 "새벽", "휴먼", "네오", "온빛", "가온"]
COMPANY_TAILS = ["테크", "소프트", "시스템즈", "전자", "물산", "바이오", "로보틱스", "에너지", "푸드", "랩스"]
TITLES = ["사원", "주임", "대리", "과장", "차장", "부장", "팀장", "실장", "이사", "상무",
          "전무", "대표이사", "수석연구원", "책임연구원", "선임개발자"]
BANKS = ["국민", "신한", "우리", "하나", "농협", "기업", "카카오뱅크", "토스뱅크"]


def make_generators(rng: random.Random):
    def digits(n):
        return "".join(rng.choice("0123456789") for _ in range(n))

    def name():
        return rng.choice(SURNAMES) + rng.choice(GIVEN)

    def company():
        head = rng.choice(COMPANY_HEADS)
        form = rng.choice(["(주){h}{t}", "{h}{t} 주식회사", "{h}{t}", "주식회사 {h}{t}"])
        return form.format(h=head, t=rng.choice(COMPANY_TAILS))

    return {
        "사람 이름": name,
        "주민등록번호": lambda: f"{digits(6)}-{rng.choice('1234')}{digits(6)}",
        "외국인등록번호": lambda: f"{digits(6)}-{rng.choice('5678')}{digits(6)}",
        "여권번호": lambda: f"{rng.choice('MS')}{digits(8)}",
        "운전면허번호": lambda: f"{rng.choice(['11','12','13','14','15','16','21','22'])}-{digits(2)}-{digits(6)}-{digits(2)}",
        "전화번호": lambda: rng.choice([
            f"010-{digits(4)}-{digits(4)}", f"02-{digits(3)}-{digits(4)}",
            f"031-{digits(3)}-{digits(4)}", f"+82-10-{digits(4)}-{digits(4)}",
            f"010{digits(8)}",
        ]),
        "이메일 주소": lambda: f"{rng.choice('abcdefghjkmnpr')}{rng.choice(['lee','kim','park','choi','jung'])}{digits(2)}@{rng.choice(DOMAINS)}",
        "계좌번호": lambda: rng.choice([
            f"{digits(3)}-{digits(3)}-{digits(6)}", f"{digits(4)}-{digits(2)}-{digits(7)}",
            f"{rng.choice(BANKS)}은행 {digits(3)}-{digits(2)}-{digits(6)}",
        ]),
        "신용카드번호": lambda: rng.choice([
            f"{digits(4)}-{digits(4)}-{digits(4)}-{digits(4)}",
            f"{digits(4)} {digits(4)} {digits(4)} {digits(4)}",
        ]),
        "사업자등록번호": lambda: f"{digits(3)}-{digits(2)}-{digits(5)}",
        "주소": lambda: f"{rng.choice(CITIES)} {rng.choice(ROADS)} {rng.randint(1, 400)}" + rng.choice(["", f" {rng.randint(1,25)}층", f" {rng.randint(101,2504)}호"]),
        "우편번호": lambda: digits(5),
        "회사명": company,
        "직책": lambda: rng.choice(TITLES),
        "사번": lambda: rng.choice([
            f"EMP-{rng.randint(2015,2026)}-{digits(4)}", f"{rng.choice('ABC')}{digits(6)}",
            f"{rng.randint(2015,2026)}{digits(4)}",
        ]),
    }


# {k}: entity slots. Templates use honorific/journalistic patterns incl. the
# particle-attachment cases (이름+은/는/이/가) GLiNER-style models miss.
# [이가]/[은는]/[을를]/[과와]/[으로로] resolve to the correct josa for the
# preceding word's final consonant after slot filling.
TEMPLATES = [
    ("{사람 이름}[은는] {전화번호}로 연락 가능하며 주소는 {주소}입니다.", ["사람 이름", "전화번호", "주소"]),
    ("{회사명} {직책} {사람 이름}입니다. 문의는 {이메일 주소}로 부탁드립니다.", ["회사명", "직책", "사람 이름", "이메일 주소"]),
    ("고객 {사람 이름}님(연락처 {전화번호})의 환불 계좌는 {계좌번호}입니다.", ["사람 이름", "전화번호", "계좌번호"]),
    ("주민등록번호 {주민등록번호}, 주소 {주소}, 우편번호 {우편번호}로 등록되었습니다.", ["주민등록번호", "주소", "우편번호"]),
    ("여권번호 {여권번호} 소지자 {사람 이름}[이가] {전화번호}로 예약을 변경했습니다.", ["여권번호", "사람 이름", "전화번호"]),
    ("결제는 카드 {신용카드번호}로 진행되었고 영수증은 {이메일 주소}로 발송됩니다.", ["신용카드번호", "이메일 주소"]),
    ("{회사명}(사업자등록번호 {사업자등록번호}) 대표 연락처는 {전화번호}입니다.", ["회사명", "사업자등록번호", "전화번호"]),
    ("사번 {사번} {직책} {사람 이름}[이가] 신규 입사자 멘토로 지정되었습니다.", ["사번", "직책", "사람 이름"]),
    ("운전면허번호 {운전면허번호} 확인 후 차량을 인도해 주세요. 인수자는 {사람 이름}입니다.", ["운전면허번호", "사람 이름"]),
    ("외국인등록번호 {외국인등록번호} 등록 건은 {직책} {사람 이름}에게 문의하세요.", ["외국인등록번호", "직책", "사람 이름"]),
    ("배송지: {주소} (우편번호 {우편번호}), 수령인 {사람 이름}, 연락처 {전화번호}", ["주소", "우편번호", "사람 이름", "전화번호"]),
    ("급여 이체 계좌를 {계좌번호}에서 {계좌번호}로 변경 요청드립니다. 사번 {사번}.", ["계좌번호", "계좌번호", "사번"]),
    ("{사람 이름} {직책}[은는] {회사명}[으로로] 이직했으며 새 이메일은 {이메일 주소}입니다.", ["사람 이름", "직책", "회사명", "이메일 주소"]),
    ("계약서에 주민등록번호 {주민등록번호}[과와] 주소 {주소}[을를] 기재해 주세요.", ["주민등록번호", "주소"]),
    ("문의하신 건은 {회사명} 고객센터 {전화번호} 또는 {이메일 주소}로 접수 가능합니다.", ["회사명", "전화번호", "이메일 주소"]),
    ("환불이 계좌 {계좌번호}(예금주 {사람 이름})로 3영업일 내 입금됩니다.", ["계좌번호", "사람 이름"]),
    ("사내 시스템 접속 오류는 사번 {사번}[과와] 함께 {이메일 주소}로 신고해 주세요.", ["사번", "이메일 주소"]),
    ("{사람 이름}[이가] 제출한 서류의 여권번호 {여권번호}[이가] 만료 예정입니다.", ["사람 이름", "여권번호"]),
    ("[인사기록] 성명: {사람 이름} / 직책: {직책} / 사번: {사번} / 내선 이메일: {이메일 주소}",
     ["사람 이름", "직책", "사번", "이메일 주소"]),
    ("안녕하세요 {사람 이름} 고객님, 가입하신 번호 {전화번호} 명의로 요금제가 변경되었습니다.",
     ["사람 이름", "전화번호"]),
    ("세금계산서 발행을 위해 {회사명} 사업자등록번호 {사업자등록번호}[과와] 담당자 메일 {이메일 주소}[을를] 회신 바랍니다.",
     ["회사명", "사업자등록번호", "이메일 주소"]),
    ("면접 안내: {사람 이름}님, {주소} 본사 3층으로 오시면 됩니다. 문의 {전화번호}",
     ["사람 이름", "주소", "전화번호"]),
    ("보험금 청구서의 계좌번호 {계좌번호}, 주민등록번호 {주민등록번호} 기재 여부를 확인했습니다.",
     ["계좌번호", "주민등록번호"]),
    ("렌터카 예약자 {사람 이름}, 운전면허번호 {운전면허번호}, 반납 지점 {주소}", ["사람 이름", "운전면허번호", "주소"]),
    ("법인차량 배정을 위해 운전면허번호 {운전면허번호}[과와] 사번 {사번}[을를] 제출하세요.", ["운전면허번호", "사번"]),
    ("체류 연장 신청: 외국인등록번호 {외국인등록번호}, 연락처 {전화번호}, 거주지 {주소}", ["외국인등록번호", "전화번호", "주소"]),
    ("환전 신청서에 여권번호 {여권번호}[과와] 외국인등록번호 {외국인등록번호}[이가] 모두 필요합니다.", ["여권번호", "외국인등록번호"]),
]

# Out-of-template gold set: sentence forms NEVER used in training data.
# Measures generalization to unseen phrasing, not just unseen values.
OOT_TEMPLATES = [
    ("퇴직 정산 안내드립니다. {사람 이름} 님 퇴직금은 {계좌번호} 계좌로 지급 예정이며, 원천징수 관련 문의는 {전화번호}입니다.",
     ["사람 이름", "계좌번호", "전화번호"]),
    ("수신: {회사명} 인사팀 / 발신: {사람 이름} {직책} / 제목: 재직증명서 발급 요청 (사번 {사번})",
     ["회사명", "사람 이름", "직책", "사번"]),
    ("아래 정보로 본인인증에 실패했습니다. 다시 확인해 주세요: 주민등록번호 {주민등록번호}, 휴대폰 {전화번호}",
     ["주민등록번호", "전화번호"]),
    ("긴급! {주소} 앞에서 접촉사고 발생. 목격자 {사람 이름}({전화번호}) 진술 확보됨. 면허 {운전면허번호} 조회 요망.",
     ["주소", "사람 이름", "전화번호", "운전면허번호"]),
    ("해외 송금 신청 내역 — 수취인: {사람 이름} / 여권: {여권번호} / 이메일: {이메일 주소}",
     ["사람 이름", "여권번호", "이메일 주소"]),
    ("정기 구독이 갱신되었습니다. 결제 수단: {신용카드번호} (명세서는 {이메일 주소} 발송)",
     ["신용카드번호", "이메일 주소"]),
    ("민원 접수번호 2026-0704호 처리 완료. 신청인 {사람 이름}, 거주지 {주소}, 우편번호 {우편번호}.",
     ["사람 이름", "주소", "우편번호"]),
    ("거래처 등록 심사: {회사명}, 사업자 {사업자등록번호}, 대표번호 {전화번호}, 담당 {직책} {사람 이름}",
     ["회사명", "사업자등록번호", "전화번호", "직책", "사람 이름"]),
    ("외국인 직원 {사람 이름}의 등록번호 {외국인등록번호} 갱신 서류를 {이메일 주소}로 보냈습니다.",
     ["사람 이름", "외국인등록번호", "이메일 주소"]),
    ("체크리스트: 계좌 {계좌번호} 해지 ✓ / 카드 {신용카드번호} 정지 ✓ / 사번 {사번} 비활성화 대기",
     ["계좌번호", "신용카드번호", "사번"]),
]

JOSA = {"[이가]": ("이", "가"), "[은는]": ("은", "는"), "[을를]": ("을", "를"),
        "[과와]": ("과", "와"), "[으로로]": ("으로", "로")}


def _resolve_josa(text: str) -> str:
    for marker, (with_batchim, without) in JOSA.items():
        while marker in text:
            idx = text.index(marker)
            prev = text[idx - 1]
            code = ord(prev)
            if 0xAC00 <= code <= 0xD7A3:
                final = (code - 0xAC00) % 28
                # ㄹ-final words take 로, not 으로
                pick = without if final == 0 or (marker == "[으로로]" and final == 8) else with_batchim
            else:
                pick = without  # digits/latin: default to the vowel form
            text = text[:idx] + pick + text[idx + len(marker):]
    return text

NEGATIVES = [
    "다음 분기 로드맵 회의는 수요일 오전 10시에 3층 회의실에서 진행됩니다.",
    "신규 기능 배포가 완료되어 전체 서비스가 정상 운영 중입니다.",
    "오늘 점심 메뉴는 비빔밥과 된장찌개 중 선택할 수 있습니다.",
    "프로젝트 마감 기한이 다음 주 금요일로 연장되었습니다.",
    "회의록은 공유 드라이브의 팀 폴더에 업로드되어 있습니다.",
    "이번 워크숍 장소는 추후 공지 예정입니다.",
]

# Ambiguous cases: PII lookalikes that must NOT be extracted (invoice/order
# numbers, quantities) next to real PII, partially-masked numbers that still
# count, international phone formats, multi-name sentences.
HARD_TEMPLATES = [
    ("송장번호 1234-5678-9012는 개인정보가 아니지만, 수령인 {사람 이름}의 연락처 {전화번호}는 마스킹 대상입니다.",
     ["사람 이름", "전화번호"]),
    ("담당자가 {사람 이름}[이가] 아니라 {사람 이름}입니다. 각각 사번 {사번}, {사번}으로 확인하세요.",
     ["사람 이름", "사람 이름", "사번", "사번"]),
    ("주문수량 54321개 발주 완료. 배송지는 {주소}, 우편번호 {우편번호}입니다.", ["주소", "우편번호"]),
    ("가입 신청서의 주민등록번호 {주민등록번호} 항목은 앞자리만 보이도록 처리하세요.", ["주민등록번호"]),
    ("해외 지사 {사람 이름} 연락처는 {전화번호}이며 내선 4567은 공용입니다.", ["사람 이름", "전화번호"]),
    ("{회사명} 법인카드 {신용카드번호} 사용 내역과 승인번호 98765432를 대조해 주세요.",
     ["회사명", "신용카드번호"]),
    ("참조: {사람 이름} {직책}, {사람 이름} {직책} (두 분 모두 {회사명} 소속)",
     ["사람 이름", "직책", "사람 이름", "직책", "회사명"]),
]


def make_example(rng: random.Random, gens: dict) -> dict:
    roll = rng.random()
    if roll < 0.08:
        text, entities, difficulty = rng.choice(NEGATIVES), [], "easy"
    else:
        if roll < 0.23:  # ~15% ambiguous/lookalike cases
            template, slots = rng.choice(HARD_TEMPLATES)
            difficulty = "hard"
        else:
            # oversample templates carrying rare types so per-type support in
            # the fixed gold split stays statistically usable (>=25)
            rare = {"운전면허번호", "외국인등록번호", "신용카드번호", "여권번호",
                    "사업자등록번호", "우편번호", "주민등록번호"}
            weights = [1 + 3 * len(rare & set(slots)) for _, slots in TEMPLATES]
            template, slots = rng.choices(TEMPLATES, weights=weights, k=1)[0]
            difficulty = "easy"
        values, entities = [], []
        for slot in slots:
            v = gens[slot]()
            values.append(v)
            entities.append({"type": slot, "text": v})
        text = _fill(template, slots, values)
        # gold must be unambiguous: every entity string appears exactly once
        if any(text.count(e["text"]) != 1 for e in entities):
            return make_example(rng, gens)
    return {
        "messages": [
            {"role": "user", "content": INSTRUCTION + text},
            {"role": "assistant", "content": json.dumps({"entities": entities}, ensure_ascii=False)},
        ],
        "meta": {"difficulty": difficulty},
    }


def _fill(template: str, slots: list, values: list) -> str:
    out = template
    for slot, value in zip(slots, values):
        out = out.replace("{" + slot + "}", value, 1)
    return _resolve_josa(out)


def make_oot_example(rng: random.Random, gens: dict) -> dict:
    template, slots = rng.choice(OOT_TEMPLATES)
    values, entities = [], []
    for slot in slots:
        v = gens[slot]()
        values.append(v)
        entities.append({"type": slot, "text": v})
    text = _fill(template, slots, values)
    if any(text.count(e["text"]) != 1 for e in entities):
        return make_oot_example(rng, gens)
    return {
        "messages": [
            {"role": "user", "content": INSTRUCTION + text},
            {"role": "assistant", "content": json.dumps({"entities": entities}, ensure_ascii=False)},
        ],
        "meta": {"difficulty": "oot"},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="data/samples/korean_pii.jsonl")
    parser.add_argument("--oot", action="store_true",
                        help="generate from the held-out templates only (eval-only gold set)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    gens = make_generators(rng)
    seen, examples = set(), []
    while len(examples) < args.n:
        ex = make_oot_example(rng, gens) if args.oot else make_example(rng, gens)
        key = ex["messages"][0]["content"]
        if key in seen:
            continue
        seen.add(key)
        examples.append(ex)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    counts: dict = {}
    for ex in examples:
        for e in json.loads(ex["messages"][1]["content"])["entities"]:
            counts[e["type"]] = counts.get(e["type"], 0) + 1
    print(f"wrote {len(examples)} examples to {out}")
    print("entity counts:", json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
