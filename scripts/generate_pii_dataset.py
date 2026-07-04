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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlx_pipeline.pii_eval import INSTRUCTION, TYPES  # noqa: E402  single source of truth

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
BANKS = ["국민은행", "신한은행", "우리은행", "하나은행", "농협은행", "기업은행",
         "카카오뱅크", "토스뱅크"]
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def make_generators(rng: random.Random):
    def digits(n):
        return "".join(rng.choice("0123456789") for _ in range(n))

    def name():
        return rng.choice(SURNAMES) + rng.choice(GIVEN)

    def company():
        head = rng.choice(COMPANY_HEADS)
        form = rng.choice(["(주){h}{t}", "{h}{t} 주식회사", "{h}{t}", "주식회사 {h}{t}"])
        return form.format(h=head, t=rng.choice(COMPANY_TAILS))

    def rrn(foreign: bool = False):
        """달력상 유효한 생년월일 + 세기와 일치하는 성별 자리."""
        year = rng.randint(1955, 2005)
        month = rng.randint(1, 12)
        day = rng.randint(1, DAYS_IN_MONTH[month - 1])
        if foreign:
            gender = rng.choice("56") if year < 2000 else rng.choice("78")
        else:
            gender = rng.choice("12") if year < 2000 else rng.choice("34")
        return f"{year % 100:02d}{month:02d}{day:02d}-{gender}{digits(6)}"

    def phone():
        return rng.choice([
            f"010-{rng.randint(2000, 9999)}-{digits(4)}",
            f"02-{rng.randint(300, 999)}-{digits(4)}",
            f"031-{rng.randint(200, 999)}-{digits(4)}",
            f"+82-10-{rng.randint(2000, 9999)}-{digits(4)}",
            f"010{rng.randint(2000, 9999)}{digits(4)}",
        ])

    return {
        "사람 이름": name,
        "주민등록번호": rrn,
        "외국인등록번호": lambda: rrn(foreign=True),
        "여권번호": lambda: f"{rng.choice('MS')}{digits(8)}",
        "운전면허번호": lambda: f"{rng.choice(['11','12','13','14','15','16','21','22'])}-{rng.randint(10, 26):02d}-{digits(6)}-{digits(2)}",
        "전화번호": phone,
        "이메일 주소": lambda: f"{rng.choice('abcdefghjkmnpr')}{rng.choice(['lee','kim','park','choi','jung'])}{digits(2)}@{rng.choice(DOMAINS)}",
        "계좌번호": lambda: rng.choice([
            f"{digits(3)}-{digits(3)}-{digits(6)}", f"{digits(4)}-{digits(2)}-{digits(7)}",
            f"{rng.choice(BANKS)} {digits(3)}-{digits(2)}-{digits(6)}",
        ]),
        "신용카드번호": lambda: rng.choice([
            f"{rng.randint(3, 9)}{digits(3)}-{digits(4)}-{digits(4)}-{digits(4)}",
            f"{rng.randint(3, 9)}{digits(3)} {digits(4)} {digits(4)} {digits(4)}",
        ]),
        "사업자등록번호": lambda: f"{rng.randint(101, 899)}-{rng.randint(10, 99)}-{digits(5)}",
        "주소": lambda: f"{rng.choice(CITIES)} {rng.choice(ROADS)} {rng.randint(1, 400)}" + rng.choice(["", f" {rng.randint(1,25)}층", f" {rng.randint(101,2504)}호"]),
        "우편번호": lambda: f"{rng.randint(1000, 63644):05d}",
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
    # ── register variety: SMS/알림톡 ──
    ("[Web발신] {사람 이름}님, {신용카드번호} 카드로 38,500원 결제 완료.", ["사람 이름", "신용카드번호"]),
    ("(광고) {회사명}입니다! 상담 예약: {전화번호} 수신거부 080-000-0000", ["회사명", "전화번호"]),
    ("택배 도착 안내 - {주소} 문 앞 보관. 문의 {전화번호}", ["주소", "전화번호"]),
    # ── 채팅체/반말 ──
    ("야 {사람 이름} 번호 {전화번호} 맞지? 메일도 알려줘 {이메일 주소}로 보내면 돼?", ["사람 이름", "전화번호", "이메일 주소"]),
    ("나 이사감 ㅋㅋ 새 주소 {주소}임 우편번호는 {우편번호}", ["주소", "우편번호"]),
    ("월급 계좌 바꿈. {계좌번호}로 이체 부탁", ["계좌번호"]),
    # ── 공문서체 ──
    ("위 본인 {사람 이름}(주민등록번호 {주민등록번호})은 상기 내용이 사실과 다름없음을 확인합니다.",
     ["사람 이름", "주민등록번호"]),
    ("제2026-45호 관련, 귀사({회사명}, 사업자등록번호 {사업자등록번호})에 아래와 같이 통보합니다.",
     ["회사명", "사업자등록번호"]),
    ("신청인 {사람 이름}의 여권번호는 {여권번호}이며 유효기간 내 서류임을 증명합니다.", ["사람 이름", "여권번호"]),
    # ── 폼/표 서식 ──
    ("이름: {사람 이름}\n연락처: {전화번호}\n이메일: {이메일 주소}\n주소: {주소}",
     ["사람 이름", "전화번호", "이메일 주소", "주소"]),
    ("■ 예금주: {사람 이름} ■ 은행/계좌: {계좌번호} ■ 연락처: {전화번호}", ["사람 이름", "계좌번호", "전화번호"]),
    ("성명(한글): {사람 이름} | 외국인등록번호: {외국인등록번호} | 우편번호: {우편번호}",
     ["사람 이름", "외국인등록번호", "우편번호"]),
    ("차량 인수 확인서 — 인수인: {사람 이름} / 면허: {운전면허번호} / 서명일: 2026-07-04",
     ["사람 이름", "운전면허번호"]),
    # ── 콜센터 스크립트 ──
    ("네 고객님, 본인확인을 위해 등록된 전화번호 {전화번호}[과와] 이메일 {이메일 주소} 맞으신가요?",
     ["전화번호", "이메일 주소"]),
    ("확인 감사합니다. 환불금은 {계좌번호} 계좌로 영업일 기준 3일 내 입금됩니다, {사람 이름} 고객님.",
     ["계좌번호", "사람 이름"]),
    # ── 사내 로그/시스템체 ──
    ("[2026-07-04 09:12] user={사번} name={사람 이름} action=LOGIN_FAILED ip=10.0.3.7",
     ["사번", "사람 이름"]),
    ("승인요청 #8812 | 기안자: {직책} {사람 이름} | 참조: {이메일 주소}", ["직책", "사람 이름", "이메일 주소"]),
    # ── 뉴스/보도자료체 ──
    ("{회사명}[은는] 4일, 신임 대표에 {사람 이름} 전 {직책}[을를] 선임했다고 밝혔다.",
     ["회사명", "사람 이름", "직책"]),
    ("경찰은 용의 차량 소유주 {사람 이름}(면허번호 {운전면허번호})[을를] 소환 조사할 예정이다.",
     ["사람 이름", "운전면허번호"]),
    # ── 이메일 전문체 ──
    ("{사람 이름} {직책}님께,\n첨부 계약서에 서명 후 {이메일 주소}로 회신 부탁드립니다.\n감사합니다.",
     ["사람 이름", "직책", "이메일 주소"]),
    ("안녕하세요, {회사명} {직책} {사람 이름}입니다. 세금계산서는 사업자번호 {사업자등록번호} 기준으로 발행되었습니다.",
     ["회사명", "직책", "사람 이름", "사업자등록번호"]),
    # ── 예약/주문 확인 ──
    ("예약이 확정되었습니다. 예약자 {사람 이름}, 여권번호 {여권번호}, 탑승일 7월 20일.", ["사람 이름", "여권번호"]),
    ("주문 #20260704-88 결제수단 {신용카드번호}, 배송지 {주소} (우편번호 {우편번호})",
     ["신용카드번호", "주소", "우편번호"]),
    # ── 관례 교정: 이름 뒤 괄호 (이름(연락처), 이름(사번), 이름(메일)) ──
    ("문의 고객 {사람 이름}({전화번호})[이가] 재연락을 요청했습니다.", ["사람 이름", "전화번호"]),
    ("출입증 재발급 대상: {사람 이름}({사번}), 수령 장소는 1층 안내데스크입니다.", ["사람 이름", "사번"]),
    ("작성자 {사람 이름}({이메일 주소})의 승인 이력을 확인해 주세요.", ["사람 이름", "이메일 주소"]),
    ("강사 {사람 이름}({전화번호}) 섭외 완료, 강의료는 {계좌번호}로 지급.", ["사람 이름", "전화번호", "계좌번호"]),
    # ── 관례 교정: 이름+의 (소유격) ──
    ("{사람 이름}의 비상연락망을 {전화번호}로 갱신했습니다.", ["사람 이름", "전화번호"]),
    ("{사람 이름}의 급여계좌({계좌번호}) 변경 신청이 접수되었습니다.", ["사람 이름", "계좌번호"]),
    ("신입사원 {사람 이름}의 사번은 {사번}로 발급되었습니다.", ["사람 이름", "사번"]),
    # ── 관례 교정: 담당/수식어 + 직책 (골드는 직책만) ──
    ("정산 담당 {직책} {사람 이름}에게 전달 완료했습니다.", ["직책", "사람 이름"]),
    ("행사 총괄 {직책} {사람 이름}[이가] 최종 승인 권한을 갖습니다.", ["직책", "사람 이름"]),
    ("업무 인계는 전임 {직책} {사람 이름}[과와] 협의하세요.", ["직책", "사람 이름"]),
    # ── 관례 교정: 부서명은 직책이 아님 ──
    ("총무팀 {직책} {사람 이름}[이가] 비품 신청을 승인했습니다.", ["직책", "사람 이름"]),
    ("개발팀 소속 {사람 이름} {직책}의 내선을 {전화번호}로 변경했습니다.", ["사람 이름", "직책", "전화번호"]),
    ("영업팀 {사람 이름}({사번}) 인사평가 면담이 예정되어 있습니다.", ["사람 이름", "사번"]),
    # ── 관례 교정: 괄호 속 숫자-사번 (전화번호 아님) ──
    ("근태 오류 정정 요청 (사번 {사번}) 처리되었습니다. 문의는 {전화번호}.", ["사번", "전화번호"]),
    ("보안 교육 미이수자 (사번 {사번}) {사람 이름}, 기한 내 이수 바랍니다.", ["사번", "사람 이름"]),
    # ── 관례 교정: 괄호 속 여권/면허 (사번 아님) ──
    ("국제선 탑승 명단: {사람 이름}({여권번호}), 좌석 21C, 특별 기내식 신청됨", ["사람 이름", "여권번호"]),
    ("비자 발급 진행 상황 — 신청인 {사람 이름}({여권번호}) 서류 심사 중", ["사람 이름", "여권번호"]),
    ("차량 인수 대장: {사람 이름}({운전면허번호}) 서명 완료, 반납일 미정", ["사람 이름", "운전면허번호"]),
    # ── 관례 교정: 역할 단어(담당/상담사/멘토/예금주 등)는 직책이 아님 ──
    ("이번 교육 진행은 외부 강사 {사람 이름}[이가] 맡고, 문의는 {전화번호}로.", ["사람 이름", "전화번호"]),
    ("신규 입사자 멘토로 {사람 이름}[이가] 지정되었습니다. 사번 {사번}.", ["사람 이름", "사번"]),
    ("상담사 {사람 이름} 연결이 지연되고 있습니다. 예상 대기 3분.", ["사람 이름"]),
    ("예금주 {사람 이름} 명의의 {계좌번호} 계좌로 이체되었습니다.", ["사람 이름", "계좌번호"]),
    # ── 프레임 커버리지: 슬래시 헤더(부서 포함), 문장 끝 엔티티 ──
    ("결재선: {회사명} 재무팀 / 기안: {사람 이름} {직책} / 협조: 법무팀", ["회사명", "사람 이름", "직책"]),
    ("받는이: 구매팀 / 보내는이: {직책} {사람 이름} / 회신처: {이메일 주소}", ["직책", "사람 이름", "이메일 주소"]),
    ("점검 결과 이상 없음. 추가 문의는 시설관리 {사람 이름}, {전화번호}", ["사람 이름", "전화번호"]),
    ("상담 종료 후 만족도 조사 링크가 발송됩니다. 수신 번호: {전화번호}", ["전화번호"]),
    ("서류 미비 시 반송됩니다. 반송지: {주소}, 우편번호 {우편번호}", ["주소", "우편번호"]),
    ("긴급 상황 시 아래 번호로 즉시 연락 바랍니다: {전화번호}", ["전화번호"]),
    # ── 직책 정밀 교정: 헤더 프레임에서 회사+부서(인사팀 등)는 골드가 아님 ──
    ("발신처: {회사명} 인사팀 | 수신처: {직책} {사람 이름} | 건명: 경비 정산 승인",
     ["회사명", "직책", "사람 이름"]),
    ("수신처: {회사명} 지원팀 / 참조: {사람 이름} {직책} / 건명: 단가 협의 요청",
     ["회사명", "사람 이름", "직책"]),
    ("문서번호 HR-2211 / 기안 부서: 인사팀 / 승인자: {직책} {사람 이름} / 시행일: 즉시",
     ["직책", "사람 이름"]),
    # ── 직책 정밀 교정: 쉼표 나열 꼬리의 담당 직책+이름 ──
    ("협력사 실사 완료: {회사명}, 대표전화 {전화번호}, 담당 {직책} {사람 이름}",
     ["회사명", "전화번호", "직책", "사람 이름"]),
    ("입점 심사 접수 — 상호 {회사명}, 사업자 {사업자등록번호}, 담당 {직책} {사람 이름}",
     ["회사명", "사업자등록번호", "직책", "사람 이름"]),
    ("발주 확정: 품목 A-102, 납기 7/20, 인수 담당 {직책} {사람 이름}, 연락처 {전화번호}",
     ["직책", "사람 이름", "전화번호"]),
    # ── 프레임 다양성 확장: 견적/이력서/메신저/영수증/회의록/계약/라벨/알림 ──
    ("[견적서] 공급자: {회사명} ({사업자등록번호}) / 견적 담당: {사람 이름} / 유효기간: 30일",
     ["회사명", "사업자등록번호", "사람 이름"]),
    ("경력사항: {회사명} {직책} (2019–2024) — 자기소개서 문의: {이메일 주소}",
     ["회사명", "직책", "이메일 주소"]),
    ("김대리> 고객 콜백 요청이요 / 누구?> {사람 이름}님, {전화번호} / 김대리> 확인!",
     ["사람 이름", "전화번호"]),
    ("[영수증] 가맹점: {회사명} / 승인카드: {신용카드번호} / 금액: 47,000원",
     ["회사명", "신용카드번호"]),
    ("회의록 v2 — 참석: {사람 이름}({직책}), {사람 이름}({직책}) / 서기: 미정",
     ["사람 이름", "직책", "사람 이름", "직책"]),
    ("제3조(통지) 모든 통지는 갑의 주소({주소})로 서면 발송한다.", ["주소"]),
    ("갑: {회사명} (대표 {사람 이름}) / 을: 개인사업자 {사업자등록번호}",
     ["회사명", "사람 이름", "사업자등록번호"]),
    ("[배송라벨] 받는분 {사람 이름} ★파손주의★ {주소} ({우편번호})",
     ["사람 이름", "주소", "우편번호"]),
    ("시스템 알림 ▶ 계정 {이메일 주소} 로그인 실패 5회 → 잠금 처리됨", ["이메일 주소"]),
    ("Q. 환불 계좌를 바꾸고 싶어요 → A. 마이페이지에서 {계좌번호} 형식으로 입력하세요.", ["계좌번호"]),
    ("출석부: 1) {사람 이름} 2) {사람 이름} 3) 불참 — 강사 연락처 {전화번호}",
     ["사람 이름", "사람 이름", "전화번호"]),
    ("여행자 명단 ✈ {사람 이름} (여권 {여권번호}) · {사람 이름} (여권 {여권번호})",
     ["사람 이름", "여권번호", "사람 이름", "여권번호"]),
    ("면허 갱신 대상자 안내 ▷ 성명 {사람 이름} ▷ 면허번호 {운전면허번호} ▷ 기한 내 방문",
     ["사람 이름", "운전면허번호"]),
    ("보증금 반환 처리됨 ✔ 수취 계좌 {계좌번호} ✔ 예금주 {사람 이름} ✔ 영수 완료",
     ["계좌번호", "사람 이름"]),
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
    # v2 확장: OOT 문형 다양화 (골드셋 반복 완화, 여전히 학습에서 격리)
    ("연말정산 기부금 영수증은 {이메일 주소}로 발급되며, 문의는 상담사 {사람 이름}에게.", ["이메일 주소", "사람 이름"]),
    ("숙소 체크인 명단 - {사람 이름}({여권번호}), 동반 1인, 차량 면허 {운전면허번호} 등록됨",
     ["사람 이름", "여권번호", "운전면허번호"]),
    ("자동이체 실패 알림: {계좌번호} 잔액 부족. {전화번호}로 재등록 바랍니다.", ["계좌번호", "전화번호"]),
    ("신원조회 요청서 — 대상자 {사람 이름}, 주민등록번호 {주민등록번호}, 결과 회신처 {이메일 주소}",
     ["사람 이름", "주민등록번호", "이메일 주소"]),
    ("법인 등기부등본상 본점 소재지가 {주소}(우편번호 {우편번호})로 이전되었습니다.", ["주소", "우편번호"]),
    ("급여명세서 수령 동의: 사번 {사번}, 성명 {사람 이름}, 수신 이메일 {이메일 주소}",
     ["사번", "사람 이름", "이메일 주소"]),
    ("출입국 기록 조회 대상: 외국인등록번호 {외국인등록번호}, 체류지 {주소}", ["외국인등록번호", "주소"]),
    ("가맹 계약 갱신 안내 — {회사명}(사업자 {사업자등록번호}), 담당 {사람 이름} {직책}",
     ["회사명", "사업자등록번호", "사람 이름", "직책"]),
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
    "인사팀에서 하반기 조직개편 안내 메일을 발송했습니다.",
    "총무팀과 재무팀이 예산 조정 회의를 진행합니다.",
    "개발팀 스프린트 회고는 금요일 오후로 옮겨졌습니다.",
    "재직증명서 발급은 인사팀 포털에서 셀프로 신청 가능합니다.",
    "수신: 전 임직원 / 발신: 지원팀 / 제목: 사옥 주차장 이용 안내",
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


TYPE_TO_TEMPLATES = None  # built lazily: type -> templates containing it


def make_example(rng: random.Random, gens: dict, type_counts: dict | None = None) -> dict:
    global TYPE_TO_TEMPLATES
    if TYPE_TO_TEMPLATES is None:
        TYPE_TO_TEMPLATES = {t: [tpl for tpl in TEMPLATES if t in tpl[1]] for t in TYPES}
    type_counts = type_counts if type_counts is not None else {}
    roll = rng.random()
    if roll < 0.08:
        text, entities, difficulty = rng.choice(NEGATIVES), [], "easy"
    else:
        if roll < 0.23:  # ~15% ambiguous/lookalike cases
            template, slots = rng.choice(HARD_TEMPLATES)
            difficulty = "hard"
        else:
            # per-type balancing: always serve the currently least-covered
            # type, so no type's calibration starves as templates are added
            target = min(TYPES, key=lambda t: type_counts.get(t, 0))
            template, slots = rng.choice(TYPE_TO_TEMPLATES[target])
            difficulty = "easy"
        values, entities = [], []
        for slot in slots:
            v = gens[slot]()
            values.append(v)
            entities.append({"type": slot, "text": v})
        text = _fill(template, slots, values)
        # gold must be unambiguous: every entity string appears exactly once
        if any(text.count(e["text"]) != 1 for e in entities):
            return make_example(rng, gens, type_counts)
    for e in entities:
        type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
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
    seen, examples, type_counts = set(), [], {}
    while len(examples) < args.n:
        ex = make_oot_example(rng, gens) if args.oot else make_example(rng, gens, type_counts)
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
