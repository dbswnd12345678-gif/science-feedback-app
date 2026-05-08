import os
import re
from datetime import datetime, timezone, timedelta
from langchain_core.tools import tool

_KST = timezone(timedelta(hours=9))


# ── Mem0 클라이언트 (장기 기억) ───────────────────────────────────────────────

def _get_mem0_client():
    """Mem0 클라이언트를 반환합니다. MEM0_API_KEY가 없으면 None을 반환합니다."""
    api_key = os.environ.get("MEM0_API_KEY", "")
    if not api_key:
        return None
    from mem0 import MemoryClient
    return MemoryClient(api_key=api_key)

# ── 분류 행렬 셀 주소 매핑 ──────────────────────────────────────────────────

ROW_MAP: dict[tuple, int] = {
    ("시각", "단순", "정성"): 6,  ("시각", "단순", "정량"): 7,
    ("시각", "조작", "정성"): 8,  ("시각", "조작", "정량"): 9,
    ("청각", "단순", "정성"): 10, ("청각", "단순", "정량"): 11,
    ("청각", "조작", "정성"): 12, ("청각", "조작", "정량"): 13,
    ("후각", "단순", "정성"): 14, ("후각", "단순", "정량"): 15,
    ("후각", "조작", "정성"): 16, ("후각", "조작", "정량"): 17,
    ("미각", "단순", "정성"): 18, ("미각", "단순", "정량"): 19,
    ("미각", "조작", "정성"): 20, ("미각", "조작", "정량"): 21,
    ("촉각", "단순", "정성"): 22, ("촉각", "단순", "정량"): 23,
    ("촉각", "조작", "정성"): 24, ("촉각", "조작", "정량"): 25,
}

COL_MAP: dict[tuple, str] = {
    ("시간독립", "다수", "전체"): "D", ("시간독립", "다수", "부분"): "E",
    ("시간독립", "단수", "전체"): "F", ("시간독립", "단수", "부분"): "G",
    ("시간종속", "다수", "전체"): "H", ("시간종속", "다수", "부분"): "I",
    ("시간종속", "단수", "전체"): "J", ("시간종속", "단수", "부분"): "K",
}

RESPONSE_SLOTS = [f"P{r}" for r in range(2, 12)]   # P2 ~ P11 (관찰 문장 열 기준)
TEMPLATE_SHEET_NAME = "시트1"


def _get_sheets_service():
    import json
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]

    sa_json_env = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    sa_file_env = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "")

    # GOOGLE_SERVICE_ACCOUNT_JSON 우선 (Railway 등 클라우드 환경)
    if sa_json_env:
        info = json.loads(sa_json_env)
        creds = Credentials.from_service_account_info(info, scopes=scopes)
    # GOOGLE_SERVICE_ACCOUNT_FILE이 JSON 내용 자체인 경우 자동 감지
    elif sa_file_env.strip().startswith("{"):
        info = json.loads(sa_file_env)
        creds = Credentials.from_service_account_info(info, scopes=scopes)
    else:
        # 로컬: JSON 파일 경로 사용
        creds = Credentials.from_service_account_file(sa_file_env, scopes=scopes)

    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def _safe_range(sheet_title: str, cell_range: str) -> str:
    """시트 이름에 특수문자가 있을 경우 작은따옴표로 감쌉니다."""
    return f"'{sheet_title}'!{cell_range}"


def _get_or_create_student_sheet(service, spreadsheet_id: str, student_id: str) -> str:
    """학생 탭이 없으면 시트1을 복제하여 생성합니다. 탭 이름(student_id)을 반환합니다."""
    meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheets = meta["sheets"]

    template_id = None
    for sheet in sheets:
        title = sheet["properties"]["title"]
        if title == TEMPLATE_SHEET_NAME:
            template_id = sheet["properties"]["sheetId"]
        if title == student_id:
            return student_id   # 이미 존재

    if template_id is None:
        raise ValueError(f"템플릿 시트 '{TEMPLATE_SHEET_NAME}'을 찾을 수 없습니다.")

    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={
            "requests": [{
                "duplicateSheet": {
                    "sourceSheetId": template_id,
                    "insertSheetIndex": len(sheets),
                    "newSheetName": student_id,
                }
            }]
        },
    ).execute()
    return student_id


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def get_scoring_criteria() -> str:
    """관찰의 수준 및 관찰 지식의 객관도 배점 기준과 판단 예시를 반환합니다."""
    return """
[관찰의 수준]
- 1점: 관찰을 통해 1차 정보를 알 수 있음
  예) "익힌 새우의 색은 붉다" → 대상의 총체적 속성, 기초 정보 (1차)
  예) "국화 기공은 타원형이다" → 기공의 모양에 관한 단일 1차 정보
- 2점: 관찰을 통해 2차 정보를 알 수 있음
  예) "익힌 새우의 붉은 껍질 속 색소체도 붉은 색이다"
      → 껍질 색(1차)을 인지한 뒤 색소체 위치·색까지 파악 (2차)
- n점: 관찰을 통해 n차 정보를 알 수 있음 (단계가 깊어질수록 고점)

[관찰 지식의 객관도]
- 0점: 공란 제출, 관찰이 아닌 응답, 주제와 관련 없는 응답
  예) "새우가 시간이 지나면서 먹음직스러워진다" → 개인 느낌, 과학적 관찰 아님
- 1점: 관찰 사실이지만 부정확하게 기술된 것
  예) "백합의 기공 속 엽록체는 크다" → 비교 기준·수치 없어 모호
- 2점: 관찰 사실이고 정성적으로 기술된 것
  예) "몸 전체에 검은 반점이 있다" → 정성적 특성 기술
- 3점: 관찰 사실이고 정량적으로 기술된 것
  예) "가열 전 14cm이던 새우의 몸길이가 13.5cm로 0.5cm 줄어들었다" → 수치 데이터
"""


@tool
def get_feedback_criteria(objectivity_score: int) -> str:
    """
    객관도 점수(0~3)에 해당하는 피드백 기준을 반환합니다.
    이 기준을 바탕으로 학생에게 전달할 피드백을 작성하세요.
    """
    criteria: dict[int, str] = {
        0: (
            "· 사실과 다른 응답을 하였습니다.\n"
            "· 관찰과 관련 없는 응답을 하였습니다.\n"
            "· 과학적인 관찰을 시도해보세요.\n"
            "· 공란이거나 잘못 입력하였다면 다시 제출해주세요."
        ),
        1: (
            "· 탐구에는 오감이 사용될 수 있습니다.\n"
            "· 관찰 대상에 조작을 가하거나 가하지 않을 수도 있습니다.\n"
            "· 대상을 정성적이거나 정량적으로 관찰할 수 있습니다.\n"
            "· 대상의 전체 또는 부분을 관찰할 수 있습니다.\n"
            "· 단일 대상과 다수 대상을 관찰할 수 있습니다.\n"
            "· 시간에 따른 변화를 관찰할 수 있습니다."
        ),
        2: (
            "· 정성적인 관찰보다 정량적인 관찰을 권장합니다.\n"
            "· 비교할 때 대상과 기준을 명확히 언급해주세요.\n"
            "· 객관적인 표현을 권장합니다.\n"
            "· 수치 데이터를 제시하는 방법을 시도해보세요."
        ),
        3: (
            "· 정량적인 관찰을 하였습니다.\n"
            "· 비교할 때 대상과 기준을 명확히 언급하였습니다.\n"
            "· 객관적인 표현으로 응답하였습니다.\n"
            "· 수치 데이터를 제시하였습니다.\n"
            "· 선생님께 검토를 요청드리세요."
        ),
    }
    return criteria.get(objectivity_score, "알 수 없는 점수입니다. 0~3 사이의 값을 입력하세요.")


@tool
def get_classification_criteria() -> str:
    """관찰 유형 6차원 분류 기준과 분류 예시를 반환합니다."""
    return """
[분류 6차원 정의]
· 감각: 시각 / 청각 / 후각 / 미각 / 촉각
· 방법: 단순(대상을 그대로 관찰) / 조작(대상에 직접 조작을 가함)
· 측정: 정성(수치 없음) / 정량(수치·측정값 포함)
· 시간: 시간독립(정적 상태 관찰) / 시간종속(시간에 따른 변화 관찰)
· 비교: 단수(단일 대상) / 다수(여러 대상을 비교)
· 범위: 전체 / 부분

[분류 예시]
· "목련 겨울눈의 겉 부분에 솜털이 있다."
  → <시각, 단순, 정성> <부분, 단수, 시간독립>
· "칠엽수 겨울눈을 손으로 자르니 풋사과 냄새가 난다."
  → <후각, 조작, 정성> <전체, 단수, 시간독립>
· "산수유 겨울눈의 껍질은 거칠다."
  → <촉각, 단순, 정성> <부분, 단수, 시간독립>
· "철쭉 겨울눈 바깥쪽에 있는 덮개들은 크기가 모두 다르다."
  → <시각, 단순, 정성> <부분, 다수, 시간독립>
· "단풍나무 겨울눈의 길이가 1.5mm이다."
  → <시각, 단순, 정량> <전체, 단수, 시간독립>
· "알바트로스는 물갈퀴를 펴고 땅을 구르면서 앞으로 전진한다."
  → <시각, 단순, 정성> <전체, 단수, 시간종속>
· "알바트로스의 부리는 머리의 1.5배 정도 된다."
  → <시각, 단순, 정량> <부분, 다수, 시간독립>
· "알바트로스와 물수리의 나아가는 속력은 점점 빨라진다."
  → <시각, 단순, 정성> <전체, 다수, 시간종속>
· "물수리의 날개 끝이 갈라져 있다."
  → <시각, 단순, 정성> <부분, 단수, 시간독립>
"""


@tool
def write_to_sheet(
    student_id: str,
    observation_text: str,
    level_score: int,
    objectivity_score: int,
    sense: str,
    method: str,
    measurement: str,
    time: str,
    comparison: str,
    scope: str,
) -> str:
    """
    학생별 시트 탭에 다음을 기록합니다.
    1. 6차원 분류 행렬 셀 카운트 +1
    2. O열: 관찰 시각(KST), P열: 관찰 문장(태그 제외), Q열: 관찰 유형(태그)
    3. R열: 관찰의 수준, S열: 관찰 지식의 객관도, T열: 누적 관찰력 지수(OQ_n)
    4. 10번째 관찰 완료 시 S13에 레이블, T13에 최종 OQ_10 기록
    학생 탭이 없으면 시트1을 복제하여 자동 생성합니다.
    예외가 발생해도 항상 문자열을 반환합니다(LangGraph ToolMessage 보장).
    """
    try:
        spreadsheet_id = os.environ["GOOGLE_SHEETS_ID"]
        service = _get_sheets_service()

        # 1. 학생 탭 확보
        sheet_title = _get_or_create_student_sheet(service, spreadsheet_id, student_id)

        # 2. 행렬 셀 주소 계산
        row = ROW_MAP.get((sense, method, measurement))
        col = COL_MAP.get((time, comparison, scope))
        if row is None or col is None:
            return (
                f"오류: 분류 조합을 찾을 수 없습니다. "
                f"입력값 — sense={sense}, method={method}, measurement={measurement}, "
                f"time={time}, comparison={comparison}, scope={scope}"
            )
        matrix_cell = f"{col}{row}"

        # 3. 전체 분류 행렬(D6:K25) 읽기
        #    - current_count: 현재 관찰의 행렬 셀 기존 값
        #    - non_zero: D_n 계산용 (non-zero 셀 수)
        _row_keys = list(ROW_MAP.keys())
        _col_keys = list(COL_MAP.keys())
        row_idx = _row_keys.index((sense, method, measurement))
        col_idx = _col_keys.index((time, comparison, scope))

        matrix_result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=_safe_range(sheet_title, "D6:K25"))
            .execute()
        )
        matrix_values = matrix_result.get("values", [])

        # 현재 셀 값 추출
        row_data = matrix_values[row_idx] if row_idx < len(matrix_values) else []
        raw = row_data[col_idx] if col_idx < len(row_data) else ""
        try:
            current_count = int(raw)
        except (ValueError, TypeError):
            current_count = 0

        # D_n 계산: 기존 non-zero 셀 수 + 이번 관찰로 새 유형 추가 여부
        non_zero = 0
        for r in matrix_values:
            for v in r:
                try:
                    if int(v) > 0:
                        non_zero += 1
                except (ValueError, TypeError):
                    pass
        if current_count == 0:
            non_zero += 1   # 이번 관찰이 새 유형을 추가함

        # 4. O2:T11 읽기: next_slot 탐색 + 이전 관찰의 누적합 계산
        #    열 인덱스: 0=O(타임스탬프), 1=P(관찰문), 2=Q(태그), 3=R(수준), 4=S(객관도), 5=T(OQ)
        slot_result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=_safe_range(sheet_title, "O2:T11"))
            .execute()
        )
        slot_values = slot_result.get("values", [])
        next_row: int | None = None
        cumulative_sum = 0
        for i in range(10):
            row_s = slot_values[i] if i < len(slot_values) else []
            obs_text = row_s[1].strip() if len(row_s) > 1 else ""  # P열: 관찰 문장
            if not obs_text:
                next_row = i + 2   # 스프레드시트 행 번호 (2~11)
                break
            lv = int(row_s[3]) if len(row_s) > 3 and row_s[3] != "" else 0  # R열: 수준
            ob = int(row_s[4]) if len(row_s) > 4 and row_s[4] != "" else 0  # S열: 객관도
            cumulative_sum += lv * ob

        # 현재 관찰의 기여 추가
        cumulative_sum += level_score * objectivity_score
        oq_n = round(cumulative_sum * non_zero, 2)

        # 5. 배치 업데이트
        tag = f"<{sense}, {method}, {measurement}> <{scope}, {comparison}, {time}>"
        timestamp = datetime.now(_KST).strftime("%Y-%m-%d %H:%M")

        updates = [{"range": _safe_range(sheet_title, matrix_cell), "values": [[current_count + 1]]}]
        if next_row is not None:
            slot_row = str(next_row)
            updates.extend([
                {"range": _safe_range(sheet_title, f"O{slot_row}"), "values": [[timestamp]]},
                {"range": _safe_range(sheet_title, f"P{slot_row}"), "values": [[observation_text]]},
                {"range": _safe_range(sheet_title, f"Q{slot_row}"), "values": [[tag]]},
                {"range": _safe_range(sheet_title, f"R{slot_row}"), "values": [[level_score]]},
                {"range": _safe_range(sheet_title, f"S{slot_row}"), "values": [[objectivity_score]]},
                {"range": _safe_range(sheet_title, f"T{slot_row}"), "values": [[oq_n]]},
            ])
            # 10번째 관찰(행 11) 완료 시 최종 관찰력 지수 별도 기록
            if slot_row == "11":
                updates.extend([
                    {"range": _safe_range(sheet_title, "S13"), "values": [["최종 관찰력 지수"]]},
                    {"range": _safe_range(sheet_title, "T13"), "values": [[oq_n]]},
                ])

        service.spreadsheets().values().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"valueInputOption": "RAW", "data": updates},
        ).execute()

        slot_msg = f"P{next_row}행에 관찰 문장 저장 (OQ={oq_n})" if next_row else "응답 문장 슬롯이 가득 찼습니다(10개 초과)"
        return f"기록 완료 — 학생: {student_id}, 행렬 셀: {matrix_cell} → {current_count + 1}, {slot_msg}"

    except Exception as e:
        return f"스프레드시트 기록 실패: {type(e).__name__}: {str(e)[:200]}"


# ── 장기 기억 Tools (Mem0) ────────────────────────────────────────────────────

@tool
def get_student_memory(student_id: str) -> str:
    """
    학생의 장기 기억을 조회합니다.
    이전 관찰문 제출 패턴, 점수 변화, 강점과 약점 정보를 반환합니다.
    새 관찰문을 평가하기 전에 반드시 호출하여 개인화된 피드백을 준비하세요.
    MEM0_API_KEY가 설정되지 않은 경우 첫 번째 관찰로 처리됩니다.
    """
    try:
        client = _get_mem0_client()
        if client is None:
            return "장기 기억 저장소가 설정되지 않았습니다. 이번이 첫 번째 관찰 평가로 처리됩니다."

        results = client.search(
            query="학생의 관찰 패턴, 점수, 강점, 약점",
            user_id=student_id,
            limit=10,
        )
        if not results:
            return f"학생 {student_id}의 이전 기록이 없습니다. 첫 번째 관찰 평가입니다."

        memories = [r["memory"] for r in results if "memory" in r]
        return f"[학생 {student_id} 장기 기억]\n" + "\n".join(f"- {m}" for m in memories)

    except Exception as e:
        return f"장기 기억 조회 실패: {type(e).__name__}: {str(e)[:200]}"


@tool
def update_student_memory(student_id: str, summary: str) -> str:
    """
    이번 관찰 평가 결과를 학생의 장기 기억에 저장합니다.
    스프레드시트 기록(Step 4) 직후, 결과 반환(Step 5) 이전에 호출하세요.
    summary에는 이번 관찰의 점수, 분류 결과, 주요 특징, 개선이 필요한 점을 포함하세요.
    MEM0_API_KEY가 설정되지 않은 경우 저장을 건너뜁니다.
    """
    try:
        client = _get_mem0_client()
        if client is None:
            return "장기 기억 저장소가 설정되지 않아 저장을 건너뜁니다."

        client.add(summary, user_id=student_id)
        return f"학생 {student_id}의 장기 기억 업데이트 완료."

    except Exception as e:
        return f"장기 기억 저장 실패: {type(e).__name__}: {str(e)[:200]}"
