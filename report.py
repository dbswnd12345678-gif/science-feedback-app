"""
학생 10회 관찰 종합 보고서 생성 모듈.

데이터 소스:
  - Google Sheets: 관찰문(O열), 수준 점수(P열), 객관도 점수(Q열),
                   타임스탬프(R열), 누적 관찰력 지수(S열), 분류 행렬(D6:K25)
  - Mem0: 장기 기억 패턴 요약
  - Claude: 성장 내러티브 생성 (단일 LLM 호출)
"""

from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage


# ── 행/열 레이블 (tools.py와 동일 순서) ─────────────────────────────────────

_ROW_LABELS: list[tuple[str, str, str]] = [
    ("시각", "단순", "정성"), ("시각", "단순", "정량"),
    ("시각", "조작", "정성"), ("시각", "조작", "정량"),
    ("청각", "단순", "정성"), ("청각", "단순", "정량"),
    ("청각", "조작", "정성"), ("청각", "조작", "정량"),
    ("후각", "단순", "정성"), ("후각", "단순", "정량"),
    ("후각", "조작", "정성"), ("후각", "조작", "정량"),
    ("미각", "단순", "정성"), ("미각", "단순", "정량"),
    ("미각", "조작", "정성"), ("미각", "조작", "정량"),
    ("촉각", "단순", "정성"), ("촉각", "단순", "정량"),
    ("촉각", "조작", "정성"), ("촉각", "조작", "정량"),
]

_COL_LABELS: list[tuple[str, str, str]] = [
    ("시간독립", "다수", "전체"), ("시간독립", "다수", "부분"),
    ("시간독립", "단수", "전체"), ("시간독립", "단수", "부분"),
    ("시간종속", "다수", "전체"), ("시간종속", "다수", "부분"),
    ("시간종속", "단수", "전체"), ("시간종속", "단수", "부분"),
]


def _safe_range(sheet_title: str, cell_range: str) -> str:
    return f"'{sheet_title}'!{cell_range}"


def _sheets_service():
    import json
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    sa_file = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "")

    if sa_json:
        info = json.loads(sa_json)
        creds = Credentials.from_service_account_info(info, scopes=scopes)
    elif sa_file.strip().startswith("{"):
        info = json.loads(sa_file)
        creds = Credentials.from_service_account_info(info, scopes=scopes)
    else:
        creds = Credentials.from_service_account_file(sa_file, scopes=scopes)

    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def _get_mem0_memories(student_id: str) -> str:
    api_key = os.environ.get("MEM0_API_KEY", "")
    if not api_key:
        return ""
    try:
        from mem0 import MemoryClient
        client = MemoryClient(api_key=api_key)
        results = client.get_all(user_id=student_id)
        if not results:
            return ""
        return "\n".join(f"- {r['memory']}" for r in results if "memory" in r)
    except Exception:
        return ""


async def generate_report(student_id: str) -> dict:
    """
    학생의 10회 관찰 종합 보고서 데이터를 생성하여 dict로 반환합니다.
    """
    spreadsheet_id = os.environ.get("GOOGLE_SHEETS_ID", "")

    observations: list[dict] = []
    matrix_nonzero: list[dict] = []
    error_msg: str | None = None

    # ── Google Sheets 데이터 조회 ─────────────────────────────────────────────
    try:
        service = _sheets_service()

        # O2:S11 — 관찰문(O), 수준 점수(P), 객관도 점수(Q), 타임스탬프(R), 관찰력 지수(S)
        obs_result = (
            service.spreadsheets()
            .values()
            .get(
                spreadsheetId=spreadsheet_id,
                range=_safe_range(student_id, "O2:S11"),
            )
            .execute()
        )
        rows = obs_result.get("values", [])
        for i, row in enumerate(rows):
            text = row[0].strip() if len(row) > 0 and row[0] else ""
            if not text:
                continue
            level = int(row[1]) if len(row) > 1 and row[1] != "" else None
            obj = int(row[2]) if len(row) > 2 and row[2] != "" else None
            timestamp = row[3] if len(row) > 3 else None
            oq_index = float(row[4]) if len(row) > 4 and row[4] != "" else None
            observations.append(
                {
                    "num": i + 1,
                    "text": text,
                    "level_score": level,
                    "objectivity_score": obj,
                    "timestamp": timestamp,
                    "oq_index": oq_index,
                }
            )

        # D6:K25 — 분류 행렬 (20행 × 8열)
        matrix_result = (
            service.spreadsheets()
            .values()
            .get(
                spreadsheetId=spreadsheet_id,
                range=_safe_range(student_id, "D6:K25"),
            )
            .execute()
        )
        matrix_rows = matrix_result.get("values", [])
        for r_idx, row_label in enumerate(_ROW_LABELS):
            row_data = matrix_rows[r_idx] if r_idx < len(matrix_rows) else []
            for c_idx, col_label in enumerate(_COL_LABELS):
                val = row_data[c_idx] if c_idx < len(row_data) else ""
                try:
                    count = int(val)
                except (ValueError, TypeError):
                    count = 0
                if count > 0:
                    matrix_nonzero.append(
                        {
                            "sense": row_label[0],
                            "method": row_label[1],
                            "measurement": row_label[2],
                            "time": col_label[0],
                            "comparison": col_label[1],
                            "scope": col_label[2],
                            "count": count,
                        }
                    )
    except Exception as e:
        error_msg = f"Google Sheets 조회 실패: {e}"

    # ── Mem0 장기 기억 조회 ───────────────────────────────────────────────────
    memories = _get_mem0_memories(student_id)

    # ── Claude 내러티브 생성 ──────────────────────────────────────────────────
    narrative = ""
    if observations:
        obs_lines = "\n".join(
            f"{o['num']}회: 수준 {o['level_score']}점, 객관도 {o['objectivity_score']}점 — {o['text']}"
            for o in observations
        )
        matrix_lines = "\n".join(
            f"<{m['sense']}, {m['method']}, {m['measurement']}> <{m['scope']}, {m['comparison']}, {m['time']}> × {m['count']}회"
            for m in sorted(matrix_nonzero, key=lambda x: -x["count"])[:10]
        )
        memory_section = f"\n[장기 기억 패턴]\n{memories}" if memories else ""

        prompt = f"""다음은 중학교 1학년 학생 {student_id}의 총 {len(observations)}회 과학적 관찰 평가 기록입니다.

[관찰 이력]
{obs_lines}

[자주 사용한 관찰 유형]
{matrix_lines}{memory_section}

위 기록을 바탕으로 이 학생의 관찰 능력 성장을 300자 이내로 요약해주세요.
- 점수 변화 추이(향상/정체/패턴)를 언급하세요.
- 자주 사용한 관찰 유형과 아직 시도하지 않은 유형을 언급하세요.
- 다음 관찰 단계를 위한 격려 메시지로 마무리하세요.
- 중학생이 이해할 수 있는 쉬운 언어로 작성하세요.
- 마크다운 헤더 없이 단락 형식으로 작성하세요."""

        try:
            model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
            resp = await model.ainvoke([HumanMessage(content=prompt)])
            narrative = resp.content
        except Exception as e:
            narrative = f"내러티브 생성 실패: {e}"

    return {
        "student_id": student_id,
        "total": len(observations),
        "observations": observations,
        "matrix": matrix_nonzero,
        "narrative": narrative,
        "error": error_msg,
    }
