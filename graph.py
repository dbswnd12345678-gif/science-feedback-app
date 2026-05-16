from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env", override=True)

import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from prompts import NODE1_PROMPT, NODE2_PROMPT
from tools import (
    get_scoring_criteria,
    get_classification_criteria,
    write_to_sheet,
    get_student_memory,
    update_student_memory,
)


# ── State ─────────────────────────────────────────────────────────────────────

class EvalState(TypedDict):
    messages: Annotated[list, add_messages]  # 대화 이력 (checkpointer 저장)
    student_id: str
    obs_num: int
    observation_text: str
    feedback_strategy: str      # Node 1에서 추출 → Node 2 피드백 근거 (다차원 전략 텍스트)
    level_score: int
    objectivity_scores: list    # 독립 정보별 객관도 점수 리스트 (예: [2, 2])
    sense: str
    method: str
    measurement: str
    time_dim: str               # 'time' 은 Python 예약어 충돌 방지를 위해 time_dim 사용
    comparison: str
    scope: str
    early_exit: bool            # 객관도 합계 0이면 True → Node 2·3 건너뜀


# ── Extraction schema ─────────────────────────────────────────────────────────

class EvalExtraction(BaseModel):
    """Node 1 평가 결과에서 구조화 데이터를 추출하기 위한 스키마."""
    level_score: int = Field(default=0, description="관찰의 수준 점수")
    objectivity_scores: list[int] = Field(
        default=[0],
        description=(
            "독립 정보별 객관도 점수 리스트. "
            "LE_n=1이면 1개, LE_n=2이면 2개, LE_n=3이면 3개 점수. "
            "과학적 관찰이 아니면 [0]."
        ),
    )
    sense: str = Field(default="시각", description="감각 분류: 시각/청각/후각/미각/촉각")
    method: str = Field(default="단순", description="방법 분류: 단순/조작")
    measurement: str = Field(default="정성", description="측정 분류: 정성/정량")
    time_dim: str = Field(default="시간독립", description="시간 분류: 시간독립/시간종속")
    comparison: str = Field(default="단수", description="비교 분류: 단수/다수")
    scope: str = Field(default="전체", description="범위 분류: 전체/부분")
    feedback_strategy: str = Field(
        default="",
        description=(
            "다차원 피드백 전략 텍스트. "
            "전략1(관찰 다양도 확대)·전략2(깊이 발전)·전략3(반복 패턴 탈출) 3가지를 포함. "
            "이번 관찰 대상에서 실제로 가능한 유형만 포함."
        ),
    )
    early_exit: bool = Field(default=False, description="객관도 합계가 0이면 True, 아니면 False")


# ── Models ────────────────────────────────────────────────────────────────────

# 스트리밍 모델: Node 1 Tool 루프 + Node 2 피드백 생성에 사용
_model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)

# 비스트리밍 모델: 구조화 추출 전용 (on_chat_model_stream 이벤트 미발생 → 학생에게 노출 없음)
_extraction_model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0, streaming=False)

_eval_tools = [
    get_student_memory,
    get_scoring_criteria,
    get_classification_criteria,
]
_eval_tools_map = {t.name: t for t in _eval_tools}
_eval_model = _model.bind_tools(_eval_tools)
_extractor = _extraction_model.with_structured_output(EvalExtraction)


# ── Node 1: 평가 (Tool 루프 + 구조화 추출) ───────────────────────────────────

async def node_evaluate(state: EvalState, config: RunnableConfig) -> dict:
    """
    장기 기억 조회 → 배점 → 분류 기준 Tool 호출.
    스트리밍으로 인사말 + 점수 + 관찰 유형을 출력한다.
    추출 모델(비스트리밍)로 구조화 데이터를 state에 저장한다.
    """
    student_id = state["student_id"]
    obs_num = state["obs_num"]
    observation_text = state["observation_text"]

    node_msgs = [
        SystemMessage(content=NODE1_PROMPT),
        SystemMessage(content=(
            f"[세션 학번: {student_id}] — Tool 호출 시 student_id 파라미터에 이 값을 사용하세요. "
            "학번은 인사말이나 답변에 절대 출력하지 마세요."
        )),
        HumanMessage(content=f"[관찰 번호: {obs_num}번째]\n[관찰문]: {observation_text}"),
    ]

    # ReAct Tool 루프 (스트리밍 모델 → on_chat_model_stream 이벤트 발생)
    while True:
        response = await _eval_model.ainvoke(node_msgs, config=config)
        node_msgs.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            fn = _eval_tools_map.get(tc["name"])
            result = await fn.ainvoke(tc["args"]) if fn else "Tool not found."
            node_msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # 구조화 추출 (비스트리밍 → 학생에게 노출되지 않음)
    extraction: EvalExtraction = await _extractor.ainvoke(
        node_msgs + [HumanMessage(content="위 평가 결과에서 점수와 분류 데이터를 추출하세요.")],
        config=config,
    )

    return {
        # 대화 이력: 학생 메시지 + 평가 AI 응답만 저장 (Tool 중간 메시지 제외)
        "messages": [
            HumanMessage(content=f"[관찰 번호: {obs_num}번째]\n[관찰문]: {observation_text}"),
            node_msgs[-1],
        ],
        "feedback_strategy": extraction.feedback_strategy,
        "level_score": extraction.level_score,
        "objectivity_scores": extraction.objectivity_scores,
        "sense": extraction.sense,
        "method": extraction.method,
        "measurement": extraction.measurement,
        "time_dim": extraction.time_dim,
        "comparison": extraction.comparison,
        "scope": extraction.scope,
        "early_exit": extraction.early_exit,
    }


def route_after_evaluate(state: EvalState) -> str:
    """객관도 0점이면 종료, 아니면 피드백 노드로 라우팅."""
    return END if state.get("early_exit") else "node_feedback"


# ── Node 2: 피드백 출력 (Tool 없음, 형식 집중) ───────────────────────────────

async def node_feedback(state: EvalState, config: RunnableConfig) -> dict:
    """
    Node 1의 구조화 데이터를 바탕으로 피드백 섹션만 생성.
    짧고 집중된 프롬프트로 번호·이모지 형식 준수율을 높인다.
    """
    scores = state['objectivity_scores']
    context = (
        f"관찰문: {state['observation_text']}\n"
        f"관찰의 수준: {state['level_score']}점\n"
        f"관찰 지식의 객관도: {scores} (합계: {sum(scores)}점)\n"
        f"관찰 유형: <{state['sense']}, {state['method']}, {state['measurement']}> "
        f"<{state['scope']}, {state['comparison']}, {state['time_dim']}>\n"
        f"피드백 전략 (다음 관찰 방향):\n{state['feedback_strategy']}"
    )
    response = await _model.ainvoke(
        [SystemMessage(content=NODE2_PROMPT), HumanMessage(content=context)],
        config=config,
    )
    return {"messages": [response]}


# ── Node 3: 기록 (LLM 없이 직접 Tool 실행) ───────────────────────────────────

async def node_record(state: EvalState, config: RunnableConfig) -> dict:
    """
    스프레드시트 기록 + 장기 기억 업데이트를 병렬로 실행.
    LLM 호출 없이 Python에서 직접 Tool 함수를 실행한다.
    """
    student_id = state["student_id"]
    scores = state['objectivity_scores']
    summary = (
        f"관찰문: {state['observation_text']}, "
        f"수준 {state['level_score']}점, 객관도 {scores} (합계: {sum(scores)}점), "
        f"유형: <{state['sense']}, {state['method']}, {state['measurement']}> "
        f"<{state['scope']}, {state['comparison']}, {state['time_dim']}>, "
        f"피드백 전략: {state['feedback_strategy']}"
    )
    await asyncio.gather(
        write_to_sheet.ainvoke({
            "student_id": student_id,
            "observation_text": state["observation_text"],
            "level_score": state["level_score"],
            "objectivity_scores": state["objectivity_scores"],
            "sense": state["sense"],
            "method": state["method"],
            "measurement": state["measurement"],
            "time": state["time_dim"],
            "comparison": state["comparison"],
            "scope": state["scope"],
        }),
        update_student_memory.ainvoke({
            "student_id": student_id,
            "summary": summary,
        }),
    )
    return {}


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_agent(checkpointer):
    """3-노드 StateGraph를 생성하여 컴파일합니다."""
    graph = StateGraph(EvalState)

    graph.add_node("node_evaluate", node_evaluate)
    graph.add_node("node_feedback", node_feedback)
    graph.add_node("node_record", node_record)

    graph.add_edge(START, "node_evaluate")
    graph.add_conditional_edges(
        "node_evaluate",
        route_after_evaluate,
        {END: END, "node_feedback": "node_feedback"},
    )
    graph.add_edge("node_feedback", "node_record")
    graph.add_edge("node_record", END)

    return graph.compile(checkpointer=checkpointer)
