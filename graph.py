from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env", override=True)

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from prompts import SYSTEM_PROMPT
from tools import (
    get_scoring_criteria,
    get_feedback_criteria,
    get_classification_criteria,
    write_to_sheet,
    get_student_memory,
    update_student_memory,
)

_TOOLS = [
    get_scoring_criteria,
    get_feedback_criteria,
    get_classification_criteria,
    write_to_sheet,
    get_student_memory,
    update_student_memory,
]


def make_prompt(state, config):
    """요청마다 학번을 SystemMessage로 주입해 HumanMessage에서 학번을 노출하지 않도록 합니다."""
    student_id = config.get("configurable", {}).get("student_id", "")
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=(
            f"[세션 학번: {student_id}]\n"
            "- Tool 호출 시 student_id 파라미터에 이 값을 사용하세요.\n"
            "- 학번은 인사말이나 답변에 절대 출력하지 마세요."
        )),
    ] + state["messages"]


def build_agent(checkpointer):
    """주어진 checkpointer로 LangGraph ReAct 에이전트를 생성합니다."""
    model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
    return create_react_agent(
        model=model,
        tools=_TOOLS,
        prompt=make_prompt,
        checkpointer=checkpointer,
    )
