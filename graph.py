from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env", override=True)

from langchain_anthropic import ChatAnthropic
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


def build_agent(checkpointer):
    """주어진 checkpointer로 LangGraph ReAct 에이전트를 생성합니다."""
    model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
    return create_react_agent(
        model=model,
        tools=_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
