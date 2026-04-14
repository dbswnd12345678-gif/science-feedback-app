from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env", override=True)

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from prompts import SYSTEM_PROMPT
from tools import (
    get_scoring_criteria,
    get_feedback_criteria,
    get_classification_criteria,
    write_to_sheet,
)

_model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
_memory = MemorySaver()

agent = create_react_agent(
    model=_model,
    tools=[
        get_scoring_criteria,
        get_feedback_criteria,
        get_classification_criteria,
        write_to_sheet,
    ],
    prompt=SYSTEM_PROMPT,
    checkpointer=_memory,
)
