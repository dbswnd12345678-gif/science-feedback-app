from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env", override=True)

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage

from graph import build_agent

# 전역 에이전트 (lifespan에서 초기화)
agent = None
_checkpointer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 PostgreSQL 연결을 관리합니다."""
    global agent, _checkpointer

    db_url = os.environ.get("DATABASE_URL", "")

    if db_url:
        # 클라우드(Railway): PostgreSQL 기반 영속 메모리
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        async with AsyncPostgresSaver.from_conn_string(db_url) as cp:
            await cp.setup()  # 체크포인트 테이블 자동 생성
            _checkpointer = cp
            agent = build_agent(cp)
            yield
    else:
        # 로컬 개발: RAM 기반 임시 메모리 (DATABASE_URL 없을 때)
        from langgraph.checkpoint.memory import MemorySaver
        cp = MemorySaver()
        _checkpointer = cp
        agent = build_agent(cp)
        yield


app = FastAPI(title="과학 관찰 피드백 앱", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


def _clear_thread(thread_id: str):
    """MemorySaver 환경에서 오염된 thread 상태를 삭제합니다."""
    try:
        if hasattr(_checkpointer, "storage") and thread_id in _checkpointer.storage:
            del _checkpointer.storage[thread_id]
    except Exception:
        pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            student_id = str(data.get("student_id", "")).strip()
            observation = str(data.get("observation", "")).strip()

            if not student_id or not observation:
                await websocket.send_json({
                    "type": "error",
                    "content": "학번과 관찰 내용을 모두 입력해주세요.",
                })
                continue

            config = {"configurable": {"thread_id": student_id}}
            message = HumanMessage(
                content=f"[학번: {student_id}]\n[관찰문]: {observation}"
            )

            await websocket.send_json({"type": "start"})

            async def stream(cfg):
                async for event in agent.astream_events(
                    {"messages": [message]},
                    config=cfg,
                    version="v2",
                ):
                    kind = event["event"]
                    if kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        content = chunk.content
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    await websocket.send_json({"type": "chunk", "content": block["text"]})
                        elif isinstance(content, str) and content:
                            await websocket.send_json({"type": "chunk", "content": content})
                    elif kind == "on_tool_start":
                        await websocket.send_json({"type": "tool_start", "tool": event.get("name", "")})
                    elif kind == "on_tool_end":
                        await websocket.send_json({"type": "tool_end", "tool": event.get("name", "")})

            try:
                await stream(config)
            except Exception as e:
                error_msg = str(e)
                is_history_error = (
                    "INVALID_CHAT_HISTORY" in error_msg
                    or "tool_calls that do not have" in error_msg
                    or "InvalidChatHistory" in type(e).__name__
                )
                if is_history_error:
                    _clear_thread(student_id)
                    try:
                        await stream(config)
                    except Exception as retry_err:
                        await websocket.send_json({"type": "error", "content": str(retry_err)})
                else:
                    await websocket.send_json({"type": "error", "content": error_msg})

            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass
