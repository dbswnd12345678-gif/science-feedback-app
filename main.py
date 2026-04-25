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

            # 기록 Tool: 학생 응답 전송 후 백그라운드에서 실행되어도 무방한 Tool
            BACKGROUND_TOOLS = {"write_to_sheet", "update_student_memory"}

            async def stream(cfg):
                """
                에이전트 스트림을 처리합니다.
                write_to_sheet / update_student_memory가 시작되는 시점에
                학생에게 done을 먼저 전송하고, 기록은 백그라운드에서 계속 실행합니다.
                반환값: early_done_sent (bool)
                """
                has_response_text = False   # 학생 응답 텍스트가 한 번이라도 전송됐는지
                early_done_sent = False     # done을 조기 전송했는지

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
                                if isinstance(block, dict) and block.get("type") == "text" and block["text"]:
                                    has_response_text = True
                                    await websocket.send_json({"type": "chunk", "content": block["text"]})
                        elif isinstance(content, str) and content:
                            has_response_text = True
                            await websocket.send_json({"type": "chunk", "content": content})

                    elif kind == "on_tool_start":
                        tool_name = event.get("name", "")
                        if tool_name in BACKGROUND_TOOLS:
                            # 학생 응답이 이미 전송된 경우 done을 먼저 보내고
                            # 기록 Tool은 백그라운드에서 계속 실행
                            if has_response_text and not early_done_sent:
                                await websocket.send_json({"type": "done"})
                                early_done_sent = True
                            await websocket.send_json({"type": "tool_start", "tool": tool_name})
                        else:
                            await websocket.send_json({"type": "tool_start", "tool": tool_name})

                    elif kind == "on_tool_end":
                        tool_name = event.get("name", "")
                        await websocket.send_json({"type": "tool_end", "tool": tool_name})

                return early_done_sent

            try:
                early_done = await stream(config)
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
                        early_done = await stream(config)
                    except Exception as retry_err:
                        await websocket.send_json({"type": "error", "content": str(retry_err)})
                        early_done = True  # 오류 시 done 중복 방지
                else:
                    await websocket.send_json({"type": "error", "content": error_msg})
                    early_done = True

            # 조기 전송하지 않은 경우에만 done 전송
            if not early_done:
                await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass
