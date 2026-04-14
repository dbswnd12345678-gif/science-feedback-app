from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env", override=True)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage

from graph import agent, _memory

app = FastAPI(title="과학 관찰 피드백 앱")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


def _clear_thread(thread_id: str):
    """MemorySaver에서 해당 thread의 오염된 상태를 삭제합니다."""
    try:
        if hasattr(_memory, "storage") and thread_id in _memory.storage:
            del _memory.storage[thread_id]
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
                    # 오염된 대화 이력 초기화 후 재시도
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
