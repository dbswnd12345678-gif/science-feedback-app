from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env", override=True)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage

from graph import agent

app = FastAPI(title="과학 관찰 피드백 앱")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


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

            try:
                async for event in agent.astream_events(
                    {"messages": [message]},
                    config=config,
                    version="v2",
                ):
                    kind = event["event"]

                    # 텍스트 청크 스트리밍
                    if kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if chunk.content:
                            content = chunk.content
                            # Claude는 리스트 형식으로 content를 반환하기도 함
                            if isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        await websocket.send_json({
                                            "type": "chunk",
                                            "content": block["text"],
                                        })
                            elif isinstance(content, str) and content:
                                await websocket.send_json({
                                    "type": "chunk",
                                    "content": content,
                                })

                    # Tool 호출 시작 알림
                    elif kind == "on_tool_start":
                        tool_name = event.get("name", "")
                        await websocket.send_json({
                            "type": "tool_start",
                            "tool": tool_name,
                        })

                    # Tool 완료 알림
                    elif kind == "on_tool_end":
                        tool_name = event.get("name", "")
                        await websocket.send_json({
                            "type": "tool_end",
                            "tool": tool_name,
                        })

            except Exception as agent_error:
                await websocket.send_json({
                    "type": "error",
                    "content": f"에이전트 오류: {str(agent_error)}",
                })

            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass
