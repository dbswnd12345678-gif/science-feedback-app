from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env", override=True)

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from graph import build_agent, EvalState
from report import generate_report
from tools import get_student_obs_count

# 전역 에이전트 (lifespan에서 초기화)
agent = None
_checkpointer = None

# 학생별 관찰 횟수 카운터 (서버 메모리 내 보관, 재시작 시 초기화)
_obs_counters: dict[str, int] = {}


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


@app.get("/report/{student_id}")
async def report_endpoint(student_id: str):
    """학생의 10회 관찰 종합 보고서 데이터를 JSON으로 반환합니다."""
    data = await generate_report(student_id)
    return JSONResponse(content=data)


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

            # 학번을 config로 분리 — HumanMessage에 학번을 포함하면 LLM이 인사말에서 학번을 사용함
            config = {"configurable": {"thread_id": student_id, "student_id": student_id}}

            # 학생별 관찰 횟수: 서버 재시작 후 첫 접속 시 Google Sheets에서 복원
            if student_id not in _obs_counters:
                _obs_counters[student_id] = get_student_obs_count(student_id)
            _obs_counters[student_id] += 1
            obs_num = _obs_counters[student_id]

            # 3-노드 StateGraph에 전달할 초기 상태
            initial_state: EvalState = {
                "messages": [],          # add_messages reducer가 누적, 초기에는 빈 리스트
                "student_id": student_id,
                "obs_num": obs_num,
                "observation_text": observation,
                "feedback_strategy": "",
                "level_score": 0,
                "objectivity_scores": [0],
                "sense": "",
                "method": "",
                "measurement": "",
                "time_dim": "",
                "comparison": "",
                "scope": "",
                "early_exit": False,
            }

            await websocket.send_json({"type": "start", "obs_num": obs_num})

            async def stream(cfg):
                """
                3-노드 StateGraph 스트림을 처리합니다.
                - Node 1 (node_evaluate): Tool 호출 배지 표시, 평가 텍스트 스트리밍
                - Node 2 (node_feedback): 피드백 텍스트 스트리밍
                - Node 3 (node_record) 시작 시점에 done을 먼저 전송, 기록은 백그라운드 계속 실행
                반환값: early_done_sent (bool)
                """
                has_response_text = False
                early_done_sent = False

                async for event in agent.astream_events(
                    initial_state,
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

                    elif kind == "on_chain_start" and event.get("name") == "node_record":
                        # node_record 시작 = 피드백 생성 완료 → done 조기 전송
                        if has_response_text and not early_done_sent:
                            await websocket.send_json({"type": "done"})
                            early_done_sent = True

                    elif kind == "on_tool_start" and not early_done_sent:
                        # done 전송 전에만 Tool 배지 표시 (Node 1의 평가 Tool만 해당)
                        await websocket.send_json({"type": "tool_start", "tool": event.get("name", "")})

                    elif kind == "on_tool_end" and not early_done_sent:
                        await websocket.send_json({"type": "tool_end", "tool": event.get("name", "")})

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
