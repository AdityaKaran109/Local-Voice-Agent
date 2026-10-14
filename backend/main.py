"""Local voice-agent lab server.  Run:  ./run.sh  (or .venv/bin/uvicorn backend.main:app --port 8765)"""
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI, Form, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config, providers, vocalbridge
from .agents import evals, telephony
from .agents.caller import CallerAgent, twilio_ready
from .agents.tictactoe import TicTacToe
from .agents.voice_agent import VoiceForAgent
from .pipeline import Agent, Session

app = FastAPI(title="Voice Agent Lab")

AGENTS = {"chat": Agent, "game": TicTacToe, "agent": VoiceForAgent, "caller": CallerAgent}


@app.get("/")
def index():
    return RedirectResponse("/static/index.html")


@app.get("/api/providers")
def get_providers():
    ok, reason = twilio_ready()
    vb_ok, vb_reason = vocalbridge.ready()
    status = lambda key, name: {"available": bool(key), "reason": "" if key else f"{name} not set"}  # noqa: E731
    return providers.catalog() | {
        "telephony": {"twilio": {"available": ok, "reason": reason}, "vocalbridge": {"available": vb_ok, "reason": vb_reason}},
        "realtime": {
            "gemini": status(config.GEMINI_API_KEY, "GEMINI_API_KEY") | {"model": config.GEMINI_LIVE_MODEL},
            "openai": status(config.OPENAI_API_KEY, "OPENAI_API_KEY") | {"model": config.OPENAI_REALTIME_MODEL},
        },
        "vocalbridge": {"available": vb_ok, "reason": vb_reason},
    }


@app.websocket("/ws/{mode}")
async def voice_ws(ws: WebSocket, mode: str):
    agent_cls = AGENTS.get(mode)
    if not agent_cls:
        await ws.close(code=4404)
        return
    await Session(ws, agent_cls()).run()


# --- Realtime voice-to-voice (OpenAI Realtime over WebRTC) ---
@app.post("/api/realtime/session")
async def realtime_session(request: Request):
    if not config.OPENAI_API_KEY:
        raise HTTPException(400, "OPENAI_API_KEY not set in .env")
    body = await request.json()
    session = {
        "type": "realtime",
        "model": config.OPENAI_REALTIME_MODEL,
        "instructions": body.get("instructions") or "You are a friendly, concise voice assistant.",
        "audio": {
            "input": {"transcription": {"model": "gpt-4o-mini-transcribe"}},
            "output": {"voice": body.get("voice") or "marin"},
        },
    }
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            "https://api.openai.com/v1/realtime/client_secrets",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
            json={"session": session},
        )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    return {"client_secret": r.json()["value"], "model": config.OPENAI_REALTIME_MODEL}


# --- Gemini Live (voice-to-voice over a browser WebSocket) ---
@app.post("/api/gemini-live/token")
async def gemini_live_token(request: Request):
    """Mint a single-use ephemeral token so the browser can open a Live API session without our API key."""
    if not config.GEMINI_API_KEY:
        raise HTTPException(400, "GEMINI_API_KEY not set in .env")
    now = datetime.now(timezone.utc)
    body = {
        "uses": 1,
        "expireTime": (now + timedelta(minutes=30)).isoformat().replace("+00:00", "Z"),
        "newSessionExpireTime": (now + timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
    }
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{config.GEMINI_BASE_URL}/auth_tokens", headers={"x-goog-api-key": config.GEMINI_API_KEY}, json=body)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    return {"token": r.json()["name"], "model": config.GEMINI_LIVE_MODEL}


# Tools a speech-native model can call (same tools as the lesson-4 support agent)
_tool_agent = None


@app.get("/api/tools")
def list_tools():
    from .agents.support_agent import TOOLS

    return [{k: t["function"][k] for k in ("name", "description", "parameters")} for t in TOOLS]


@app.post("/api/tools/{name}")
async def call_tool(name: str, request: Request):
    global _tool_agent
    from .agents.support_agent import TOOLS, SupportAgent

    if name not in {t["function"]["name"] for t in TOOLS}:
        raise HTTPException(404, f"unknown tool {name}")
    _tool_agent = _tool_agent or SupportAgent()
    return await getattr(_tool_agent, name)(**(await request.json()))


# --- Vocal Bridge (the course's platform) ---
async def _vb(coro):
    try:
        return await coro
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.post("/api/vocalbridge/token")
async def vb_token(request: Request):
    body = await request.json() if await request.body() else {}
    return await _vb(vocalbridge.token(body.get("participant_name") or "User", body.get("session_id")))


@app.get("/api/vocalbridge/agent")
async def vb_agent():
    return await _vb(vocalbridge.agent())


@app.get("/api/vocalbridge/logs")
async def vb_logs(limit: int = 20):
    return await _vb(vocalbridge.logs(limit))


@app.get("/api/vocalbridge/logs/{session_id}")
async def vb_log(session_id: str):
    return await _vb(vocalbridge.log(session_id))


@app.post("/api/vocalbridge/logs/{session_id}/eval")
async def vb_eval(session_id: str, request: Request):
    body = await request.json()
    return await _vb(vocalbridge.evaluate(session_id, body.get("objective"), body.get("scenario")))


@app.post("/api/vocalbridge/call")
async def vb_call(request: Request):
    body = await request.json()
    return await _vb(vocalbridge.call(body["phone_number"], body.get("participant_name")))


# Vocal Bridge "AI Agent mode": the voice agent delegates questions to OUR text agent (lesson 4 pattern).
_vb_agent_sessions: dict[str, list] = {}


@app.post("/api/agent/query")
async def agent_query(request: Request):
    from .agents.support_agent import SupportAgent
    from .providers import llm as llm_providers

    body = await request.json()
    history = _vb_agent_sessions.setdefault(body.get("session") or "default", [])
    history.append({"role": "user", "content": body["query"]})
    answer = await SupportAgent().chat(llm_providers.get(body.get("llm") or "ollama"), history)
    return {"answer": answer}


# --- Evals ---
@app.get("/api/evals/scenarios")
def eval_scenarios():
    from .agents.support_agent import SYSTEM_PROMPT

    return {"scenarios": evals.scenarios(), "default_prompt": SYSTEM_PROMPT}


@app.post("/api/evals/run")
async def eval_run(request: Request):
    return {"id": await evals.run(await request.json())}


@app.get("/api/evals/runs")
def eval_runs():
    return evals.list_runs()


@app.get("/api/evals/runs/{run_id}")
def eval_get(run_id: str):
    run = evals.get_run(run_id)
    if not run:
        raise HTTPException(404)
    return run


# --- Twilio webhooks (only used for real calls) ---
def twiml(xml: str) -> Response:
    return Response(content=xml, media_type="application/xml")


@app.post("/twilio/voice/{call_id}")
async def twilio_voice(call_id: str):
    return twiml(await telephony.on_answer(call_id))


@app.post("/twilio/turn/{call_id}")
async def twilio_turn(call_id: str, SpeechResult: str | None = Form(None)):
    return twiml(await telephony.on_turn(call_id, SpeechResult))


@app.post("/twilio/status/{call_id}")
async def twilio_status(call_id: str, CallStatus: str = Form("")):
    telephony.on_status(call_id, CallStatus)
    return Response(status_code=204)


app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="static")


@app.get("/favicon.ico")
def favicon():
    return FileResponse(config.FRONTEND_DIR / "favicon.svg", media_type="image/svg+xml")
