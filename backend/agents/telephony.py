"""Real outbound phone calls via Twilio (optional).

Uses Twilio's <Gather input="speech"> webhook loop: Twilio does the telephony
speech recognition and <Say> synthesis, our LLM decides what to say. Twilio
must reach the webhooks, so PUBLIC_URL must be a public tunnel to this server
(e.g. `ngrok http 8000`).
"""
import asyncio
import uuid
from xml.sax.saxutils import escape

import httpx

from .. import config

CALLS: dict[str, dict] = {}

CALLER_PROMPT = """You are an AI assistant on a real phone call on behalf of your user (name: Alex).
Objective: {objective}
Be brief and natural: 1-2 short sentences per turn. Say up front that you are an AI assistant.
When done (or if it's impossible), say goodbye and end your message with [END]."""


async def place_call(session, number: str, objective: str, timeout_s: int = 300) -> dict:
    from .caller import twilio_ready

    ok, reason = twilio_ready()
    if not ok:
        raise RuntimeError(f"Twilio not configured: {reason}")
    call_id = uuid.uuid4().hex[:12]
    CALLS[call_id] = {"session": session, "objective": objective, "history": [], "transcript": [], "done": asyncio.Event()}
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{config.TWILIO_ACCOUNT_SID}/Calls.json",
            auth=(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN),
            data={
                "To": number,
                "From": config.TWILIO_FROM_NUMBER,
                "Url": f"{config.PUBLIC_URL}/twilio/voice/{call_id}",
                "StatusCallback": f"{config.PUBLIC_URL}/twilio/status/{call_id}",
            },
        )
        r.raise_for_status()
    await session.event("call", state="ringing", number=number, callee=number, mode="twilio")
    try:
        await asyncio.wait_for(CALLS[call_id]["done"].wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        pass
    call = CALLS.pop(call_id)
    await session.event("call", state="ended", number=number, callee=number)
    return {"callee": number, "transcript": call["transcript"]}


async def _next_line(call_id: str, heard: str | None) -> tuple[str, bool]:
    call = CALLS[call_id]
    s = call["session"]
    if heard:
        call["transcript"].append({"speaker": "callee", "text": heard})
        call["history"].append({"role": "user", "content": heard})
        await s.send({"type": "assistant", "role": "callee", "text": heard})
    elif not call["history"]:
        call["history"].append({"role": "user", "content": "(the call was just answered)"})
    line = (await s.llm().complete(CALLER_PROMPT.format(objective=call["objective"]), call["history"]))["content"]
    ended = "[END]" in line
    line = line.replace("[END]", "").strip()
    call["history"].append({"role": "assistant", "content": line})
    call["transcript"].append({"speaker": "caller", "text": line})
    await s.send({"type": "assistant", "role": "caller", "text": line})
    return line, ended


def _twiml(call_id: str, line: str, ended: bool) -> str:
    if ended:
        return f"<Response><Say>{escape(line)}</Say><Hangup/></Response>"
    action = f"{config.PUBLIC_URL}/twilio/turn/{call_id}"
    return (
        f'<Response><Gather input="speech" speechTimeout="auto" action="{action}">'
        f"<Say>{escape(line)}</Say></Gather><Redirect>{action}</Redirect></Response>"
    )


async def on_answer(call_id: str) -> str:
    if call_id not in CALLS:
        return "<Response><Hangup/></Response>"
    await CALLS[call_id]["session"].event("call", state="connected")
    return _twiml(call_id, *await _next_line(call_id, None))


async def on_turn(call_id: str, speech: str | None) -> str:
    if call_id not in CALLS:
        return "<Response><Hangup/></Response>"
    return _twiml(call_id, *await _next_line(call_id, speech or "(silence)"))


def on_status(call_id: str, status: str):
    if call_id in CALLS and status in ("completed", "failed", "busy", "no-answer", "canceled"):
        CALLS[call_id]["done"].set()


async def place_call_vocalbridge(session, number: str, objective: str, timeout_s: int = 300) -> dict:
    """Outbound call through the course's platform. The Vocal Bridge agent (configured in its dashboard)
    does the talking; the objective is passed as session metadata, and we poll call logs for the transcript."""
    from .. import vocalbridge

    result = await vocalbridge.call(number, "Lab caller", session_metadata={"objective": objective})
    await session.event("call", state="ringing", number=number, callee=number, mode="vocalbridge", call_id=result.get("call_id"))
    started = asyncio.get_running_loop().time()
    seen_id = None
    while asyncio.get_running_loop().time() - started < timeout_s:
        await asyncio.sleep(5)
        sessions = (await vocalbridge.logs(5)).get("sessions", [])
        room = result.get("room_name")
        # prefer the session for this room; otherwise the newest outbound session
        match = next((x for x in sessions if room and room in str(x)), None) or next(
            (x for x in sessions if x.get("call_direction") == "outbound"), None
        )
        if not match:
            continue
        seen_id = match["id"]
        if match.get("status") in ("completed", "failed"):
            break
    transcript = []
    if seen_id:
        detail = await vocalbridge.log(seen_id)
        for m in detail.get("transcript") or []:
            speaker = "caller" if m.get("role") in ("assistant", "agent") else "callee"
            text = m.get("content") or m.get("text") or ""
            transcript.append({"speaker": speaker, "text": text})
            await session.send({"type": "assistant", "role": speaker, "text": text})
        if not transcript and detail.get("transcript_text"):
            transcript.append({"speaker": "caller", "text": detail["transcript_text"]})
    await session.event("call", state="ended", number=number, callee=number, session_id=seen_id)
    return {"callee": number, "transcript": transcript}
