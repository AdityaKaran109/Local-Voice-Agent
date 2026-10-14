"""Vocal Bridge — the platform the DeepLearning.AI course uses.

The API key stays on this server; the browser SDK (@vocalbridgeai/sdk) fetches
short-lived LiveKit connection tokens from /api/vocalbridge/token. Endpoints
mirror what the official `vb` CLI calls.
"""
import httpx

from . import config


def ready() -> tuple[bool, str]:
    return (True, "") if config.VOCAL_BRIDGE_API_KEY else (False, "VOCAL_BRIDGE_API_KEY not set")


async def request(method: str, path: str, json: dict | None = None, params: dict | None = None, timeout: int = 30) -> dict:
    ok, reason = ready()
    if not ok:
        raise RuntimeError(reason)
    headers = {"X-API-Key": config.VOCAL_BRIDGE_API_KEY, "Content-Type": "application/json"}
    if config.VOCAL_BRIDGE_AGENT_ID:
        headers["X-Agent-Id"] = config.VOCAL_BRIDGE_AGENT_ID
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.request(method, f"{config.VOCAL_BRIDGE_API_URL}{path}", headers=headers, json=json, params=params)
    if r.status_code >= 400:
        raise RuntimeError(f"Vocal Bridge {r.status_code}: {r.text[:300]}")
    return r.json() if r.content else {}


async def token(participant_name: str = "User", session_id: str | None = None) -> dict:
    body = {"participant_name": participant_name} | ({"session_id": session_id} if session_id else {})
    return await request("POST", "/api/v1/token", json=body)


async def agent() -> dict:
    return await request("GET", "/api/v1/agent")


async def logs(limit: int = 20) -> dict:
    return await request("GET", "/api/v1/logs", params={"limit": limit, "offset": 0})


async def log(session_id: str) -> dict:
    return await request("GET", f"/api/v1/logs/{session_id}")


async def evaluate(session_id: str, objective: str | None = None, scenario: str | None = None) -> dict:
    body = {k: v for k, v in {"objective": objective, "scenario": scenario}.items() if v}
    return await request("POST", f"/api/v1/logs/{session_id}/eval", json=body, timeout=180)


async def call(phone_number: str, participant_name: str | None = None, session_metadata: dict | None = None) -> dict:
    body = {"phone_number": phone_number}
    if participant_name:
        body["participant_name"] = participant_name
    if session_metadata:
        body["session_metadata"] = session_metadata
    return await request("POST", "/api/v1/calls", json=body)
