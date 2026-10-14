"""Live check of every Gemini integration. Run: .venv/bin/python -m scripts.check_gemini"""
import asyncio
import json
import time

import httpx
import websockets

from backend import config
from backend.agents.support_agent import KB, SupportAgent
from backend.providers import llm, stt, tts


async def step(name, coro):
    t = time.time()
    try:
        out = await coro
        print(f"✅ {name} ({time.time() - t:.1f}s): {str(out)[:140]}")
        return out
    except Exception as e:
        print(f"❌ {name}: {e}")


async def live_check():
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post("http://localhost:8765/api/gemini-live/token")
        r.raise_for_status()
        tok = r.json()
    url = (
        "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta."
        f"GenerativeService.BidiGenerateContentConstrained?access_token={tok['token']}"
    )
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"setup": {"model": f"models/{tok['model']}", "generationConfig": {"responseModalities": ["AUDIO"]}}}))
        msg = json.loads(await asyncio.wait_for(ws.recv(), 15))
        return "setupComplete" if "setupComplete" in msg else msg


async def main():
    if not config.GEMINI_API_KEY:
        return print("Set GEMINI_API_KEY in .env first.")
    g = llm.get("gemini")
    await step("LLM stream", _collect(g.stream("Be brief.", [{"role": "user", "content": "Say hi in five words."}])))
    await step("LLM tools (multi-turn, thought signatures)", SupportAgent().chat(g, [{"role": "user", "content": "Where is order VB-1042?"}]))
    await step("LLM fast model", _collect(llm.get("gemini-lite").stream("Be brief.", [{"role": "user", "content": "One word: hi"}])))
    audio = await step("TTS", tts.get("gemini").synthesize("Hello, can you check order V B one zero four two?", "Kore"))
    if audio:
        await step("STT (on the TTS audio)", stt.get("gemini").transcribe(audio[0]))
    KB.backend = None
    await step("Embeddings + RAG", KB.search("battery warranty"))
    print("   embedding backend:", KB.backend)
    await step("Live API token + WebSocket setup (needs server running)", live_check())


async def _collect(gen):
    return "".join([c async for c in gen])


asyncio.run(main())
