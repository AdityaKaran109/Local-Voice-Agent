"""Lesson 4 — Voice for your agent.

The SupportAgent is used as-is. This file is the entire voice layer:
- "single" mode: speech -> text agent -> speech.
- "dual" mode: a fast FOREGROUND agent keeps the conversation alive (acknowledges
  immediately) while the reasoning BACKGROUND agent (RAG + tools) works.
"""
import asyncio

from ..pipeline import Agent, Session
from ..providers import llm as llm_providers
from .support_agent import SupportAgent

FOREGROUND_PROMPT = """You are the real-time voice of a customer support line. A slower specialist is looking up the answer.
Reply with ONE short natural sentence (under 15 words) that acknowledges what the customer asked and says you're checking.
Never answer the question yourself and never invent facts."""

FOREGROUND_GRACE_S = 0.8  # if the background agent answers faster than this, skip the filler


class VoiceForAgent(Agent):
    def __init__(self):
        self.agent = SupportAgent()

    async def on_start(self, s: Session):
        await s.event("ready", greeting="Hi, this is Velo Bikes support. How can I help?")

    async def respond(self, s: Session, text: str):
        s.history.append({"role": "user", "content": text})

        async def on_call(call):
            await s.event("tool_call", tool=call["name"], arguments=call["arguments"])

        background = asyncio.create_task(self.agent.chat(s.llm(), s.history, on_call=on_call))
        try:
            if s.cfg.get("mode") == "dual":
                await s.event("agent", which="background", state="started")
                done, _ = await asyncio.wait({background}, timeout=FOREGROUND_GRACE_S)
                if not done:
                    fg = llm_providers.get(s.cfg.get("fg_llm") or s.cfg["llm"])
                    await s.event("agent", which="foreground", state="speaking")
                    async for chunk in fg.stream(FOREGROUND_PROMPT, [{"role": "user", "content": text}]):
                        yield chunk
                    yield " "
            answer = await background
            if s.cfg.get("mode") == "dual":
                await s.event("agent", which="background", state="answered")
            yield answer
        finally:
            if not background.done():
                background.cancel()
