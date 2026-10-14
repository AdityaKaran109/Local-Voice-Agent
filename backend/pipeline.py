"""The cascaded voice pipeline: STT -> agent (LLM) -> sentence chunking -> TTS.

One Session per browser WebSocket. Agents plug in by implementing `respond`,
which yields text chunks; the session handles transcription, streaming TTS,
barge-in (cancelling the current turn) and latency metrics.
"""
import asyncio
import base64
import json
import re
import time
from typing import AsyncIterator

from fastapi import WebSocket, WebSocketDisconnect

from .providers import llm, stt, tts

SENTENCE_END = re.compile(r"[.!?;:](?=\s)|\n")


def clean_for_speech(text: str) -> str:
    text = text.translate(str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2014": ", ", "\u2013": ", "}))
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # markdown links
    text = re.sub(r"[*_#`>|]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    text = re.sub(r"[^\w\s.,!?;:'\"$%&()/@+\-]", "", text)  # emoji and symbols TTS engines read out badly
    return re.sub(r"\s+", " ", text).strip()


async def sentences(chunks: AsyncIterator[str], min_len: int = 12) -> AsyncIterator[str]:
    """Re-chunk a token stream into speakable sentences so TTS can start early."""
    buf = ""
    async for chunk in chunks:
        buf += chunk
        while True:
            cut = next((m.end() for m in SENTENCE_END.finditer(buf) if len(buf[: m.end()].strip()) >= min_len), None)
            if cut is None:
                break
            sentence, buf = buf[:cut].strip(), buf[cut:]
            if sentence:
                yield sentence
    if buf.strip():
        yield buf.strip()


class Agent:
    """Base agent: plain streaming chat."""

    system = "You are a friendly voice assistant. Keep replies short and conversational (1-3 sentences). No markdown or lists."

    async def on_start(self, s: "Session"):
        pass

    async def on_action(self, s: "Session", data: dict):
        pass

    async def respond(self, s: "Session", text: str) -> AsyncIterator[str]:
        s.history.append({"role": "user", "content": text})
        reply = ""
        try:
            async for chunk in s.llm().stream(self.system, s.history):
                reply += chunk
                yield chunk
        finally:  # keep partial replies in history even if the user barged in
            s.history.append({"role": "assistant", "content": reply or "(interrupted)"})


class Session:
    def __init__(self, ws: WebSocket, agent: Agent):
        self.ws = ws
        self.agent = agent
        self.cfg = {"stt": "whisper", "llm": "ollama", "tts": "kokoro", "voice": None}
        self.history: list[dict] = []
        self.turn_task: asyncio.Task | None = None
        self.closed = False

    # --- provider accessors (re-read per call so the UI can switch live) ---
    def llm(self):
        return llm.get(self.cfg["llm"])

    def tts_name(self):
        return self.cfg["tts"]

    def voice_for(self, index: int):
        """Pick a distinct voice per speaker (e.g. caller vs callee) for the active TTS."""
        voices = tts.REGISTRY.get(self.tts_name(), tts.REGISTRY["kokoro"]).voices
        return voices[index % len(voices)] if voices else None

    async def send(self, obj: dict):
        if self.closed:
            return
        try:
            await self.ws.send_text(json.dumps(obj))
        except Exception:
            self.closed = True

    async def event(self, name: str, **data):
        await self.send({"type": "event", "name": name, **data})

    # --- main loop ---
    async def run(self):
        await self.ws.accept()
        await self.agent.on_start(self)
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    await self.interrupt()
                    self.turn_task = asyncio.create_task(self._guard(self.handle_audio(msg["bytes"])))
                elif msg.get("text"):
                    await self.handle_json(json.loads(msg["text"]))
        except WebSocketDisconnect:
            pass
        finally:
            self.closed = True
            await self.interrupt()

    async def handle_json(self, data: dict):
        kind = data.get("type")
        if kind == "config":
            self.cfg.update({k: v for k, v in data.items() if k != "type"})
        elif kind == "text":
            await self.interrupt()
            text = data.get("text", "").strip()
            if text:
                await self.send({"type": "transcript", "role": "user", "text": text})
                self.turn_task = asyncio.create_task(self._guard(self.turn(text, {"t0": time.perf_counter()})))
        elif kind == "interrupt":
            await self.interrupt()
        elif kind == "reset":
            await self.interrupt()
            self.history.clear()
            await self.agent.on_start(self)
        elif kind == "action":
            await self.interrupt()
            self.turn_task = asyncio.create_task(self._guard(self.agent.on_action(self, data)))

    async def interrupt(self):
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
            try:
                await self.turn_task
            except (asyncio.CancelledError, Exception):
                pass
            await self.send({"type": "interrupted"})

    async def _guard(self, coro):
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.send({"type": "error", "message": str(e)})

    # --- turn handling ---
    async def handle_audio(self, wav: bytes):
        t0 = time.perf_counter()
        await self.send({"type": "status", "text": "transcribing"})
        text = await stt.get(self.cfg["stt"]).transcribe(wav)
        metrics = {"t0": t0, "stt_ms": round((time.perf_counter() - t0) * 1000)}
        if not text:
            await self.send({"type": "status", "text": "heard nothing"})
            return
        await self.send({"type": "transcript", "role": "user", "text": text})
        await self.turn(text, metrics)

    async def turn(self, text: str, metrics: dict):
        await self.send({"type": "status", "text": "thinking"})
        await self.speak_stream(self.agent.respond(self, text), metrics=metrics)

    async def speak_stream(self, chunks: AsyncIterator[str], metrics: dict | None = None, voice=None, role="assistant"):
        """Stream text to the client and synthesize it sentence by sentence, overlapping LLM and TTS."""
        metrics = metrics or {"t0": time.perf_counter()}
        t_llm = time.perf_counter()
        queue: asyncio.Queue = asyncio.Queue()
        full = []

        async def produce():
            first = True

            async def tap():
                nonlocal first
                async for c in chunks:
                    if first:
                        metrics["llm_first_token_ms"] = round((time.perf_counter() - t_llm) * 1000)
                        first = False
                    full.append(c)
                    await self.send({"type": "delta", "role": role, "text": c})
                    yield c

            async for sentence in sentences(tap()):
                await queue.put(sentence)
            await queue.put(None)

        async def consume():
            while (sentence := await queue.get()) is not None:
                await self._speak(sentence, voice, role, metrics)

        await asyncio.gather(produce(), consume())
        text = "".join(full).strip()
        metrics["total_ms"] = round((time.perf_counter() - metrics["t0"]) * 1000)
        await self.send({"type": "assistant", "role": role, "text": text})
        if role == "assistant":
            await self.send({"type": "metrics", **{k: v for k, v in metrics.items() if k != "t0"}})
        return text

    async def say(self, text: str, voice=None, role="assistant"):
        """Speak a fixed string (no LLM)."""

        async def one():
            yield text

        return await self.speak_stream(one(), voice=voice, role=role)

    async def _speak(self, sentence: str, voice, role, metrics):
        spoken = clean_for_speech(sentence)
        if not re.search(r"\w", spoken):  # skip fragments like ")" with nothing to say
            return
        name = self.tts_name()
        if name == "browser" or name == "none":
            await self.send({"type": "speak", "role": role, "text": spoken, "voice": voice})
        else:
            t = time.perf_counter()
            audio, mime = await tts.get(name).synthesize(spoken, voice or self.cfg.get("voice"))
            metrics.setdefault("tts_first_ms", round((time.perf_counter() - t) * 1000))
            await self.send(
                {"type": "audio", "role": role, "mime": mime, "text": spoken, "data": base64.b64encode(audio).decode()}
            )
        metrics.setdefault("first_audio_ms", round((time.perf_counter() - metrics["t0"]) * 1000))
