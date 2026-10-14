"""Text-to-speech providers. Output is (audio bytes, mime type)."""
import asyncio
import base64
from pathlib import Path

import httpx

from .. import config
import numpy as np

from .audio import float32_to_wav


class TTS:
    label = ""
    kind = "cloud"
    voices: list[str] = []

    def available(self) -> tuple[bool, str]:
        return True, ""

    def info(self) -> dict:
        ok, reason = self.available()
        return {"label": self.label, "kind": self.kind, "available": ok, "reason": reason, "voices": self.voices}

    async def synthesize(self, text: str, voice: str | None = None) -> tuple[bytes, str]:
        raise NotImplementedError


class KokoroTTS(TTS):
    label = "Kokoro-82M (ONNX)"
    kind = "open-source"
    voices = ["af_heart", "af_bella", "af_nicole", "am_michael", "am_adam", "bf_emma", "bm_george"]
    _engine = None

    def available(self):
        if Path(config.KOKORO_MODEL).exists() and Path(config.KOKORO_VOICES).exists():
            return True, ""
        return False, "model files missing in models/ (see README)"

    def _run(self, text, voice):
        if KokoroTTS._engine is None:
            from kokoro_onnx import Kokoro

            KokoroTTS._engine = Kokoro(config.KOKORO_MODEL, config.KOKORO_VOICES)
        voice = voice if voice in self.voices else self.voices[0]
        samples, sr = KokoroTTS._engine.create(text, voice=voice, speed=1.05, lang="en-gb" if voice[0] == "b" else "en-us")
        return float32_to_wav(samples, sr)

    async def synthesize(self, text, voice=None):
        return await asyncio.to_thread(self._run, text, voice), "audio/wav"


class OpenAITTS(TTS):
    label = f"OpenAI ({config.OPENAI_TTS_MODEL})"
    voices = ["alloy", "ash", "coral", "echo", "sage", "shimmer", "verse"]

    def available(self):
        return (True, "") if config.OPENAI_API_KEY else (False, "OPENAI_API_KEY not set")

    async def synthesize(self, text, voice=None):
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
                json={
                    "model": config.OPENAI_TTS_MODEL,
                    "voice": voice if voice in self.voices else "alloy",
                    "input": text,
                    "response_format": "wav",
                },
            )
            r.raise_for_status()
            return r.content, "audio/wav"


class ElevenLabsTTS(TTS):
    label = f"ElevenLabs ({config.ELEVENLABS_MODEL})"
    voices = [config.ELEVENLABS_VOICE_ID]

    def available(self):
        return (True, "") if config.ELEVENLABS_API_KEY else (False, "ELEVENLABS_API_KEY not set")

    async def synthesize(self, text, voice=None):
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice or config.ELEVENLABS_VOICE_ID}",
                params={"output_format": "mp3_44100_128"},
                headers={"xi-api-key": config.ELEVENLABS_API_KEY},
                json={"text": text, "model_id": config.ELEVENLABS_MODEL},
            )
            r.raise_for_status()
            return r.content, "audio/mpeg"


class DeepgramTTS(TTS):
    label = "Deepgram Aura"
    voices = [config.DEEPGRAM_TTS_MODEL, "aura-2-apollo-en", "aura-2-luna-en"]

    def available(self):
        return (True, "") if config.DEEPGRAM_API_KEY else (False, "DEEPGRAM_API_KEY not set")

    async def synthesize(self, text, voice=None):
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.deepgram.com/v1/speak",
                params={"model": voice if voice in self.voices else self.voices[0], "encoding": "linear16", "container": "wav"},
                headers={"Authorization": f"Token {config.DEEPGRAM_API_KEY}"},
                json={"text": text},
            )
            r.raise_for_status()
            return r.content, "audio/wav"


class GeminiTTS(TTS):
    label = f"Gemini ({config.GEMINI_TTS_MODEL})"
    voices = ["Kore", "Puck", "Charon", "Aoede", "Fenrir", "Leda", "Orus", "Zephyr", "Sulafat", "Achird"]

    def available(self):
        return (True, "") if config.GEMINI_API_KEY else (False, "GEMINI_API_KEY not set")

    async def synthesize(self, text, voice=None):
        voice = voice if voice in self.voices else self.voices[0]
        headers = {"x-goog-api-key": config.GEMINI_API_KEY}
        async with httpx.AsyncClient(timeout=60) as client:
            # generateContent with an AUDIO response modality
            r = await client.post(
                f"{config.GEMINI_BASE_URL}/models/{config.GEMINI_TTS_MODEL}:generateContent",
                headers=headers,
                json={
                    "contents": [{"parts": [{"text": text}]}],
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
                    },
                },
            )
            if r.status_code < 400:
                for part in r.json()["candidates"][0]["content"]["parts"]:
                    if "inlineData" in part:
                        return self._to_wav(part["inlineData"]), "audio/wav"
            # newer Interactions API (documented form for the 3.8 TTS models)
            r2 = await client.post(
                f"{config.GEMINI_BASE_URL}/interactions",
                headers=headers,
                json={
                    "model": config.GEMINI_TTS_MODEL,
                    "input": [{"type": "user_input", "content": [{"type": "text", "text": text}]}],
                    "response_format": {"type": "audio"},
                    "generation_config": {"speech_config": [{"voice": voice}]},
                },
            )
            if r2.status_code >= 400:
                raise RuntimeError(f"Gemini TTS failed: {r.status_code} {r.text[:200]} | {r2.status_code} {r2.text[:200]}")
            for step in r2.json().get("steps", []):
                for c in step.get("content", []) or []:
                    if c.get("type") == "audio":
                        return base64.b64decode(c["data"]), c.get("mime_type", "audio/wav")
            raise RuntimeError("Gemini TTS returned no audio")

    @staticmethod
    def _to_wav(inline: dict) -> bytes:
        data = base64.b64decode(inline["data"])
        mime = inline.get("mimeType", "")
        if data[:4] == b"RIFF":
            return data
        rate = int(mime.split("rate=")[1].split(";")[0]) if "rate=" in mime else 24000
        pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0  # raw 16-bit little-endian PCM
        return float32_to_wav(pcm, rate)


class BrowserTTS(TTS):
    """speechSynthesis in the browser; the server sends text only."""

    label = "Browser speechSynthesis"
    kind = "browser"


REGISTRY: dict[str, TTS] = {
    "kokoro": KokoroTTS(),
    "browser": BrowserTTS(),
    "gemini": GeminiTTS(),
    "openai": OpenAITTS(),
    "elevenlabs": ElevenLabsTTS(),
    "deepgram": DeepgramTTS(),
}


def get(name: str) -> TTS:
    provider = REGISTRY.get(name) or REGISTRY["kokoro"]
    ok, reason = provider.available()
    if not ok:
        raise RuntimeError(f"TTS '{name}' unavailable: {reason}")
    return provider
