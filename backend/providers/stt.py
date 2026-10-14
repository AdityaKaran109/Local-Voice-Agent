"""Speech-to-text providers. Input is WAV bytes, output is text."""
import asyncio
import base64

import httpx

from .. import config
from .audio import wav_to_float32


class STT:
    label = ""
    kind = "cloud"  # "open-source" | "cloud" | "browser"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def info(self) -> dict:
        ok, reason = self.available()
        return {"label": self.label, "kind": self.kind, "available": ok, "reason": reason}

    async def transcribe(self, wav: bytes) -> str:
        raise NotImplementedError


class FasterWhisperSTT(STT):
    label = f"faster-whisper ({config.WHISPER_MODEL})"
    kind = "open-source"
    _model = None

    def _load(self):
        if FasterWhisperSTT._model is None:
            from faster_whisper import WhisperModel

            FasterWhisperSTT._model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
        return FasterWhisperSTT._model

    def _run(self, wav: bytes) -> str:
        audio = wav_to_float32(wav)
        segments, _ = self._load().transcribe(audio, beam_size=1, language="en", vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, wav: bytes) -> str:
        return await asyncio.to_thread(self._run, wav)


class OpenAICompatSTT(STT):
    def __init__(self, label, base_url, key, model):
        self.label, self.base_url, self.key, self.model = label, base_url, key, model

    def available(self):
        return (True, "") if self.key else (False, "API key not set in .env")

    async def transcribe(self, wav: bytes) -> str:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{self.base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.key}"},
                files={"file": ("audio.wav", wav, "audio/wav")},
                data={"model": self.model, "language": "en"},
            )
            r.raise_for_status()
            return r.json()["text"].strip()


class DeepgramSTT(STT):
    label = f"Deepgram ({config.DEEPGRAM_STT_MODEL})"

    def available(self):
        return (True, "") if config.DEEPGRAM_API_KEY else (False, "DEEPGRAM_API_KEY not set")

    async def transcribe(self, wav: bytes) -> str:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.deepgram.com/v1/listen",
                params={"model": config.DEEPGRAM_STT_MODEL, "smart_format": "true"},
                headers={"Authorization": f"Token {config.DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"},
                content=wav,
            )
            r.raise_for_status()
            return r.json()["results"]["channels"][0]["alternatives"][0]["transcript"].strip()


class GeminiSTT(STT):
    """Gemini audio understanding, prompted to return a verbatim transcript."""

    label = f"Gemini ({config.GEMINI_STT_MODEL})"
    PROMPT = "Transcribe this audio verbatim in English. Output only the spoken words, nothing else. If there is no speech, output nothing."

    def available(self):
        return (True, "") if config.GEMINI_API_KEY else (False, "GEMINI_API_KEY not set")

    async def transcribe(self, wav: bytes) -> str:
        body = {
            "contents": [
                {"parts": [{"text": self.PROMPT}, {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(wav).decode()}}]}
            ],
            "generationConfig": {"temperature": 0},
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{config.GEMINI_BASE_URL}/models/{config.GEMINI_STT_MODEL}:generateContent",
                headers={"x-goog-api-key": config.GEMINI_API_KEY},
                json=body,
            )
            r.raise_for_status()
        parts = (r.json().get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()


class BrowserSTT(STT):
    """Web Speech API — runs in the browser; the server only receives text."""

    label = "Browser Web Speech API"
    kind = "browser"


REGISTRY: dict[str, STT] = {
    "whisper": FasterWhisperSTT(),
    "browser": BrowserSTT(),
    "gemini": GeminiSTT(),
    "openai": OpenAICompatSTT(
        f"OpenAI ({config.OPENAI_STT_MODEL})", "https://api.openai.com/v1", config.OPENAI_API_KEY, config.OPENAI_STT_MODEL
    ),
    "groq": OpenAICompatSTT(
        f"Groq ({config.GROQ_STT_MODEL})", "https://api.groq.com/openai/v1", config.GROQ_API_KEY, config.GROQ_STT_MODEL
    ),
    "deepgram": DeepgramSTT(),
}


def get(name: str) -> STT:
    provider = REGISTRY.get(name) or REGISTRY["whisper"]
    ok, reason = provider.available()
    if not ok:
        raise RuntimeError(f"STT '{name}' unavailable: {reason}")
    return provider
