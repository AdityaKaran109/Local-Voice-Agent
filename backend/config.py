"""Central configuration. Every provider/model can be overridden in .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FRONTEND_DIR = ROOT / "frontend"


def env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    return value if value not in (None, "") else default


# --- API keys (all optional; the open-source stack needs none) ---
OPENAI_API_KEY = env("OPENAI_API_KEY")
ANTHROPIC_API_KEY = env("ANTHROPIC_API_KEY")
GEMINI_API_KEY = env("GEMINI_API_KEY") or env("GOOGLE_API_KEY")
VOCAL_BRIDGE_API_KEY = env("VOCAL_BRIDGE_API_KEY")
VOCAL_BRIDGE_AGENT_ID = env("VOCAL_BRIDGE_AGENT_ID")  # only needed with an account-scoped key
VOCAL_BRIDGE_API_URL = env("VOCAL_BRIDGE_API_URL", "https://vocalbridgeai.com")
GROQ_API_KEY = env("GROQ_API_KEY")
DEEPGRAM_API_KEY = env("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = env("ELEVENLABS_API_KEY")
TWILIO_ACCOUNT_SID = env("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = env("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = env("TWILIO_FROM_NUMBER")
PUBLIC_URL = env("PUBLIC_URL")  # e.g. an ngrok https URL, required for real Twilio calls

# --- Open-source / local ---
OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = env("OLLAMA_MODEL", "ministral-3:latest")
OLLAMA_NUM_CTX = env("OLLAMA_NUM_CTX", "8192")  # keep small so the model fits on the GPU
OLLAMA_EMBED_MODEL = env("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
WHISPER_MODEL = env("WHISPER_MODEL", "base.en")  # tiny.en | base.en | small.en | medium.en | large-v3
KOKORO_MODEL = env("KOKORO_MODEL", str(MODELS_DIR / "kokoro-v1.0.onnx"))
KOKORO_VOICES = env("KOKORO_VOICES", str(MODELS_DIR / "voices-v1.0.bin"))

# --- Google Gemini ---
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_LLM_MODEL = env("GEMINI_LLM_MODEL", "gemini-3.8-flash")
GEMINI_FAST_MODEL = env("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite")  # foreground agent / simulator
GEMINI_STT_MODEL = env("GEMINI_STT_MODEL", "gemini-3.5-flash-lite")  # audio understanding used as STT
GEMINI_TTS_MODEL = env("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
GEMINI_LIVE_MODEL = env("GEMINI_LIVE_MODEL", "gemini-3.8-live")
GEMINI_EMBED_MODEL = env("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_REASONING_EFFORT = env("GEMINI_REASONING_EFFORT", "low")  # keep voice turns snappy; "" to use model default

# --- Other cloud model names ---
OPENAI_LLM_MODEL = env("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_STT_MODEL = env("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
OPENAI_TTS_MODEL = env("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_REALTIME_MODEL = env("OPENAI_REALTIME_MODEL", "gpt-realtime")
OPENAI_EMBED_MODEL = env("OPENAI_EMBED_MODEL", "text-embedding-3-small")
ANTHROPIC_MODEL = env("ANTHROPIC_MODEL", "claude-haiku-5-5")
GROQ_LLM_MODEL = env("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_STT_MODEL = env("GROQ_STT_MODEL", "whisper-large-v3-turbo")
DEEPGRAM_STT_MODEL = env("DEEPGRAM_STT_MODEL", "nova-3")
DEEPGRAM_TTS_MODEL = env("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")
ELEVENLABS_MODEL = env("ELEVENLABS_MODEL", "eleven_flash_v2_5")
ELEVENLABS_VOICE_ID = env("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
