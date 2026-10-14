"""Small audio helpers: decode/encode WAV and resample without ffmpeg."""
import io

import numpy as np
import soundfile as sf


def wav_to_float32(data: bytes, target_sr: int = 16000) -> np.ndarray:
    samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if sr != target_sr and len(mono):
        duration = len(mono) / sr
        new_len = int(duration * target_sr)
        mono = np.interp(np.linspace(0, len(mono), new_len, endpoint=False), np.arange(len(mono)), mono)
    return mono.astype(np.float32)


def float32_to_wav(samples: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
