from . import llm, stt, tts


def catalog() -> dict:
    """Everything the frontend needs to build provider dropdowns."""
    return {
        "stt": [{"id": k, **v.info()} for k, v in stt.REGISTRY.items()],
        "llm": [{"id": k, **v.info()} for k, v in llm.REGISTRY.items()],
        "tts": [{"id": k, **v.info()} for k, v in tts.REGISTRY.items()],
    }
