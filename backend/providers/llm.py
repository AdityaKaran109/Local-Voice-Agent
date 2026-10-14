"""LLM providers behind one interface.

Messages and tools use the OpenAI chat format everywhere; the Anthropic
adapter converts at the boundary.
"""
import json
import uuid
from typing import AsyncIterator

import httpx
from openai import AsyncOpenAI

from .. import config


class LLM:
    label = ""
    kind = "cloud"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def info(self) -> dict:
        ok, reason = self.available()
        return {"label": self.label, "kind": self.kind, "available": ok, "reason": reason}

    async def stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        raise NotImplementedError
        yield

    async def complete(self, system: str, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Returns {"content": str, "tool_calls": [{"id", "name", "arguments": dict}]}."""
        raise NotImplementedError


class OpenAICompatLLM(LLM):
    """OpenAI, Groq and Ollama all speak the OpenAI chat-completions API."""

    def __init__(self, label, base_url, key, model, kind="cloud", extra: dict | None = None):
        self.label, self.base_url, self.key, self.model, self.kind = label, base_url, key, model, kind
        self.extra = extra or {}

    def available(self):
        if self.kind == "open-source":
            try:
                tags = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=1).json()
                names = [m["name"] for m in tags.get("models", [])]
                if self.model not in names:
                    return False, f"model {self.model} not pulled (run: ollama pull {self.model})"
                return True, ""
            except Exception:
                return False, "Ollama not running (run: ollama serve)"
        return (True, "") if self.key else (False, "API key not set in .env")

    def _client(self):
        return AsyncOpenAI(base_url=self.base_url, api_key=self.key or "ollama")

    async def stream(self, system, messages):
        resp = await self._client().chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": system}, *messages], stream=True, temperature=0.6, **self.extra
        )
        async for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def complete(self, system, messages, tools=None):
        kwargs = {**self.extra, **({"tools": tools} if tools else {})}
        resp = await self._client().chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": system}, *messages], temperature=0.4, **kwargs
        )
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            call = {"id": tc.id, "name": tc.function.name, "arguments": args}
            extra_content = (tc.model_extra or {}).get("extra_content")
            if extra_content:  # Gemini 3 thought signatures must be echoed back on the next turn
                call["extra_content"] = extra_content
            calls.append(call)
        return {"content": msg.content or "", "tool_calls": calls}


class OllamaLLM(OpenAICompatLLM):
    """Native Ollama API: unlike /v1 it lets us cap num_ctx (huge default contexts spill onto the CPU)."""

    def __init__(self):
        super().__init__(f"Ollama ({config.OLLAMA_MODEL})", None, None, config.OLLAMA_MODEL, "open-source")

    def _body(self, system, messages, **extra):
        msgs = [{"role": "system", "content": system}]
        for m in messages:
            m = dict(m)
            if m.get("tool_calls"):
                m["tool_calls"] = [
                    {"function": {"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"] or "{}")}}
                    for tc in m["tool_calls"]
                ]
            m.pop("tool_call_id", None)
            msgs.append(m)
        return {
            "model": self.model,
            "messages": msgs,
            "keep_alive": "30m",
            "options": {"num_ctx": int(config.OLLAMA_NUM_CTX), "temperature": 0.5},
            **extra,
        }

    async def stream(self, system, messages):
        async with httpx.AsyncClient(timeout=300) as c:
            async with c.stream("POST", f"{config.OLLAMA_BASE_URL}/api/chat", json=self._body(system, messages, stream=True)) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line:
                        text = json.loads(line).get("message", {}).get("content")
                        if text:
                            yield text

    async def complete(self, system, messages, tools=None):
        extra = {"stream": False, **({"tools": tools} if tools else {})}
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.post(f"{config.OLLAMA_BASE_URL}/api/chat", json=self._body(system, messages, **extra))
            r.raise_for_status()
        msg = r.json()["message"]
        calls = [
            {"id": f"call_{i}_{uuid.uuid4().hex[:6]}", "name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or {}}
            for i, tc in enumerate(msg.get("tool_calls") or [])
        ]
        return {"content": msg.get("content") or "", "tool_calls": calls}


class AnthropicLLM(LLM):
    label = f"Anthropic ({config.ANTHROPIC_MODEL})"

    def available(self):
        return (True, "") if config.ANTHROPIC_API_KEY else (False, "ANTHROPIC_API_KEY not set")

    def _client(self):
        from anthropic import AsyncAnthropic

        return AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)

    @staticmethod
    def _convert(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                for tc in m["tool_calls"]:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"] or "{}"),
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
            else:
                role = "assistant" if m["role"] == "assistant" else "user"
                if out and out[-1]["role"] == role and isinstance(out[-1]["content"], str):
                    out[-1]["content"] += "\n" + m["content"]  # Anthropic requires alternating roles
                else:
                    out.append({"role": role, "content": m["content"]})
        return out

    async def stream(self, system, messages):
        async with self._client().messages.stream(
            model=config.ANTHROPIC_MODEL, max_tokens=1024, system=system, messages=self._convert(messages)
        ) as s:
            async for text in s.text_stream:
                yield text

    async def complete(self, system, messages, tools=None):
        kwargs = {}
        if tools:
            kwargs["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]
        resp = await self._client().messages.create(
            model=config.ANTHROPIC_MODEL, max_tokens=1024, system=system, messages=self._convert(messages), **kwargs
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "arguments": b.input} for b in resp.content if b.type == "tool_use"]
        return {"content": text, "tool_calls": calls}


REGISTRY: dict[str, LLM] = {
    "ollama": OllamaLLM(),
    "openai": OpenAICompatLLM(
        f"OpenAI ({config.OPENAI_LLM_MODEL})", "https://api.openai.com/v1", config.OPENAI_API_KEY, config.OPENAI_LLM_MODEL
    ),
    "gemini": OpenAICompatLLM(
        f"Gemini ({config.GEMINI_LLM_MODEL})",
        f"{config.GEMINI_BASE_URL}/openai/",
        config.GEMINI_API_KEY,
        config.GEMINI_LLM_MODEL,
        extra={"reasoning_effort": config.GEMINI_REASONING_EFFORT} if config.GEMINI_REASONING_EFFORT else {},
    ),
    "gemini-lite": OpenAICompatLLM(
        f"Gemini fast ({config.GEMINI_FAST_MODEL})", f"{config.GEMINI_BASE_URL}/openai/", config.GEMINI_API_KEY, config.GEMINI_FAST_MODEL
    ),
    "anthropic": AnthropicLLM(),
    "groq": OpenAICompatLLM(
        f"Groq ({config.GROQ_LLM_MODEL})", "https://api.groq.com/openai/v1", config.GROQ_API_KEY, config.GROQ_LLM_MODEL
    ),
}


def get(name: str) -> LLM:
    provider = REGISTRY.get(name) or REGISTRY["ollama"]
    ok, reason = provider.available()
    if not ok:
        raise RuntimeError(f"LLM '{name}' unavailable: {reason}")
    return provider


async def run_tools(
    llm: LLM, system: str, messages: list[dict], tools: list[dict], handlers: dict, on_call=None, max_rounds: int = 6
) -> str:
    """Generic tool-use loop. Mutates `messages` with the assistant/tool turns and returns the final text."""
    for _ in range(max_rounds):
        result = await llm.complete(system, messages, tools)
        if not result["tool_calls"]:
            messages.append({"role": "assistant", "content": result["content"]})
            return result["content"]
        messages.append(
            {
                "role": "assistant",
                "content": result["content"],
                "tool_calls": [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                    | ({"extra_content": c["extra_content"]} if c.get("extra_content") else {})
                    for c in result["tool_calls"]
                ],
            }
        )
        for call in result["tool_calls"]:
            if on_call:
                await on_call(call)
            handler = handlers.get(call["name"])
            try:
                output = await handler(**call["arguments"]) if handler else {"error": f"unknown tool {call['name']}"}
            except Exception as e:  # surface tool errors to the model instead of crashing the turn
                output = {"error": str(e)}
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": output if isinstance(output, str) else json.dumps(output)}
            )
    final = await llm.complete(system, messages)
    messages.append({"role": "assistant", "content": final["content"]})
    return final["content"]
