"""An ordinary TEXT agent: RAG over a knowledge base + tools.

It knows nothing about voice. Lesson 4 ("Voice for your agent") wraps it with a
voice layer without changing its prompt, retrieval or tools.
"""
import math
import re

import httpx

from .. import config
from ..providers.llm import LLM, run_tools

SYSTEM_PROMPT = """You are the customer support agent for Velo Bikes, an online bicycle store.
ALWAYS call search_knowledge_base before answering anything about policies, shipping, returns, warranty, products or hours,
and answer ONLY with facts from the knowledge base and tool results. If you don't know, say so.
Use get_order_status for any question about a specific order, and create_return to start a return once the customer confirms.
Never invent discount codes, prices or policies. Be concise: 1-3 sentences."""

ORDERS = {
    "VB-1042": {"item": "Velo City (M)", "status": "shipped", "carrier": "UPS", "eta": "Thursday", "delivered_days_ago": None},
    "VB-2077": {"item": "Velo Volt", "status": "delivered", "delivered_days_ago": 12},
    "VB-3150": {"item": "Velo Trail (L)", "status": "delivered", "delivered_days_ago": 45},
    "VB-4001": {"item": "Helmet + lights bundle", "status": "processing", "eta": "ships tomorrow"},
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search Velo Bikes policies and product info (shipping, returns, warranty, products, store hours).",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up an order by ID, e.g. VB-1042.",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_return",
            "description": "Start a return for a delivered order. Only call after the customer confirms.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["order_id", "reason"],
            },
        },
    },
]


def normalize_order_id(raw: str) -> str:
    digits = re.sub(r"\D", "", raw or "")
    return f"VB-{digits}" if digits else (raw or "").upper()


class KnowledgeBase:
    def __init__(self):
        self.chunks: list[dict] = []
        for path in sorted((config.DATA_DIR / "kb").glob("*.md")):
            for para in path.read_text().split("\n\n"):
                para = para.strip()
                if para and not para.startswith("#"):
                    self.chunks.append({"source": path.stem, "text": para, "vec": None})
        self.backend = None

    async def _embed(self, texts: list[str]) -> list[list[float]] | None:
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(f"{config.OLLAMA_BASE_URL}/api/embed", json={"model": config.OLLAMA_EMBED_MODEL, "input": texts})
                r.raise_for_status()
                self.backend = "ollama:" + config.OLLAMA_EMBED_MODEL
                return r.json()["embeddings"]
        except Exception:
            pass
        if config.GEMINI_API_KEY:
            try:
                async with httpx.AsyncClient(timeout=30) as c:
                    r = await c.post(
                        f"{config.GEMINI_BASE_URL}/models/{config.GEMINI_EMBED_MODEL}:batchEmbedContents",
                        headers={"x-goog-api-key": config.GEMINI_API_KEY},
                        json={
                            "requests": [
                                {"model": f"models/{config.GEMINI_EMBED_MODEL}", "content": {"parts": [{"text": t}]}} for t in texts
                            ]
                        },
                    )
                    r.raise_for_status()
                    self.backend = "gemini:" + config.GEMINI_EMBED_MODEL
                    return [e["values"] for e in r.json()["embeddings"]]
            except Exception:
                pass
        if config.OPENAI_API_KEY:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
                    json={"model": config.OPENAI_EMBED_MODEL, "input": texts},
                )
                r.raise_for_status()
                self.backend = "openai:" + config.OPENAI_EMBED_MODEL
                return [d["embedding"] for d in r.json()["data"]]
        return None

    async def search(self, query: str, k: int = 3) -> list[dict]:
        if self.chunks and self.chunks[0]["vec"] is None and self.backend != "keyword":
            vecs = await self._embed([c["text"] for c in self.chunks])
            if vecs:
                for c, v in zip(self.chunks, vecs):
                    c["vec"] = v
            else:
                self.backend = "keyword"
        if self.backend == "keyword":  # no embedding model available: word-overlap fallback
            q = set(re.findall(r"\w+", query.lower()))
            scored = [(len(q & set(re.findall(r"\w+", c["text"].lower()))), c) for c in self.chunks]
        else:
            qv = (await self._embed([query]))[0]
            scored = [(_cos(qv, c["vec"]), c) for c in self.chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"source": c["source"], "text": c["text"]} for _, c in scored[:k]]


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1)


KB = KnowledgeBase()


class SupportAgent:
    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.returns: list[dict] = []

    async def search_knowledge_base(self, query: str):
        return await KB.search(query)

    async def get_order_status(self, order_id: str):
        oid = normalize_order_id(order_id)
        return {"order_id": oid, **ORDERS[oid]} if oid in ORDERS else {"error": f"No order found with ID {oid}"}

    async def create_return(self, order_id: str, reason: str):
        oid = normalize_order_id(order_id)
        order = ORDERS.get(oid)
        if not order:
            return {"error": f"No order found with ID {oid}"}
        if order.get("delivered_days_ago") is None:
            return {"error": "Order has not been delivered yet"}
        if order["delivered_days_ago"] > 30:
            return {"error": "Outside the 30-day return window; not eligible for refund"}
        rma = f"RMA-{len(self.returns) + 501}"
        self.returns.append({"rma": rma, "order_id": oid, "reason": reason})
        return {"rma": rma, "label": "emailed to customer", "refund": "within 5 business days of arrival"}

    async def chat(self, llm: LLM, history: list[dict], on_call=None) -> str:
        """Text in, text out. `history` is mutated with the new assistant/tool turns."""
        handlers = {
            "search_knowledge_base": self.search_knowledge_base,
            "get_order_status": self.get_order_status,
            "create_return": self.create_return,
        }
        return await run_tools(llm, self.system_prompt, history, TOOLS, handlers, on_call=on_call)
