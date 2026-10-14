"""Lesson 6 — Voice AI evals (evaluation-driven development).

For each scenario a simulated caller (LLM persona) talks to the support agent.
Optionally every caller line goes through a real audio round-trip
(TTS -> WAV -> STT) so speech-recognition errors flow into the agent just like
in production, and we measure word error rate. A judge LLM grades the
transcript against the scenario rubric. Runs are saved so prompt changes can be
compared and regressions caught.
"""
import asyncio
import json
import re
import time
import uuid
from datetime import datetime

from .. import config
from ..providers import llm as llm_providers
from ..providers import stt as stt_providers
from ..providers import tts as tts_providers
from .support_agent import SYSTEM_PROMPT, SupportAgent

RUNS_DIR = config.DATA_DIR / "evals" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)  # ignored by git, so create it on a fresh clone
ACTIVE: dict[str, dict] = {}

SIMULATOR_PROMPT = """You are role-playing a CUSTOMER calling a support line (you are NOT the agent).
Persona: {persona}
Your goal: {goal}
Speak like a real caller: one short sentence. If your goal is met or the agent clearly can't help, reply with exactly [DONE]."""

JUDGE_PROMPT = """You are a strict QA evaluator for a voice customer-support agent.
Grade the conversation against each rubric criterion. Return ONLY JSON:
{"criteria":[{"criterion":"...","pass":true,"reason":"..."}],"failure_modes":["short tag", ...],"summary":"one sentence"}"""


def scenarios() -> list[dict]:
    return json.loads((config.DATA_DIR / "evals" / "scenarios.json").read_text())


NUMBER_WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine".split())} | {"oh": "0"}


STOP = {"an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we"}


def _words(text: str) -> list[str]:
    """Normalize so spoken and written forms match: 'V B four zero zero one' == 'VB-4001', '50%' == '50 percent'."""
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower().replace("%", " percent"))
    out: list[str] = []
    for tok in re.findall(r"[a-z0-9']+", text):
        tok = NUMBER_WORDS.get(tok, tok)
        prev = out[-1] if out else ""
        if tok.isdigit() and (prev[-1:].isdigit() or (re.fullmatch(r"[a-z]{2}", prev) and prev not in STOP)):
            out[-1] += tok  # "4 0 0 1" -> "4001", "vb 4001" -> "vb4001"
        elif len(tok) == 1 and tok.isalpha() and len(prev) == 1 and prev.isalpha() and tok not in "ai":
            out[-1] += tok  # spelled-out letters "v b" -> "vb"
        else:
            out.append(tok)
    return out


def wer(ref: str, hyp: str) -> float:
    r, h = _words(ref), _words(hyp)
    if not r:
        return 0.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return round(d[len(h)] / len(r), 3)


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


async def run_scenario(sc: dict, opts: dict) -> dict:
    agent_llm = llm_providers.get(opts["llm"])
    judge_llm = llm_providers.get(opts.get("judge_llm") or opts["llm"])
    agent = SupportAgent(opts.get("system_prompt") or None)
    history: list[dict] = []
    sim_history: list[dict] = []
    turns = []
    user_line = sc["opening"]
    for _ in range(sc.get("max_turns", 3)):
        turn = {"intended": user_line}
        heard = user_line
        if opts.get("audio_roundtrip"):
            t = time.perf_counter()
            audio, _ = await tts_providers.get(opts["tts"]).synthesize(user_line)
            heard = await stt_providers.get(opts["stt"]).transcribe(audio)
            turn.update(heard=heard, wer=wer(user_line, heard), asr_ms=round((time.perf_counter() - t) * 1000))
        history.append({"role": "user", "content": heard})
        tools_used = []

        async def on_call(call):
            tools_used.append(call["name"])

        t = time.perf_counter()
        reply = await agent.chat(agent_llm, history, on_call=on_call)
        turn.update(agent=reply, agent_ms=round((time.perf_counter() - t) * 1000), tools=tools_used)
        turns.append(turn)

        sim_history += [{"role": "assistant", "content": user_line}, {"role": "user", "content": reply}]
        nxt = (await agent_llm.complete(SIMULATOR_PROMPT.format(**sc), sim_history))["content"].strip()
        if not nxt or "[DONE]" in nxt:
            break
        user_line = nxt

    convo = "\n".join(f"CUSTOMER: {t.get('heard', t['intended'])}\nAGENT: {t['agent']}" for t in turns)
    rubric = "\n".join(f"- {c}" for c in sc["rubric"])
    verdict = parse_json(
        (await judge_llm.complete(JUDGE_PROMPT, [{"role": "user", "content": f"Rubric:\n{rubric}\n\nConversation:\n{convo}"}]))["content"]
    )
    criteria = verdict.get("criteria") or [{"criterion": c, "pass": False, "reason": "judge returned no verdict"} for c in sc["rubric"]]
    passed = sum(1 for c in criteria if c.get("pass"))
    wers = [t["wer"] for t in turns if "wer" in t]
    return {
        "id": sc["id"],
        "score": round(passed / max(len(criteria), 1), 2),
        "criteria": criteria,
        "failure_modes": [f for f in verdict.get("failure_modes", []) if f and f.strip().lower() not in ("none", "n/a", "")],
        "summary": verdict.get("summary", ""),
        "turns": turns,
        "avg_agent_ms": round(sum(t["agent_ms"] for t in turns) / len(turns)),
        "avg_wer": round(sum(wers) / len(wers), 3) if wers else None,
    }


async def run(opts: dict) -> str:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    selected = [s for s in scenarios() if not opts.get("scenarios") or s["id"] in opts["scenarios"]]
    record = {
        "id": run_id,
        "label": opts.get("label") or run_id,
        "status": "running",
        "started": datetime.now().isoformat(timespec="seconds"),
        "options": {k: v for k, v in opts.items() if k != "system_prompt"},
        "system_prompt": opts.get("system_prompt") or SYSTEM_PROMPT,
        "total": len(selected),
        "results": [],
    }
    ACTIVE[run_id] = record

    async def go():
        try:
            for sc in selected:
                try:
                    record["results"].append(await run_scenario(sc, opts))
                except Exception as e:
                    record["results"].append({"id": sc["id"], "score": 0, "error": str(e), "criteria": [], "turns": []})
            scores = [r["score"] for r in record["results"]]
            record["score"] = round(sum(scores) / len(scores), 2) if scores else 0
            record["status"] = "done"
        except Exception as e:
            record.update(status="error", error=str(e))
        finally:
            (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(record, indent=2))
            ACTIVE.pop(run_id, None)

    asyncio.create_task(go())
    return run_id


def get_run(run_id: str) -> dict | None:
    if run_id in ACTIVE:
        return ACTIVE[run_id]
    path = RUNS_DIR / f"{run_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def list_runs() -> list[dict]:
    runs = [json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob("*.json"), reverse=True)]
    runs = list(ACTIVE.values()) + runs
    return [
        {k: r.get(k) for k in ("id", "label", "status", "started", "score", "total", "options")} | {"done": len(r["results"])}
        for r in runs
    ]
