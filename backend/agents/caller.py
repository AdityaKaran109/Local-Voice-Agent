"""Lesson 5 — Voice as a tool.

A normal assistant gets a `make_phone_call` tool and decides on its own when a
call is the right move. Calls are either:
- simulated: our caller agent talks to a demo business agent, both voiced by the
  selected TTS, transcript streamed live to the UI (works fully offline), or
- real: an outbound Twilio call to an actual phone (see telephony.py).
"""
import re

from .. import config
from ..pipeline import Agent, Session
from ..providers.llm import run_tools
from . import telephony
from .support_agent import SupportAgent

ASSISTANT_PROMPT = """You are a personal assistant that can place phone calls on the user's behalf with the make_phone_call tool.
Use it when the user wants something done that requires calling a business (bookings, appointments, questions).
Known numbers: Luigi's Trattoria +1-555-0100, Bright Smile Dental +1-555-0142, Velo Bikes support +1-555-0199.
Before calling, make sure you know what the user wants. After the call, report the outcome in 1-2 short sentences. No markdown."""

CALLER_PROMPT = """You are an AI assistant making a phone call on behalf of your user.
Objective: {objective}
Speak naturally like a person on the phone: one or two short sentences per turn. Introduce yourself as calling on behalf of your user.
Answer the other side's questions sensibly (the user's name is Alex if asked). If the exact request isn't possible, use your judgment
with any flexibility the objective allows. Only AFTER the other person has confirmed the outcome (or it is clearly impossible), say a
polite goodbye and end that final message with [END]. Never put [END] in a message that asks a question."""

DEMO_BUSINESSES = {
    "5550100": {
        "name": "Luigi's Trattoria",
        "greeting": "Luigi's Trattoria, this is Maria speaking, how can I help you?",
        "prompt": "You are Maria, the host at Luigi's Trattoria, answering the phone. Open 5pm-10pm, closed Mondays. "
        "This Friday 7pm is fully booked; 6:30pm and 8:15pm are free for up to 6 people. Ask for name and party size before confirming. "
        "Keep replies to 1-2 short spoken sentences.",
    },
    "5550142": {
        "name": "Bright Smile Dental",
        "greeting": "Bright Smile Dental, front desk. How can I help?",
        "prompt": "You are the receptionist at Bright Smile Dental. Next openings: Wednesday 10am, Thursday 3:30pm. Cleanings take 45 minutes "
        "and cost $120 without insurance. Ask for patient name and whether they're a new patient. 1-2 short spoken sentences.",
    },
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "make_phone_call",
            "description": "Place a phone call to a business and have a voice conversation to accomplish an objective. Returns transcript and outcome.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string"},
                    "objective": {"type": "string", "description": "What to accomplish, with all details the user gave."},
                },
                "required": ["phone_number", "objective"],
            },
        },
    }
]

MAX_CALL_TURNS = 8


def digits(number: str) -> str:
    d = re.sub(r"\D", "", number)
    return d[-7:] if len(d) >= 7 else d


async def simulate_call(s: Session, number: str, objective: str) -> dict:
    biz = DEMO_BUSINESSES.get(digits(number))
    support = SupportAgent() if digits(number) == "5550199" else None  # reuse the lesson-4 agent as a callee
    name = biz["name"] if biz else ("Velo Bikes support" if support else "Unknown business")
    llm = s.llm()
    await s.event("call", state="ringing", number=number, callee=name, mode="simulated")

    callee_history: list[dict] = []
    caller_history: list[dict] = []
    transcript: list[dict] = []
    caller_voice, callee_voice = s.voice_for(3), s.voice_for(1)

    async def callee_says(text):
        transcript.append({"speaker": "callee", "text": text})
        callee_history.append({"role": "assistant", "content": text})
        caller_history.append({"role": "user", "content": text})
        await s.say(text, voice=callee_voice, role="callee")

    greeting = biz["greeting"] if biz else ("Velo Bikes support, how can I help?" if support else "Hello?")
    await s.event("call", state="connected", number=number, callee=name)
    await callee_says(greeting)

    for turn in range(MAX_CALL_TURNS):
        line = (await llm.complete(CALLER_PROMPT.format(objective=objective), caller_history))["content"].strip()
        ended = "[END]" in line and turn > 0  # small models sometimes end prematurely on the first line
        line = line.replace("[END]", "").strip()
        transcript.append({"speaker": "caller", "text": line})
        caller_history.append({"role": "assistant", "content": line})
        callee_history.append({"role": "user", "content": line})
        await s.say(line, voice=caller_voice, role="caller")
        if ended:
            break
        if support:
            reply = await support.chat(llm, callee_history)
            callee_history.pop()  # chat() appended it; callee_says re-adds it
        elif biz:
            reply = (await llm.complete(biz["prompt"], callee_history))["content"].strip()
        else:
            reply = "Sorry, I think you have the wrong number. Goodbye."
        await callee_says(reply)

    await s.event("call", state="ended", number=number, callee=name)
    return {"callee": name, "transcript": transcript}


class CallerAgent(Agent):
    system = ASSISTANT_PROMPT

    async def respond(self, s: Session, text: str):
        s.history.append({"role": "user", "content": text})

        async def make_phone_call(phone_number: str, objective: str):
            await s.say(f"Okay, calling {phone_number} now.")
            if s.cfg.get("call_mode") == "twilio":
                result = await telephony.place_call(s, phone_number, objective)
            elif s.cfg.get("call_mode") == "vocalbridge":
                result = await telephony.place_call_vocalbridge(s, phone_number, objective)
            else:
                result = await simulate_call(s, phone_number, objective)
            convo = "\n".join(f"{t['speaker']}: {t['text']}" for t in result["transcript"])
            summary = await s.llm().complete(
                "Summarize the outcome of this phone call in one sentence for the user (what was agreed, times, prices).",
                [{"role": "user", "content": convo or "(no conversation)"}],
            )
            await s.event("call", state="summary", summary=summary["content"])
            return {"callee": result.get("callee"), "outcome": summary["content"]}

        async def on_call(call):
            await s.event("tool_call", tool=call["name"], arguments=call["arguments"])

        yield await run_tools(s.llm(), self.system, s.history, TOOLS, {"make_phone_call": make_phone_call}, on_call)


def twilio_ready() -> tuple[bool, str]:
    missing = [k for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER", "PUBLIC_URL") if not getattr(config, k)]
    return (not missing, f"missing in .env: {', '.join(missing)}" if missing else "")
