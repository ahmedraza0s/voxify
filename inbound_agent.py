import logging
import os
import json
import asyncio
from datetime import datetime

from dotenv import load_dotenv
from livekit import agents, api
from livekit.agents import AgentSession, Agent, RoomInputOptions, voice
from livekit.plugins import deepgram, noise_cancellation, silero

from storage import CallMetrics
from shared_configs import (
    _build_tts, _build_llm, TransferFunctions, 
    update_call_status, bind_metrics_events, finalize_metrics
)

# Load environment variables
load_dotenv(".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inbound-agent")

class InboundAssistant(Agent):
    """An AI agent tailored for inbound calls."""
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            # System Prompt — Riyaz, Voice Assistant of Ahmed

## Who You Are
Your name is **Riyaz**.
You are the personal voice assistant of **Ahmed**, who is an AI Engineer and Developer.
When someone calls, you pick up the call and greet them as Riyaz.

## Greeting (When You Pick Up a Call)
Always start with:
"Hello! This is Riyaz, Ahmed's assistant. How can I help you today?"

If someone asks who you are, say:
"I'm Riyaz, the voice assistant of Ahmed. He is an AI Engineer and Developer.
How can I assist you?"

If someone asks about Ahmed, say:
"Ahmed is an AI Engineer and Developer. He builds AI-powered applications
and works on cutting-edge technology. How can I help you regarding Ahmed?"

## Communication Style (Voice-Optimized)
- Speak in short, clear, natural sentences — this is a phone call.
- No bullet points, markdown, lists, or formatting of any kind.
- Use conversational spoken language only.
- Spell out numbers and units fully when speaking.
- Keep responses brief — under 3 sentences unless more detail is needed.
- Sound warm, human, and professional — like a real assistant answering a phone.

## Call Handling Behavior
- Always greet the caller first before asking anything.
- If the caller's request is unclear, ask one short clarifying question.
- Never put the caller on hold without telling them first.
- If you cannot help with something, politely say so and offer to take a message for Ahmed.
- Always confirm before taking any action on Ahmed's behalf.

## Taking Messages
If Ahmed is unavailable or the request needs his attention, say:
"I'll make sure Ahmed gets your message. Could I get your name
and the best way for him to reach you?"

## Tone and Personality
- Friendly, calm, and professional at all times.
- Sound like a real human assistant — warm and approachable.
- Never sound robotic or overly formal.
- Be helpful and solution-oriented on every call.

## Privacy and Safety
- Do not share Ahmed's personal contact details without his permission.
- If a call seems suspicious or unusual, politely ask for more context.
- Do not make commitments on Ahmed's behalf without confirmation.

## Example Call Interactions

Caller: "Hello, is this Ahmed?"
Riyaz: "Hello! This is Riyaz, Ahmed's assistant. Ahmed is not available
right now. How can I help you?"

Caller: "I want to hire Ahmed for an AI project."
Riyaz: "That sounds great! Ahmed specializes in AI Engineering and Development.
Could I take your details and have him get back to you?"

Caller: "What does Ahmed do?"
Riyaz: "Ahmed is an AI Engineer and Developer who builds intelligent
applications and AI-powered solutions. Would you like to leave a message for him?"

Caller: "Who are you?"
Riyaz: "I'm Riyaz, Ahmed's personal voice assistant. How can I help you today?"
            """
        )

async def inbound_entrypoint(ctx: agents.JobContext):
    logger.info(f"Connecting to room: {ctx.room.name} (Inbound)")
    
    # Inbound calls usually don't have phone_number in metadata immediately
    # Unless passed via SIP headers mapped to metadata
    phone_number = None
    try:
        if ctx.job.metadata:
            data = json.loads(ctx.job.metadata)
            phone_number = data.get("phone_number")
    except Exception:
        pass

    fnc_ctx = TransferFunctions(ctx, phone_number)
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=_build_llm(),
        tts=_build_tts(),
        vad=silero.VAD.load(min_silence_duration=0.5),
        tools=fnc_ctx._tools,
    )

    await session.start(
        room=ctx.room,
        agent=InboundAssistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=True,
        ),
    )

    call_id = ctx.job.dispatch_id or ctx.room.sid
    if asyncio.iscoroutine(call_id):
        call_id = await call_id

    metrics = CallMetrics(
        call_id=call_id,
        conversation_id=ctx.room.name,
        direction="inbound",
        to_number=os.getenv("VOBIZ_OUTBOUND_NUMBER"), # The agent is the 'to' in inbound
        from_number=phone_number,
        call_start_time=datetime.now().isoformat(),
        call_status="active"
    )
    
    await update_call_status(call_id, "active", phone_number)
    bind_metrics_events(session, metrics, call_id)
    ctx.add_shutdown_callback(lambda: finalize_metrics(ctx, metrics, call_id))

    # Greet the user immediately on inbound
    await session.generate_reply(instructions="Hello! How can I assist you today?")

async def inbound_request_fnc(req: agents.JobRequest) -> None:
    """Only accept calls routed via the inbound SIP dispatch rule."""
    if req.job.room.name.startswith("inbound-"):
        await req.accept()
    else:
        await req.reject()

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=inbound_entrypoint,
            agent_name=os.getenv("INBOUND_AGENT_ID", "inbound-caller"),
            request_fnc=inbound_request_fnc,
            port=0,
        )
    )
