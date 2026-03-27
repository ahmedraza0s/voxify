import logging
import os
import json
import asyncio
import uuid
import time
import threading
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, BackgroundTasks
import uvicorn
from livekit import agents, api
from livekit.agents import AgentSession, Agent, RoomInputOptions, voice
from livekit.plugins import deepgram, noise_cancellation, silero

from storage import CallMetrics, TranscriptSegment
from shared_configs import (
    _build_tts, _build_llm, TransferFunctions, 
    update_call_status, bind_metrics_events, finalize_metrics
)

# Load environment variables
load_dotenv(".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("outbound-agent")

# FastAPI app for triggering calls
app = FastAPI()

async def async_trigger_call(phone: str, variables: dict):
    """Asynchronously trigger the outbound call using LiveKit dispatch logic."""
    url = os.getenv("LIVEKIT_URL")
    lk_api_key = os.getenv("LIVEKIT_API_KEY")
    lk_api_secret = os.getenv("LIVEKIT_API_SECRET")
    agent_id = os.getenv("OUTBOUND_AGENT_ID", "outbound-caller")

    lk_api = api.LiveKitAPI(url=url, api_key=lk_api_key, api_secret=lk_api_secret)
    room_name = f"call-{phone.replace('+', '')}-{uuid.uuid4().hex[:4]}"
    
    metadata = variables.copy()
    metadata["phone_number"] = phone
    
    try:
        logger.info(f"Triggering outbound call to {phone}...")
        dispatch_request = api.CreateAgentDispatchRequest(
            agent_name=agent_id,
            room=room_name,
            metadata=json.dumps(metadata)
        )
        dispatch = await lk_api.agent_dispatch.create_dispatch(dispatch_request)
        logger.info(f"✅ Call Dispatched Successfully to {phone}! ID: {dispatch.id}")
        return dispatch.id
    finally:
        await lk_api.aclose()

@app.post("/trigger-call")
async def trigger_call(request: Request, authorization: str = Header(None)):
    api_key = os.getenv("API_KEY")
    if not authorization or not authorization.startswith("Bearer ") or authorization.split(" ")[1] != api_key:
        raise HTTPException(status_code=401, detail={"error": "unauthorized"})

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "invalid json"})

    phone = data.get("phone")
    if not phone:
        raise HTTPException(status_code=400, detail={"error": "phone is required"})

    agent_id_env = os.getenv("OUTBOUND_AGENT_ID", "outbound-caller")
    agent_id_req = data.get("agent_id")
    if not agent_id_req:
        raise HTTPException(status_code=400, detail={"error": "agent_id is required"})
    if agent_id_req != agent_id_env:
        raise HTTPException(status_code=400, detail={"error": "invalid agent_id"})

    try:
        dispatch_id = await async_trigger_call(phone, data)
        await update_call_status(dispatch_id, "queued", phone)
        return {"status": "call queued", "call_id": dispatch_id}
    except Exception as e:
        logger.error(f"Failed to trigger call: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/call-status/{call_id}")
async def get_call_status(call_id: str, authorization: str = Header(None)):
    api_key = os.getenv("API_KEY")
    if not authorization or not authorization.startswith("Bearer ") or authorization.split(" ")[1] != api_key:
        raise HTTPException(status_code=401, detail={"error": "unauthorized"})

    metrics_file = os.path.join("KMS", "logs", f"call_{call_id}.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            return json.load(f)

    STATUS_STORE_FILE = os.path.join("KMS", "logs", "active_calls.json")
    if os.path.exists(STATUS_STORE_FILE):
        with open(STATUS_STORE_FILE, "r") as f:
            data = json.load(f)
            status_data = data.get(call_id)
            if status_data:
                return status_data

    raise HTTPException(status_code=404, detail={"error": "call not found"})

class OutboundAssistant(Agent):
    """An AI agent tailored for outbound calls."""
    def __init__(self) -> None:
        super().__init__(
            instructions="""# System Prompt — Riyaz, Voice Assistant of Ahmed

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

async def outbound_entrypoint(ctx: agents.JobContext):
    logger.info(f"Connecting to room: {ctx.room.name} (Outbound)")
    
    phone_number = None
    try:
        if ctx.job.metadata:
            data = json.loads(ctx.job.metadata)
            phone_number = data.get("phone_number")
    except Exception:
        logger.warning("No valid JSON metadata found.")

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
        agent=OutboundAssistant(),
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
        direction="outbound",
        to_number=phone_number,
        from_number=os.getenv("VOBIZ_OUTBOUND_NUMBER"),
        call_start_time=datetime.now().isoformat(),
        call_status="active"
    )
    
    await update_call_status(call_id, "active", phone_number)
    bind_metrics_events(session, metrics, call_id)
    ctx.add_shutdown_callback(lambda: finalize_metrics(ctx, metrics, call_id))

    if phone_number:
        logger.info(f"Initiating outbound SIP call to {phone_number}...")
        await update_call_status(call_id, "dialing", phone_number)
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=os.getenv("OUTBOUND_TRUNK_ID"),
                    sip_call_to=phone_number,
                    participant_identity=f"sip_{phone_number}",
                    wait_until_answered=True,
                )
            )
            logger.info("Call answered! Agent is now listening.")
            await update_call_status(call_id, "answered", phone_number)
            
            # Greet the user as soon as they pick up!
            await session.generate_reply(instructions="Hello! This is a test from Vobiz. How can I assist you today?")
            
        except Exception as e:
            logger.error(f"Failed to place outbound call: {e}")
            error_status = "no-answer" if "timeout" in str(e).lower() else "failed"
            await update_call_status(call_id, error_status, phone_number)
            ctx.shutdown()
    else:
        await session.generate_reply(instructions="Greet the user.")

def run_fastapi():
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    port = int(base_url.split(":")[-1].split("/")[0])
    logger.info(f"Starting trigger API on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")

async def outbound_request_fnc(req: agents.JobRequest) -> None:
    """Only accept explicit API outbound calls (call-*) and ignore inbound SIP calls."""
    if req.job.room.name.startswith("call-"):
        await req.accept()
    else:
        await req.reject()


if __name__ == "__main__":
    # 1. Start FastAPI server in a background thread
    threading.Thread(target=run_fastapi, daemon=True).start()

    # 2. Run the LiveKit worker
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=outbound_entrypoint,
            agent_name=os.getenv("OUTBOUND_AGENT_ID", "outbound-caller"),
            request_fnc=outbound_request_fnc,
            port=0,
        )
    )
