import asyncio
import logging
import pickle
from dotenv import load_dotenv
import json
import os
from time import perf_counter
from typing import Annotated

from livekit import rtc, api
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, rag, silero, elevenlabs, cartesia
from dotenv import load_dotenv  

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("phone-rag-agent")
logger.setLevel(logging.INFO)


# Load RAG resources
annoy_index = rag.annoy.AnnoyIndex.load("vdb_data")  # see build_data.py
embeddings_dimension = 1536
with open("my_data.pkl", "rb") as f:
    paragraphs_by_uuid = pickle.load(f)

# RAG enrichment callback: it takes the last user message, computes its embedding,
# queries the ANN index, and injects retrieved context into the chat context.
async def _enrich_with_rag(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext):
    user_msg = chat_ctx.messages[-1]
    user_embedding = await openai.create_embeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        input=[user_msg.content],
        model="text-embedding-3-small",
        dimensions=embeddings_dimension,
    )
    result = annoy_index.query(user_embedding[0].embedding, n=1)[0]
    paragraph = paragraphs_by_uuid[result.userdata]
    if paragraph:
        logger.info("enriching with RAG:")
        rag_msg = llm.ChatMessage.create(
            text="Context:\n" + paragraph,
            role="assistant",
        )
        # Replace the last user message with the enriched context and re-append the user message.
        chat_ctx.messages[-1] = rag_msg
        chat_ctx.messages.append(user_msg)

# SIP configuration
outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")
_default_instructions = (
    "You are a customer service assistant. Your interface with user will be voice. "
    "You will be on a call with a customer who have many questions regarding NVIDIA Store. Your goal is to solve the user queries. "
    "As a customer service representative, you will be polite and professional at all times. Allow user to end the conversation."
)

async def entrypoint(ctx: JobContext):
    global _default_instructions, outbound_trunk_id
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    user_identity = "phone_user"
    # The phone number to dial is provided in the job metadata.
    phone_number = ctx.job.metadata
    logger.info(f"dialing {phone_number} to room {ctx.room.name}")

    # Build the instructions with appointment details.
    instructions = (
        _default_instructions
        + " The customer's name is Kyle. His appointment is next Tuesday at 3pm for visiting store."
    )

    # Create a SIP participant to start dialing.
    await ctx.api.sip.create_sip_participant(
        api.CreateSIPParticipantRequest(
            room_name=ctx.room.name,
            sip_trunk_id=outbound_trunk_id,
            sip_call_to=phone_number,
            participant_identity=user_identity,
        )
    )
    
    # Wait for the participant to join the room.
    participant = await ctx.wait_for_participant(identity=user_identity)

    # Start the voice pipeline agent (which now uses the RAG enrichment callback).
    run_voice_pipeline_agent(ctx, participant, instructions)

    # Monitor the call status for up to 30 seconds.
    start_time = perf_counter()

    while perf_counter() - start_time < 30:
        call_status = participant.attributes.get("sip.callStatus")
        if call_status == "active":
            logger.info("user has picked up")
            return
        elif call_status == "automation":
            # If DTMF is used for dialing (extension or PIN), participant may be in 'automation'.
            pass
        elif participant.disconnect_reason == rtc.DisconnectReason.USER_REJECTED:
            logger.info("user rejected the call, exiting job")
            break
        elif participant.disconnect_reason == rtc.DisconnectReason.USER_UNAVAILABLE:
            logger.info("user did not pick up, exiting job")
            break
        await asyncio.sleep(0.1)

    logger.info("session timed out, exiting job")
    ctx.shutdown()

class CallActions(llm.FunctionContext):
    """
    Contains callable functions to manage call actions.
    """
    def __init__(self, *, api: api.LiveKitAPI, participant: rtc.RemoteParticipant, room: rtc.Room):
        super().__init__()
        self.api = api
        self.participant = participant
        self.room = room

    async def hangup(self):
        try:
            await self.api.room.remove_participant(
                api.RoomParticipantIdentity(
                    room=self.room.name,
                    identity=self.participant.identity,
                )
            )
        except Exception as e:
            # The participant may have already hung up.
            logger.info(f"received error while ending call: {e}")

    @llm.ai_callable()
    async def end_call(self):
        """Called when the user wants to end the call."""
        logger.info(f"ending the call for {self.participant.identity}")
        await self.hangup()

    @llm.ai_callable()
    async def look_up_availability(self, date: Annotated[str, "The date of the appointment to check availability for"]):
        """Called when the user asks about alternative appointment availability."""
        logger.info(f"looking up availability for {self.participant.identity} on {date}")
        await asyncio.sleep(3)
        return json.dumps({
            "available_times": ["1pm", "2pm", "3pm"],
        })

    @llm.ai_callable()
    async def confirm_appointment(self, date: Annotated[str, "date of the appointment"], time: Annotated[str, "time of the appointment"]):
        """Called when the user confirms their appointment."""
        logger.info(f"confirming appointment for {self.participant.identity} on {date} at {time}")
        return "reservation confirmed"

    @llm.ai_callable()
    async def detected_answering_machine(self):
        """Called when an answering machine is detected."""
        logger.info(f"detected answering machine for {self.participant.identity}")
        await self.hangup()

def run_voice_pipeline_agent(ctx: JobContext, participant: rtc.RemoteParticipant, instructions: str):
    logger.info("starting voice pipeline agent")
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=instructions,
    )

    # Instantiate the VoicePipelineAgent with the RAG enrichment callback added.
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-2-phonecall"),
        llm=openai.LLM(),
        # tts=openai.TTS(),
        # tts=deepgram.TTS(),
        tts=elevenlabs.TTS(),
        chat_ctx=initial_ctx,
        fnc_ctx=CallActions(api=ctx.api, participant=participant, room=ctx.room),
        before_llm_cb=_enrich_with_rag,  # RAG callback 
    )
    agent.start(ctx.room, participant)

def prewarm(proc: JobProcess):
    # Prewarm the VAD model for faster call response.
    proc.userdata["vad"] = silero.VAD.load()

if __name__ == "__main__":
    if not outbound_trunk_id or not outbound_trunk_id.startswith("ST_"):
        raise ValueError("SIP_OUTBOUND_TRUNK_ID is not set")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller",
            prewarm_fnc=prewarm,
        )
    )
