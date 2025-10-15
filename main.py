from src.logger import setup_logger
setup_logger()

import time
import os
from src.trt_engine import OrpheusModelTRT
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging
import json
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import warnings
import asyncio

warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv()

logger = logging.getLogger(__name__)

def healthcheck(message: str): 
    print('Running healthcheck')
    return {'result': message}

class TTSRequest(BaseModel):
    input: str = "Hey there, looks like you forgot to provide a prompt!"
    voice: str = "tara"


class TTSStreamRequest(BaseModel):
    input: str
    voice: str = "tara"
    continue_: bool = Field(True, alias="continue")
    segment_id: str


class VoiceDetail(BaseModel):
    name: str
    description: str
    language: str
    gender: str
    accent: str
    preview_url: Optional[str] = None


class VoicesResponse(BaseModel):
    voices: List[VoiceDetail]
    default: str
    count: int
    
engine: OrpheusModelTRT = None
VOICE_DETAILS: List[VoiceDetail] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the TTS engine on application startup."""
    global engine, VOICE_DETAILS
    engine = OrpheusModelTRT()

    # Dynamically generate voice details from the loaded engine
    VOICE_DETAILS = [
        VoiceDetail(
            name=voice,
            description=f"A standard {voice} voice.",
            language="en",
            gender="unknown",
            accent="american"
        ) for voice in engine.available_voices
    ]
    yield
    # Clean up the model and other resources if needed

app = FastAPI(lifespan=lifespan)


@app.post('/v1/audio/speech/stream')
async def tts_stream(data: TTSRequest):
    """
    Generates audio speech from text in a streaming fashion.
    This endpoint is optimized for low latency (Time to First Byte).
    """
    start_time = time.perf_counter()

    async def generate_audio_stream():
        first_chunk = True
        try:
            audio_generator = engine.generate_speech_async(
                prompt=data.input,
                voice=data.voice,
            )

            async for chunk in audio_generator:
                if first_chunk:
                    ttfb = time.perf_counter() - start_time
                    logger.info(f"Time to first audio chunk (TTFB): {ttfb*1000:.2f} ms")
                    first_chunk = False
                yield chunk
        except Exception:
            logger.exception("An error occurred during audio generation")


    return StreamingResponse(generate_audio_stream(), media_type='audio/pcm')


@app.websocket("/v1/audio/speech/stream/ws")
async def tts_stream_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("connection open")
    try:
        while True:
            data = await websocket.receive_json()

            if not data.get("continue", True):
                logger.info("End of stream message received, closing connection.")
                break

            if not (input_text := data.get("input", "").strip()):
                logger.info("Empty or whitespace-only input received, skipping audio generation.")
                continue

            voice = data.get("voice", "tara")
            segment_id = data.get("segment_id", "no_segment_id")

            start_time = time.perf_counter()
            try:
                await websocket.send_json({"type": "start", "segment_id": segment_id})

                if input_text:
                    logger.info(f"Generating audio for input: '{input_text}'")
                    audio_generator = engine.generate_speech_async(
                        prompt=input_text,
                        voice=voice,
                    )

                    first_chunk = True
                    async for chunk in audio_generator:
                        if first_chunk:
                            ttfb = time.perf_counter() - start_time
                            logger.info(f"Time to first audio chunk (TTFB): {ttfb*1000:.2f} ms")
                            first_chunk = False
                        await websocket.send_bytes(chunk)
                else:
                    logger.info("Empty or whitespace-only input received, skipping audio generation.")
                
                await websocket.send_json({"type": "end", "segment_id": segment_id})

                if not data.get("continue", True):
                    await websocket.send_json({"done": True})
                    break

            except Exception as e:
                logger.exception("An error occurred during audio generation in websocket.")
                await websocket.send_json({"error": str(e), "done": True})
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected from websocket.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the websocket endpoint: {e}")
    finally:
        logger.info("Closing websocket connection.")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

@app.get("/api/voices", response_model=VoicesResponse)
async def get_voices():
    """Get available voices with detailed information."""
    default_voice = engine.available_voices[0] if engine and engine.available_voices else "tara"
    return {
        "voices": VOICE_DETAILS,
        "default": default_voice,
        "count": len(VOICE_DETAILS)
    }
