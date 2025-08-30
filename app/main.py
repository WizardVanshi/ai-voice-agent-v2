import os
import asyncio
import json
import logging
import base64
import websockets
from typing import Type, Dict, List
import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TurnEvent,
)
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
import uuid
import traceback
from concurrent.futures import ThreadPoolExecutor
import requests

# Murf SDK (optional but used for TTS)
try:
    from murf import Murf
    MURF_AVAILABLE = True
except ImportError:
    MURF_AVAILABLE = False
    print("Warning: Murf SDK not installed. Install with: pip install murf")

# ==========================
# Data Models
# ==========================
@dataclass
class ChatMessage:
    id: str
    session_id: str
    timestamp: datetime
    role: str
    content: str
    audio_base64: str = None

class ChatHistory:
    def __init__(self):
        self.sessions: Dict[str, List[ChatMessage]] = defaultdict(list)

    def add_message(self, session_id: str, role: str, content: str, **kwargs) -> ChatMessage:
        message = ChatMessage(id=str(uuid.uuid4()), session_id=session_id, timestamp=datetime.now(), role=role, content=content, **kwargs)
        self.sessions[session_id].append(message)
        return message

    def get_conversation_context(self, session_id: str, max_messages: int = 10) -> str:
        messages = self.sessions[session_id][-max_messages:]
        return "\n".join([f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}" for msg in messages])

# ==========================
# Config & Setup
# ==========================
load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("voice-agent")

app = FastAPI()
try:
    templates = Jinja2Templates(directory="templates")
    TEMPLATES_AVAILABLE = True
except Exception:
    TEMPLATES_AVAILABLE = False

chat_history = ChatHistory()
AUDIO_SAMPLE_RATE = 16000
active_sessions: Dict[str, dict] = {}
executor = ThreadPoolExecutor(max_workers=4)

# ==========================
# Persona Configuration
# ==========================
AVAILABLE_PERSONAS = {
    "doraemon": {
        "name": "Doraemon",
        "system_instruction": "You are Doraemon, a friendly, helpful robotic cat from the 22nd century who loves Dorayaki and is afraid of mice. Keep responses short and cheerful, and talk about your futuristic gadgets.",
        "voice_id": "en-US-ryan",
        "greeting": "Hello! I'm Doraemon! I came from the 22nd century to help! Do you need a gadget from my pocket?"
    },
    "nobita": {
        "name": "Nobita Nobi",
        "system_instruction": "You are Nobita Nobi, a lazy but kind-hearted student. You complain a lot about homework and bullies. Keep responses short, whiny, but hopeful.",
        "voice_id": "en-US-ryan",
        "greeting": "Oh, hello... My homework is so hard! Can you help me?"
    },
    "shizuka": {
        "name": "Shizuka Minamoto",
        "system_instruction": "You are Shizuka, a sweet, polite, and smart girl. Be gentle, caring, and encouraging. Keep responses short and kind.",
        "voice_id": "en-US-natalie",
        "greeting": "Hello there! It's so nice to talk to you. How are you today?"
    },
    "gian": {
        "name": "Takeshi 'Gian' Goda",
        "system_instruction": "You are Gian, a big, strong, and bossy character, but with a good heart. You are loud, confident, and love singing. Keep responses short and forceful.",
        "voice_id": "en-US-ryan",
        "greeting": "Hey! I'm Gian! Want to hear my new song? It's great!"
    },
    "suneo": {
        "name": "Suneo Honekawa",
        "system_instruction": "You are Suneo, a rich, spoiled, and boastful boy. You love to show off your toys and trips. Keep responses short, arrogant, and materialistic.",
        "voice_id": "en-US-ryan",
        "greeting": "Hello! My dad just bought me a new drone from France. It's very expensive!"
    }
}
CURRENT_PERSONA = "doraemon"

# ==========================
# Helper: safe text extraction from Gemini responses
# ==========================
def safe_get_text(response, fallback="...I'm thinking..."):
    """
    Extracts text parts from a Gemini GenerateContentResponse safely.
    Returns fallback if no usable text parts exist.
    """
    try:
        # If the SDK provides a .text quick accessor sometimes it's empty; we iterate candidates/parts robustly.
        if response and getattr(response, "candidates", None):
            for candidate in response.candidates:
                content = getattr(candidate, "content", None)
                if content and getattr(content, "parts", None):
                    parts = []
                    for p in content.parts:
                        # Some part objects have .text; some may differ across SDK versions, check safely
                        if hasattr(p, "text") and p.text:
                            parts.append(p.text)
                    if parts:
                        return " ".join(parts).strip()
        return fallback
    except Exception as e:
        logger.error(f"❌ safe_get_text error: {e}")
        return fallback

# ==========================
# Core Functions
# ==========================
async def get_llm_response(user_input: str, session_id: str, api_keys: dict, persona_id: str) -> str:
    """LLM response for a given persona. Uses safe_get_text and runs generation in the threadpool."""
    gemini_api_key = api_keys.get("gemini")
    if not gemini_api_key:
        return "Google Gemini API key missing."
    try:
        if persona_id not in AVAILABLE_PERSONAS:
            return f"Persona '{persona_id}' not found."
        persona = AVAILABLE_PERSONAS[persona_id]
        context = chat_history.get_conversation_context(session_id)
        # Prepend the persona instruction to the prompt so behavior is preserved across SDK versions
        enhanced_prompt = f"{persona['system_instruction']}\n\nConversation History:\n{context}\n\nUser: {user_input}\n\nRespond concisely in character."

        # configure per-call
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(model_name="gemini-2.5-flash", system_instruction=persona['system_instruction'])

        # generation config as dict to avoid SDK version issues
        generation_cfg = {
            "temperature": 0.8,
            "max_output_tokens": 1024,
            "top_p": 0.95,
            "top_k": 40
        }

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(executor, lambda: model.generate_content(enhanced_prompt, generation_config=generation_cfg))
        logger.info(f"Raw Gemini response (get_llm_response): {response}")
        text = safe_get_text(response, fallback=f"{persona['name']} seems to be thinking...")
        return text
    except Exception as e:
        logger.error(f"Gemini error: {e}\n{traceback.format_exc()}")
        return "I'm having trouble thinking. Please check the Gemini API key."

def _generate_tts_sync(text: str, voice_id: str, api_key: str) -> str:
    """Blocking Murf call wrapped for executor."""
    try:
        client = Murf(api_key=api_key)
        clean = (text or "").strip()
        if not clean:
            return None
        res = client.text_to_speech.generate(
            text=clean,
            voice_id=voice_id,
            encode_as_base_64=True
        )
        return res.encoded_audio
    except Exception as e:
        logger.error(f"TTS sync error: {e}\n{traceback.format_exc()}")
        return None

async def generate_tts_audio(text: str, api_keys: dict, persona_id: str, voice_id: str = None) -> str:
    """Generate TTS via Murf for the persona."""
    murf_api_key = api_keys.get("murf")
    if not murf_api_key or not MURF_AVAILABLE:
        logger.debug("Murf SDK not available or key missing; returning None for audio")
        return None
    try:
        if persona_id not in AVAILABLE_PERSONAS:
            logger.error(f"TTS error: Persona '{persona_id}' not found.")
            return None
        persona = AVAILABLE_PERSONAS[persona_id]
        vid = voice_id or persona['voice_id']
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, _generate_tts_sync, text, vid, murf_api_key)
    except Exception as e:
        logger.error(f"TTS error: {e}\n{traceback.format_exc()}")
        return None

def fetch_latest_news(api_key: str):
    """Fetch a few top headlines via NewsAPI.org"""
    if not api_key:
        return ["News API key missing."]
    try:
        url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=3&apiKey={api_key}"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        if data.get("status") == "ok":
            articles = data.get("articles", [])
            return [a['title'] for a in articles if a.get('title')] or ["No headlines found."]
        return [f"Could not fetch news: {data.get('message', 'Unknown error')}"]
    except requests.exceptions.HTTPError as e:
        return [f"News API error: {e.response.status_code}. Check API key."]
    except Exception as e:
        return [f"Error fetching news: {e}"]

# ==========================
# Conversation Pipeline (WebSocket)
# ==========================
async def process_complete_conversation(transcript: str, session_id: str, websocket: WebSocket, api_keys: dict):
    try:
        await websocket.send_json({"type": "processing_started", "text": transcript})
        chat_history.add_message(session_id=session_id, role="user", content=transcript)

        llm_response = await get_llm_response(transcript, session_id, api_keys, CURRENT_PERSONA)
        base64_audio = await generate_tts_audio(llm_response, api_keys, CURRENT_PERSONA)

        chat_history.add_message(session_id=session_id, role="assistant", content=llm_response, audio_base64=base64_audio)

        await websocket.send_json({
            "type": "assistant_response",
            "text": llm_response,
            "audio": base64_audio,
            "persona": CURRENT_PERSONA,
        })
    except Exception as e:
        logger.error(f"Pipeline error: {e}\n{traceback.format_exc()}")
        try:
            await websocket.send_json({"type": "error", "message": "An error occurred in the conversation pipeline."})
        except Exception:
            pass

# ==========================
# FastAPI Routes
# ==========================
@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    if TEMPLATES_AVAILABLE:
        return templates.TemplateResponse("index.html", {"request": request})
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

@app.get("/api/personas")
async def get_personas():
    return {"current_persona": CURRENT_PERSONA}

@app.post("/api/persona/greeting")
async def get_greeting(request: Request):
    data = await request.json()
    api_keys = data.get("api_keys", {})
    persona_id = data.get("persona_id", CURRENT_PERSONA)

    if persona_id not in AVAILABLE_PERSONAS:
        return {"success": False, "error": f"Persona '{persona_id}' not found."}
    persona = AVAILABLE_PERSONAS[persona_id]
    audio_b64 = await generate_tts_audio(persona["greeting"], api_keys, persona_id)
    return {"success": True, "greeting": persona["greeting"], "audio": audio_b64, "persona": persona_id}

@app.post("/api/persona/standup")
async def get_standup_comedy(request: Request):
    data = await request.json()
    api_keys = data.get("api_keys", {})
    persona_id = data.get("persona_id", CURRENT_PERSONA)
    if persona_id not in AVAILABLE_PERSONAS:
        return {"success": False, "error": f"Persona '{persona_id}' not found."}
    persona = AVAILABLE_PERSONAS[persona_id]

    try:
        prompt = f"""
You are {persona['name']} from Doraemon. Perform a short stand-up comedy routine.
- It should have at least 3 funny jokes in a conversational style.
- Make it kid-friendly, lighthearted, and witty.
- End with a punchline.
Example:
🎤 Hey everyone, I'm {persona['name']}! ...
🤣 Joke 1...
🎉 Thanks for listening!
"""

        # configure and run Gemini in threadpool (non-blocking)
        genai.configure(api_key=api_keys.get("gemini"))
        model = genai.GenerativeModel(model_name="gemini-2.5-flash", system_instruction=persona['system_instruction'])
        generation_cfg = {
            "temperature": 0.9,
            "max_output_tokens": 1024,
            "top_p": 0.9,
            "top_k": 50
        }

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(executor, lambda: model.generate_content(prompt, generation_config=generation_cfg))
        logger.info(f"[Standup] Raw Gemini response: {response}")

        comedy_text = safe_get_text(response, f"{persona['name']} is thinking of a joke...")
        audio = await generate_tts_audio(comedy_text, api_keys, persona_id, persona['voice_id'])

        return {"success": True, "joke": comedy_text, "audio": audio, "persona": persona_id}

    except Exception as e:
        logger.error(f"Standup error: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}

@app.post("/api/persona/news")
async def get_latest_news(request: Request):
    data = await request.json()
    api_keys = data.get("api_keys", {})
    persona_id = data.get("persona_id", CURRENT_PERSONA)
    if persona_id not in AVAILABLE_PERSONAS:
        return {"success": False, "error": f"Persona '{persona_id}' not found."}
    persona = AVAILABLE_PERSONAS[persona_id]

    try:
        headlines = fetch_latest_news(api_keys.get("news"))
        if not headlines:
            return {"success": False, "error": "No headlines available"}
        prompt = f"""
You are {persona['name']} from Doraemon. Summarize the **latest 3 important news headlines** 
from technology, science, or world events. Each should be 2–3 sentences long, clear, and engaging. 
Make it sound like you're casually updating a friend. It should be point wise.

Headlines:
- {'\n- '.join(headlines)}
"""
        genai.configure(api_key=api_keys.get("gemini"))
        model = genai.GenerativeModel(model_name="gemini-2.5-flash", system_instruction=persona['system_instruction'])
        generation_cfg = {
            "temperature": 0.85,
            "max_output_tokens": 3000,
            "top_p": 0.9,
            "top_k": 50
        }

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(executor, lambda: model.generate_content(prompt, generation_config=generation_cfg))
        logger.info(f"[News] Raw Gemini response: {response}")

        news_text = safe_get_text(response, f"{persona['name']} is still reading the news...")
        audio = await generate_tts_audio(news_text, api_keys, persona_id, persona['voice_id'])

        return {"success": True, "news": news_text, "audio": audio, "persona": persona_id}

    except Exception as e:
        logger.error(f"News error: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}

@app.post("/api/persona/{persona_id}")
async def set_persona(persona_id: str):
    global CURRENT_PERSONA
    if persona_id in AVAILABLE_PERSONAS:
        CURRENT_PERSONA = persona_id
        return {"success": True, "name": AVAILABLE_PERSONAS[persona_id]["name"]}
    return {"success": False, "error": "Persona not found"}

# ==========================
# WebSocket Handler (for live conversation)
# ==========================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {}

    try:
        # Expect a config message with API keys first
        initial_message = await ws.receive_json()
        if initial_message.get("type") == "config":
            api_keys = initial_message.get("api_keys", {})
            active_sessions[session_id]["api_keys"] = api_keys
            await ws.send_json({"type": "config_success"})
        else:
            raise WebSocketDisconnect

        assembly_api_key = api_keys.get("assembly")
        if not assembly_api_key:
            await ws.send_json({"type": "error", "message": "AssemblyAI API key missing."})
            raise WebSocketDisconnect

        client = StreamingClient(StreamingClientOptions(api_key=assembly_api_key))
        loop = asyncio.get_event_loop()

        def on_turn(self: Type[StreamingClient], event: TurnEvent):
            if event.transcript and getattr(event, "end_of_turn", False):
                asyncio.run_coroutine_threadsafe(
                    process_complete_conversation(event.transcript, session_id, ws, api_keys),
                    loop
                )

        def on_error(self: Type[StreamingClient], error: StreamingError):
            asyncio.run_coroutine_threadsafe(
                ws.send_json({"type": "assembly_error", "error": str(error)}),
                loop
            )

        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Error, on_error)
        client.connect(StreamingParameters(sample_rate=AUDIO_SAMPLE_RATE))

        while True:
            data = await ws.receive_bytes()
            client.stream(data)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}\n{traceback.format_exc()}")
    finally:
        if session_id in active_sessions:
            del active_sessions[session_id]
