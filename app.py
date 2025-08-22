# app.py
import logging
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
from starlette.concurrency import run_in_threadpool
from threading import Lock
import time

# --- Config (edit model path as needed) ---
MODEL_GGUF_FILE = ""
SYSTEM_PROMPT = "You're a helpful assistant. Be concise and accurate."
LOG_FILE = "llamacpp.log"
MAX_HISTORY_MESSAGES = 10   # avoid unbounded growth
SESSION_TTL = 1800          # optional: seconds before idle sessions expire (30 min)
# ------------------------------------------

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="LFM2-1 REST API")

# Allow CORS for testing from browser; lock down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session-specific models + histories ---
session_models: dict = {}       # session_id -> Llama
session_histories: dict = {}    # session_id -> list of messages
session_timestamps: dict = {}   # session_id -> last access time
session_lock = Lock()
# -------------------------------------------


class GenerateResponse(BaseModel):
    response: str
    session: Optional[str] = None
    ok: bool = True


def get_llm_for_session(session_id: str) -> Llama:
    """Return (or load) a model instance for this session."""
    with session_lock:
        if session_id not in session_models:
            logging.info(f"Loading new model instance for session {session_id}")
            session_models[session_id] = Llama(model_path=MODEL_GGUF_FILE, n_ctx=512)
            # Initialize history with system prompt
            session_histories[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        session_timestamps[session_id] = time.time()
        return session_models[session_id]


def cleanup_sessions():
    """Optional: unload sessions idle for too long."""
    now = time.time()
    with session_lock:
        expired = [sid for sid, ts in session_timestamps.items() if now - ts > SESSION_TTL]
        for sid in expired:
            logging.info(f"Cleaning up expired session {sid}")
            # free history + model reference
            session_histories.pop(sid, None)
            session_models.pop(sid, None)
            session_timestamps.pop(sid, None)


@app.get("/generate", response_model=GenerateResponse)
async def generate(
    prompt: str = Query(..., min_length=1),
    session: Optional[str] = Query(None),
    reset: Optional[bool] = Query(False)
):
    """
    Generate assistant response.
    """
    if not session:
        session = "default"   # fallback if none provided

    llm = get_llm_for_session(session)

    # Reset or append to history
    with session_lock:
        if reset or session not in session_histories:
            session_histories[session] = [{"role": "system", "content": SYSTEM_PROMPT}]
        history = session_histories[session]
        history.append({"role": "user", "content": prompt})

        # Trim history
        if len(history) > MAX_HISTORY_MESSAGES:
            history = [history[0]] + history[-(MAX_HISTORY_MESSAGES - 1):]
            session_histories[session] = history

    # Run inference (per-session model, so no global lock needed)
    try:
        result = await run_in_threadpool(
            lambda: llm.create_chat_completion(messages=history)
        )
    except Exception as exc:
        logging.exception("Model inference error")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    try:
        generated_message = result["choices"][0]["message"]["content"]
    except Exception:
        logging.exception("Unexpected model output structure")
        raise HTTPException(status_code=500, detail="Unexpected model output format")

    # Save assistant reply
    with session_lock:
        session_histories[session].append({"role": "assistant", "content": generated_message})
        if len(session_histories[session]) > MAX_HISTORY_MESSAGES:
            session_histories[session] = [session_histories[session][0]] + session_histories[session][- (MAX_HISTORY_MESSAGES - 1):]

    # Periodic cleanup (optional)
    cleanup_sessions()

    return GenerateResponse(response=generated_message, session=session, ok=True)


@app.get("/health")
def health():
    with session_lock:
        return {
            "ok": True,
            "active_sessions": len(session_models),
            "sessions": list(session_models.keys())
        }

