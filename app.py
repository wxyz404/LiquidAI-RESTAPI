# app.py
import logging
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
from starlette.concurrency import run_in_threadpool
from threading import Lock

# --- Config (edit model path as needed) ---
MODEL_GGUF_FILE = ""
SYSTEM_PROMPT = "You're are helpful assistant. Be concise and accurate."
LOG_FILE = "llamacpp.log"
# ------------------------------------------

# Redirect llama.cpp stderr to a log file (like your original script)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="LFM2-1 REST API")

# Allow CORS for testing from browser; lock down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and lock
llm: Optional[Llama] = None
model_lock = Lock()

# Optional in-memory session histories (simple, not persistent)
# session_id -> list[dict] messages
sessions: dict = {}
sessions_lock = Lock()
MAX_HISTORY_MESSAGES = 10  # avoid unbounded growth


class GenerateResponse(BaseModel):
    response: str
    session: Optional[str] = None
    ok: bool = True


@app.on_event("startup")
def load_model():
    global llm
    try:
        logging.info(f"Loading model from {MODEL_GGUF_FILE}")
        # load model once
        llm = Llama(model_path=MODEL_GGUF_FILE, n_ctx=512)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.exception("Failed loading model at startup.")
        # re-raise so uvicorn shows the problem early
        raise RuntimeError(f"Failed to load Llama model: {e}") from e


@app.get("/generate", response_model=GenerateResponse)
async def generate(
    prompt: str = Query(..., min_length=1),
    session: Optional[str] = Query(None),
    reset: Optional[bool] = Query(False)
):
    """
    Generate assistant response.

    Query params:
      - prompt: user prompt (required)
      - session: optional session id to keep conversation history (string). If omitted, runs statelessly.
      - reset: optional boolean; if true and session provided, clears that session's history before processing.
    """
    global llm
    if llm is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Build conversation history
    if session:
        with sessions_lock:
            if reset or session not in sessions:
                # initialize with system prompt
                sessions[session] = [{"role": "system", "content": SYSTEM_PROMPT}]
            history = sessions[session]
            history.append({"role": "user", "content": prompt})

            # Trim history if too long
            if len(history) > MAX_HISTORY_MESSAGES:
                history = [history[0]] + history[-(MAX_HISTORY_MESSAGES - 1):]
                sessions[session] = history
    else:
        history = [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": prompt}]

    # Call the model safely under lock
    try:
        with model_lock:
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

    # Save assistant response to history
    if session:
        with sessions_lock:
            sessions[session].append({"role": "assistant", "content": generated_message})
            if len(sessions[session]) > MAX_HISTORY_MESSAGES:
                sessions[session] = [sessions[session][0]] + sessions[session][- (MAX_HISTORY_MESSAGES - 1):]

    return GenerateResponse(response=generated_message, session=session or None, ok=True)


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": llm is not None}

