import gc
import os
import tempfile
from pathlib import Path
from typing import Annotated

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel


MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

OLLAMA_URL = os.getenv("OLLAMA_URL", "https://ollama.fleming.ai/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:latest")
CHUNK_SIZE_BYTES = int(os.getenv("SUMMARY_CHUNK_SIZE", str(32 * 1024)))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))


app = FastAPI(title="Whisper Transcription Service", version="1.0.0")
whisper_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global whisper_model
    if whisper_model is None:
        whisper_model = WhisperModel(
            MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return whisper_model


def split_text_by_bytes(text: str, max_chunk_size: int) -> list[str]:
    if len(text.encode("utf-8")) <= max_chunk_size:
        return [text]

    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current_chunk = ""

    for line in lines:
        if len((current_chunk + line).encode("utf-8")) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""
        current_chunk += line

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def summarize_chunk(chunk: str) -> str:
    prompt = (
        "Summarize the following transcript chunk focusing on key points, decisions, "
        "action items, and concise takeaways:\n\n"
        f"{chunk}"
    )
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    try:
        return response.json()["message"]["content"].strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Unexpected response format from summarization service") from exc


def summarize_transcript(transcript_text: str) -> str:
    chunks = split_text_by_bytes(transcript_text, CHUNK_SIZE_BYTES)
    summaries: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_summary = summarize_chunk(chunk)
        if len(chunks) == 1:
            summaries.append(chunk_summary)
        else:
            summaries.append(f"Chunk {index}:\n{chunk_summary}")

    return "\n\n".join(summaries)


def transcribe_temp_file(file_path: str) -> str:
    model = get_model()
    segments, _info = model.transcribe(file_path, beam_size=5, language=WHISPER_LANGUAGE)

    output_lines: list[str] = []
    repeated = ""

    for segment in segments:
        segment_text = segment.text.strip()
        if segment_text and segment_text != repeated:
            repeated = segment_text
            output_lines.append(segment_text)

    return "\n".join(output_lines).strip()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/transcribe",
    responses={
        400: {"description": "Bad request (missing file name or empty upload)"},
        422: {"description": "No speech detected in uploaded file"},
        500: {"description": "Unexpected server-side processing failure"},
        502: {"description": "Summarization provider request failed"},
    },
)
def transcribe(file: Annotated[UploadFile, File(...)]) -> dict[str, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided")

    suffix = Path(file.filename).suffix or ".wav"
    temp_path = ""

    try:
        file_bytes = file.file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        transcription = transcribe_temp_file(temp_path)
        if not transcription:
            raise HTTPException(status_code=422, detail="No speech detected in uploaded file")

        summary = summarize_transcript(transcription)
        return {
            "filename": file.filename,
            "transcription": transcription,
            "summary": summary,
        }

    except HTTPException:
        raise
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Summarization request failed: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.on_event("shutdown")
def shutdown_cleanup() -> None:
    global whisper_model
    whisper_model = None
    gc.collect()
