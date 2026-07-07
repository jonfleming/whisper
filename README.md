## Whisper ASR

usage: whisper.py [-h] [-f] [-w WAKE_WORD]

Real-time audio transcription with VAD and FasterWhisper.

options:
  -h, --help            show this help message and exit
  -f, --force           Force bypass the lock file test.
  -w WAKE_WORD,         Set custom wake word (use 'none' to disable)
  --wake-word WAKE_WORD Default wake word is 'Teresa'  

  ## Transccribe

## Web Service

This repository now includes a FastAPI service for uploading an audio file and
receiving both transcription and summary in JSON.

### Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the service:

```bash
uvicorn web_service:app --host 0.0.0.0 --port 7000
```

### Endpoints

- `GET /health`: health check
- `POST /transcribe`: upload one audio file and get transcription + summary

Example request:

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -F "file=@Input/example.wav"
```

### Environment Variables

- `WHISPER_MODEL_SIZE` (default: `large-v3`)
- `WHISPER_DEVICE` (default: `cuda`)
- `WHISPER_COMPUTE_TYPE` (default: `float16`)
- `WHISPER_LANGUAGE` (default: `en`)
- `OLLAMA_URL` (default: `https://ollama.fleming.ai/api/chat`)
- `OLLAMA_MODEL` (default: `gemma4:latest`)
- `SUMMARY_CHUNK_SIZE` (default: `32768` bytes)
- `OLLAMA_TIMEOUT_SECONDS` (default: `180`)
  
