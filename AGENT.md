# AGENT.md

## Overview
This repository is a Python-based speech-to-text workflow built around Faster-Whisper.

It currently supports:
- Batch transcription of WAV files from `Input/` via `transcribe_file.py`
- Optional transcript summarization using an Ollama-compatible chat endpoint
- An archived real-time microphone transcription pipeline in `archive/whisper.py`

The code is optimized for Windows + NVIDIA CUDA and includes explicit teardown logic to avoid native shutdown crashes.

## Architecture

### 1) Batch Pipeline (Primary)
Entry point: `transcribe_file.py`

Core flow:
1. Parse CLI flags (`--separate`)
2. Initialize a global Faster-Whisper model (`large-v3`, CUDA FP16)
3. Scan `Input/` for `.wav` files
4. "Convert" step (`convert_audio_files`):
   - Currently does not run ffmpeg (conversion call is intentionally bypassed)
   - Moves source WAVs from `Input/` to `Converted/`
5. Process WAVs from `Converted/` (`process_files`):
   - Transcribe with Whisper
   - Deduplicate repeated adjacent segments
   - Write transcript(s) to `Converted/`
   - Move processed WAVs to `Transcribed/`
6. Summarize transcript text with Ollama API (`summarize`)
7. Cleanup model/CUDA handles and force-safe process exit

Transcript modes:
- Combined mode (default): one timestamped transcript file for all files in the run
- Separate mode (`--separate`): one transcript per input file (`transcript_<basename>.txt`)

### 2) Real-Time Pipeline (Archived)
Entry point: `archive/whisper.py`

This script is legacy/archived but documents a second architecture:
- Microphone capture via PyAudio callback
- Voice activity detection via WebRTC VAD
- Wake-word gate (default: "Teresa")
- Segment transcription via Faster-Whisper
- Optional keystroke injection into active app (`pyautogui`)
- Single-instance lock file (`whisper.lock`)

This path is useful as reference for low-latency interactive transcription features.

## Technology Stack

### Runtime
- Python 3.x
- Windows (primary tested environment)

### ML / Speech
- `faster-whisper==1.1.1`
- Whisper model: `large-v3`
- GPU execution: `device="cuda"`, `compute_type="float16"`

### Networking / API
- `requests==2.32.3`
- Ollama-compatible chat endpoint (configured in code):
  - URL: `https://ollama.fleming.ai/api/chat`
  - Model: `gemma4:latest`

### Utility Dependencies
- `tqdm==4.67.1`
- `argcomplete==3.5.1`

### Archived Real-Time Dependencies
Used by `archive/whisper.py` (not included in active requirements list):
- `pyaudio`
- `webrtcvad`
- `pyautogui`
- `keyboard`
- `numpy`

## Repository Structure
- `transcribe_file.py`: active batch transcription + summarization pipeline
- `archive/whisper.py`: archived real-time VAD/wake-word transcription pipeline
- `Input/`: incoming WAV files to process
- `Converted/`: staging area for WAVs and generated transcript outputs
- `Transcribed/`: processed WAV archive
- `requirements.txt`: pinned runtime dependencies for active pipeline
- `requirements.in`: currently empty

## Operational Notes
- The batch script currently assumes WAV input files and does not perform active transcoding.
- Summarization is done in chunks (32 KB bytes per chunk) to keep prompt payload size manageable.
- Windows stability workaround is implemented:
  - explicit model cleanup
  - explicit CUDA DLL unloading
  - `os._exit(code)` on shutdown to avoid interpreter teardown crashes

## Extension Points
- Re-enable ffmpeg conversion in `convert_audio_files` for mixed-format inputs.
- Externalize runtime settings (model name, Ollama URL/model, directories) into a config file or env vars.
- Add retry/backoff and timeout handling for summary API calls.
- Add test coverage around file movement and transcript generation edge cases.
- Promote real-time pipeline from `archive/` into a supported module if interactive dictation is still a product goal.
