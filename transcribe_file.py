import os
import shutil
import subprocess
import sys
import gc
from datetime import datetime
from tqdm import tqdm
import requests
from faster_whisper import WhisperModel
import argparse
from pathlib import Path

input_dir = "Input"
output_dir = "Converted"
model_size = "large-v3"
ollama_url = "https://ollama.fleming.ai/api/chat"
ollama_model = "gemma4:latest"
# NAME             ID              SIZE      MODIFIED
# gemma4:31b       6316f0629137    19 GB     4 hours ago
# llama3.2:3b      a80c4f17acd5    2.0 GB    4 hours ago
# qwen3.6:35b      07d35212591f    23 GB     4 hours ago
# gemma4:latest    c6eb396dbd59    9.6 GB    21 hours ago

chunk_size = 32 * 1024  # 32KB

# Parse command line arguments
parser = argparse.ArgumentParser(description='Transcribe audio files using Whisper.')
parser.add_argument('--separate', action='store_true', help='Create separate transcripts for each file instead of a single combined transcript')
args = parser.parse_args()
separate_transcripts = args.separate

# Generate a timestamped transcript file name if not separate
if not separate_transcripts:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    transcript_file = os.path.join(output_dir, f"transcript_{timestamp}.txt")

# Run on GPU with FP16 
whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")


def _unload_cuda_dlls():
    """Unload CUDA DLLs that ctranslate2 loaded via ctypes.CDLL().
    Prevents the Win32 crash during Python shutdown on Windows.
    DLLs must be freed in reverse dependency order."""
    try:
        from ctypes import windll, c_wchar_p, c_ulong
        GetModuleHandleW = windll.kernel32.GetModuleHandleW
        GetModuleHandleW.argtypes = [c_wchar_p]
        GetModuleHandleW.restype = c_ulong
        FreeLibrary = windll.kernel32.FreeLibrary
        FreeLibrary.argtypes = [c_ulong]
        FreeLibrary.restype = c_ulong

        # Reverse dependency order: cusolver depends on cublas/cusparse, etc.
        dlls = [
            "cusolver64_64.dll",
            "cublasLt64_64.dll",
            "cublas64_64.dll",
            "cusparse64_64.dll",
            "cudart64_64.dll",
        ]
        for dll in dlls:
            try:
                handle = GetModuleHandleW(dll)
                if handle:
                    FreeLibrary(handle)
            except Exception:
                pass
    except Exception:
        pass


def cleanup_whisper_model():
    global whisper_model
    if whisper_model is None:
        return
    try:
        model_ref = whisper_model
        whisper_model = None
        del model_ref
        gc.collect()
    except Exception:
        pass
    _unload_cuda_dlls()


def safe_exit(code=0):
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)

def transcribe_file(filename):
    global whisper_model, transcript_file, separate_transcripts
    print(f"\nTranscribing file: {filename}")

    if separate_transcripts:
        base_name = os.path.splitext(os.path.basename(filename))[0]
        current_transcript_file = os.path.join(output_dir, f"transcript_{base_name}.txt")
    else:
        current_transcript_file = transcript_file

    segments, info = whisper_model.transcribe(filename, beam_size=5, language="en")
    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    output_lines = []
    repeated = ""
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        if (segment.text.strip() != repeated):
            repeated = segment.text.strip()
            output_lines.append(segment.text)
            output_lines.append("\n")

    # Write to the transcript file
    mode = "w" if separate_transcripts else "a"
    with open(current_transcript_file, mode, encoding="utf-8") as f:
        f.writelines(output_lines)

    if separate_transcripts:
        # Summarize the transcript
        summarize(current_transcript_file)

def summarize(filename):
    global chunk_size
    print(f"\nSummarizing transcript file: {filename}")

    with open(filename, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    if len(transcript_text.encode("utf-8")) <= chunk_size:
        chunks = [transcript_text]
    else:
        # Split at line breaks, keeping each chunk <= chunk_size bytes
        lines = transcript_text.splitlines(keepends=True)
        chunks = []
        current_chunk = ""
        for line in lines:
            if len((current_chunk + line).encode("utf-8")) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            current_chunk += line
        if current_chunk:
            chunks.append(current_chunk)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Summarizing chunk {i+1}/{len(chunks)} ---\n")
        response = requests.post(ollama_url, json={
            "model": ollama_model,
            "messages": [
                {"role": "user", "content": f"Summarize the following transcript chunk focusing on the key points and takeaways:\n\n{chunk}"}
            ],
            "stream": False
        })
        summary = response.json()["message"]["content"]
        print(summary)
        with open(filename, "a", encoding="utf-8") as f:
            f.writelines(["\n--- Summary of chunk {} ---\n".format(i+1), summary, "\n"])
        

def convert_audio_files():
    global input_dir, output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    # Convert all .wav files in the input directory to the desired format    
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(".wav"):
            input_path = os.path.join( os.path.abspath(input_dir), filename)
            output_path = os.path.join(output_dir, filename.replace(".WAV", ".wav"))

            # Force output to PCM 16-bit, mono, 16kHz
            cmd = [
                "ffmpeg",
                "-y",  # overwrite without asking
                "-i", input_path,
                "-acodec", "pcm_s16le",  # 16-bit PCM
                "-ac", "1",              # mono
                "-ar", "16000",          # 16kHz sample rate
                output_path
            ]

            try:
                # New recorder doesn't need conversion; simulate success
                # subprocess.run(cmd, check=True)
                pass
            except subprocess.CalledProcessError:
                print(f"Error converting {filename}")

            move_to_converted(input_path)

def process_files():
    global input_dir, output_dir, separate_transcripts, transcript_file
    convert_audio_files()

    for filename in tqdm(list(Path(output_dir).glob("*.wav"))):
        if filename.suffix.lower() == ".wav":
            file_path = str(filename)
            try:
                transcribe_file(file_path)
            except Exception as e:
                print(f"Error transcribing {filename}: {e}")
                continue

            move_to_transcribed(file_path)

    if not separate_transcripts:
        summarize(transcript_file)

def move_to_converted(src_path):
    converted_dir = os.path.join(os.path.dirname(src_path), '..', 'Converted')
    converted_dir = os.path.abspath(converted_dir)
    os.makedirs(converted_dir, exist_ok=True)
    dst_path = os.path.join(converted_dir, os.path.basename(src_path))
    print(f"Converted: Moving {src_path} to {dst_path} without conversion")
    shutil.move(src_path, dst_path)
    print(f"Moved {src_path} to {dst_path}")

def move_to_transcribed(src_path):
    transcribed_dir = os.path.join(os.path.dirname(src_path), '..', 'Transcribed')
    transcribed_dir = os.path.abspath(transcribed_dir)
    os.makedirs(transcribed_dir, exist_ok=True)
    dst_path = os.path.join(transcribed_dir, os.path.basename(src_path))
    print(f"Transcribed: Moving {src_path} to {dst_path} without conversion")
    shutil.move(src_path, dst_path)
    print(f"Moved {src_path} to {dst_path}")


def main():
    exit_code = 0
    try:
        process_files()
    except Exception as e:
        print(f"Fatal error: {e}")
        exit_code = 1
    finally:
        # Avoid native teardown crashes in Windows by disposing model before process exit.
        cleanup_whisper_model()
        safe_exit(exit_code)


if __name__ == "__main__":
    main()
# summarize("Converted/transcript_20260422-161436.txt")