import os
import shutil
import subprocess
from datetime import datetime
from tqdm import tqdm
import requests
from faster_whisper import WhisperModel

input_dir = "Input"
output_dir = "Converted"
model_size = "large-v3"
ollama_url = "http://localhost:11434/api/chat"
ollama_model = "qwen3"
chunk_size = 32 * 1024  # 32KB

# Generate a timestamped transcript file name
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
transcript_file = f"converted/transcript_{timestamp}.txt"

# Run on GPU with FP16 
whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")

def transcribe_file(filename):
    global transcript_file, whisper_model
    print(f"\nTranscribing file: {filename}")

    segments, info = whisper_model.transcribe(filename, beam_size=5)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    output_lines = []
    repeated = ""
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        if (segment.text.strip() != repeated):
            repeated = segment.text.strip()
            output_lines.append(segment.text)
            output_lines.append("\n")

    # Write to a single output file
    with open(transcript_file, "a", encoding="utf-8") as f:
        f.writelines(output_lines)

def summarize(filename):
    global transcript_file, chunk_size
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
        with open(transcript_file, "a", encoding="utf-8") as f:
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

            move_to_processed(input_path)

def process_files():
    global input_dir, output_dir
    convert_audio_files()

    for filename in tqdm(os.listdir(output_dir)):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(output_dir, filename)
            try:
                transcribe_file(file_path)
            except Exception as e:
                print(f"Error transcribing {filename}: {e}")
                continue

            move_to_transcribed(file_path)

    summarize(transcript_file)

def move_to_processed(src_path):
    processed_dir = os.path.join(os.path.dirname(src_path), '..', 'Processed')
    processed_dir = os.path.abspath(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)
    dst_path = os.path.join(processed_dir, os.path.basename(src_path))
    print(f"Processed: Moving {src_path} to {dst_path} without conversion")
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

process_files()
