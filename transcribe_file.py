import os
import subprocess
from tqdm import tqdm
import requests
from faster_whisper import WhisperModel

input_dir = "Input"
output_dir = "Converted"
model_size = "large-v3"
transcript_file = "converted/transcript.txt"
ollama_url = "http://localhost:11434/api/chat"

def transcribe_file(filename):
    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, info = model.transcribe(filename, beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    output_lines = []
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        output_lines.append(segment.text)
        output_lines.append("\n")

    # Write to a single output file
    with open(transcript_file, "a", encoding="utf-8") as f:
        f.writelines(output_lines)

def summarize(filename):
    chunk_size = 32 * 1024  # 32KB
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
            "model": "qwen3",
            "messages": [
                {"role": "user", "content": f"Summarize the following transcript chunk:\n\n{chunk}"}
            ],
            "stream": False
        })
        summary = response.json()["message"]["content"]
        print(summary)

def convert_audio_files():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    # Convert all .wav files in the input directory to the desired format    
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

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
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"Error converting {filename}")

            transcribe_file(os.path.join(output_dir, filename))

# convert_audio_files()
summarize(transcript_file)