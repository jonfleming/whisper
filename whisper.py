import pyaudio
import wave
from faster_whisper import WhisperModel

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
OUTPUT_FILENAME = "recorded_audio.wav"

# Record audio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* Recording...")

frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* Done recording")

stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded data as a WAV file
wf = wave.open(OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# Load the Whisper model and transcribe
print("* Transcribing...")
model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe(OUTPUT_FILENAME, beam_size=5)

# Print the transcription
print("\nTranscription:")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")