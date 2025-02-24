import numpy as np
import pyaudio
import pyautogui
import queue
import signal
import threading
import time
import wave
from faster_whisper import WhisperModel

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5  # Length of each chunk to process
WAVE_OUTPUT_FILE = "temp_recording.wav"

# Global flag for handling Ctrl+C gracefully
running = True

def signal_handler(sig, frame):
    global running
    print("\nStopping recording...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# Initialize Whisper model
model = WhisperModel("base", device="cpu", compute_type="int8")

# Queue for audio chunks
audio_queue = queue.Queue()

def transcribe_audio():
    while running or not audio_queue.empty():
        if not audio_queue.empty():
            frames = audio_queue.get()
            
            # Save temporary WAV file
            wf = wave.open(WAVE_OUTPUT_FILE, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # pyaudio.paInt16 = 2 bytes
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # Transcribe
            segments = model.transcribe(WAVE_OUTPUT_FILE, beam_size=5)
            
            # Print transcription
            for segment in segments:
                try:
                    if hasattr(segment, 'text') and segment.text:
                        print(f"\n[{time.strftime('%H:%M:%S')}] {segment.text}")
                        if segment.text == "enter":
                            pyautogui.press("enter")
                        else:
                            pyautogui.write(segment.text)
                    else:
                        print(f"\n[{time.strftime('%H:%M:%S')}] Warning: Empty transcription segment")
                except AttributeError:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Warning: Invalid transcription segment")
        
        time.sleep(0.1)  # Small delay to prevent CPU overuse

# Start transcription thread
transcribe_thread = threading.Thread(target=transcribe_audio)
transcribe_thread.start()

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* Recording started (Press Ctrl+C to stop)")

try:
    while running:
        frames = []
        # Record for RECORD_SECONDS
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            if not running:
                break
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        # Add recorded chunk to queue
        if frames:
            audio_queue.put(frames)

except Exception as e:
    print(f"Error: {str(e)}")
finally:
    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Wait for transcription thread to finish
    transcribe_thread.join()
    print("\nRecording and transcription stopped.")