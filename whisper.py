import pyaudio
import pyautogui
import wave
import time
import webrtcvad
from faster_whisper import WhisperModel
import numpy as np
from queue import Queue
from threading import Thread, Event
import struct

# Audio settings
SAMPLE_RATE = 16000  # WebRTC VAD requires 8kHz, 16kHz, 32kHz, or 48kHz
CHUNK_DURATION_MS = 20  # VAD works with 10ms, 20ms, or 30ms chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # Frames per chunk
MIN_SPEECH_DURATION = 0.5  # Minimum speech duration to process (in seconds)
BUFFER_MAX_SIZE = 100  # Maximum number of chunks to keep in buffer

# Initialize FasterWhisper model
model = WhisperModel("small.en", device="cpu", compute_type="int8")

class AudioBuffer:
    def __init__(self):
        self.buffer = Queue(maxsize=BUFFER_MAX_SIZE)
        self.stop_recording = Event()
        self.is_recording = False

    def callback(self, in_data, frame_count, time_info, status):
        if not self.stop_recording.is_set():
            try:
                self.buffer.put(in_data)
            except Exception:  # Queue.Full exception
                # If buffer is full, remove oldest chunk
                try:
                    self.buffer.get_nowait()
                    self.buffer.put(in_data)
                except Exception:  # Queue.Empty exception
                    pass
        return (None, pyaudio.paContinue)

    def get_chunk(self):
        try:
            return self.buffer.get_nowait()
        except Exception:  # Queue.Empty exception
            return None

def is_speech(frame, vad, sample_rate=SAMPLE_RATE):
    """Check if the audio frame contains speech using WebRTC VAD."""
    try:
        return vad.is_speech(frame, sample_rate)
    except Exception:
        return False

def process_audio_buffer(audio_buffer, vad, filename, max_duration=5, sample_rate=SAMPLE_RATE):
    """Process audio from buffer when speech is detected."""
    frames, speech_detected, speech_start_time, total_duration, silence_duration = initialize_processing_vars()

    while total_duration < max_duration:
        chunk = audio_buffer.get_chunk()
        if chunk is None:
            time.sleep(0.01)  # Small sleep to prevent busy waiting
            continue

        total_duration += CHUNK_DURATION_MS / 1000

        if is_speech(chunk, vad):
            speech_detected, speech_start_time, silence_duration = handle_speech_detected(
                speech_detected, speech_start_time, silence_duration, chunk, frames)
        else:
            silence_duration = handle_silence_detected(
                speech_detected, silence_duration, chunk, frames, speech_start_time)
            if silence_duration is None:
                break

    if not frames:
        print(".", end="")
        return False

    save_recorded_audio(filename, frames, sample_rate)
    return True

def initialize_processing_vars():
    frames = []
    speech_detected = False
    speech_start_time = None
    total_duration = 0
    silence_duration = 0
    return frames, speech_detected, speech_start_time, total_duration, silence_duration

def handle_speech_detected(speech_detected, speech_start_time, silence_duration, chunk, frames):
    if not speech_detected:
        speech_detected = True
        speech_start_time = time.time()
    silence_duration = 0
    frames.append(chunk)
    return speech_detected, speech_start_time, silence_duration

def handle_silence_detected(speech_detected, silence_duration, chunk, frames, speech_start_time):
    MAX_SILENCE_DURATION = 0.5  # Maximum silence duration before stopping
    if speech_detected:
        silence_duration += CHUNK_DURATION_MS / 1000
        frames.append(chunk)
        if silence_duration >= MAX_SILENCE_DURATION:
            if (time.time() - speech_start_time) >= MIN_SPEECH_DURATION:
                return None
    return silence_duration

def save_recorded_audio(filename, frames, sample_rate):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)  # pyaudio.paInt16 = 2 bytes
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def main():
    # Initialize WebRTC VAD
    vad = webrtcvad.Vad(3)

    # Set up PyAudio with buffered recording
    audio_buffer = AudioBuffer()
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                   channels=1,
                   rate=SAMPLE_RATE,
                   input=True,
                   frames_per_buffer=CHUNK_SIZE,
                   stream_callback=audio_buffer.callback)

    stream.start_stream()
    print("Starting real-time transcription with VAD. Speak into your microphone...")

    try:
        while True:
            audio_file = "temp_chunk.wav"
            if process_audio_buffer(audio_buffer, vad, audio_file, max_duration=5):
                # Transcribe the chunk if speech was detected
                segments, info = model.transcribe(audio_file, beam_size=1)
                for segment in segments:
                    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
                    if (segment.text == "enter."):
                        pyautogui.press("enter")
                    else:
                        pyautogui.write(segment.text)

    except KeyboardInterrupt:
        print("\nStopped by user.")
        audio_buffer.stop_recording.set()

    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    main()