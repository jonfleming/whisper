# Transcribes audio in real-time using WebRTC VAD and FasterWhisper
# This script uses PyAudio for real-time audio capture and FasterWhisper for transcription.
# It also uses pyautogui to simulate keyboard input for transcription output.
# The script is designed to run in a loop, continuously capturing audio and transcribing it until interrupted.
# It uses a lock file to prevent multiple instances from running simultaneously.
# The script requires the following libraries:
import os
import pyaudio
import pyautogui
import wave
import time
import webrtcvad
from faster_whisper import WhisperModel
import numpy as np
from queue import Queue
import sys
from threading import Thread, Event
import struct
import keyboard
import argparse
import math

# Audio settings
AWAKE_TIME = 250
BUFFER_MAX_SIZE = 100  # Maximum number of chunks to keep in buffer
CHUNK_DURATION_MS = 10  # VAD works with 10ms, 20ms, or 30ms chunks
SAMPLE_RATE = 16000  # WebRTC VAD requires 8kHz, 16kHz, 32kHz, or 48kHz
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # Frames per chunk
LOCK_FILE = "whisper.lock"
MIN_SPEECH_DURATION = 2.0  # Minimum speech duration to process (in seconds)
OUTPUT_FILE = "transcription_output.txt"
WAKE_WORD = "Teresa"  # Default wake word

DEBUG = True
SEND_KEYS = True

awake = False
wake_word_enabled = True  # New variable to control wake word functionality
sleep_countdown = 0
prompt = ">"
output = ""
class AudioBuffer:
    def __init__(self):
        self.buffer = Queue(maxsize=BUFFER_MAX_SIZE)
        self.stop_recording = Event()
        self.is_recording = False

# filepath: c:\Projects\whisper\whisper.py
    def callback(self, in_data, frame_count, time_info, status):
        if not self.stop_recording.is_set():
            try:
                # Add overlapping frames
                if not self.buffer.empty():
                    last_chunk = self.buffer.queue[-1]
                    in_data = last_chunk[-CHUNK_SIZE:] + in_data
                self.buffer.put(in_data)
                self.display_audio_meter(in_data)  # Add this line to visualize audio
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

    def display_audio_meter(self, in_data):
        """Display an audio meter based on the amplitude of the audio chunk."""
        audio_np = np.frombuffer(in_data, dtype=np.int16)
        rms = math.sqrt(np.mean(audio_np**2))  # Calculate root mean square (RMS) amplitude
        meter = "#" * int(rms / 2)  # Scale the amplitude to create a visual meter
        padding = " " * (50 - len(meter))  # Pad to fixed width
        meter = f"[{meter}{padding}] {rms:.2f}"
        print(f"Audio Meter: {meter}", end="\r")  # Print the meter in the terminal

def is_speech(frame, vad, sample_rate=SAMPLE_RATE):
    """Check if the audio frame contains speech using WebRTC VAD."""
    try:
        return vad.is_speech(frame, sample_rate)
    except Exception:
        return False

def process_audio_buffer(audio_buffer, vad, model, max_duration=3):
    global awake, sleep_countdown
    """Process audio from buffer and transcribe in real-time without saving to file."""
    frames, speech_detected, speech_start_time, total_duration, silence_duration = initialize_processing_vars()
    
    # Track speech frames count to ensure we have actual content
    speech_frames_count = 0
    total_frames_count = 0
    
    # Continue with normal processing
    while total_duration < max_duration:
        chunk = audio_buffer.get_chunk()
        if chunk is None:
            time.sleep(0.01)  # Small sleep to prevent busy waiting
            continue

        total_duration += CHUNK_DURATION_MS / 1000
        total_frames_count += 1
        
        if is_speech(chunk, vad):            
            speech_detected, speech_start_time, silence_duration = handle_speech_detected(
                speech_detected, speech_start_time, silence_duration, chunk, frames)
            speech_frames_count += 1
            if awake:
                sleep_countdown = AWAKE_TIME # Reset awake time when speech is detected
        else:
            silence_duration = handle_silence_detected(
                speech_detected, silence_duration, chunk, frames, speech_start_time)
            if awake:
                sleep_countdown -= 1
                if sleep_countdown <= 0:
                    clear_prompt()
                    awake = False
    
    # Calculate speech ratio to filter out ambient noise
    speech_ratio = speech_frames_count / total_frames_count if total_frames_count > 0 else 0
    
    # Only process if we have frames AND detected enough speech frames
    # Increased threshold from 5 to 10 frames minimum with speech
    if not frames or speech_frames_count < 10:  
        return False

    # Only process if we've detected actual speech (not just noise)
    # Ensure at least 20% of frames contain speech
    if not speech_detected or speech_ratio < 0.2:
        return False

    transcribe_audio(frames, model)
    return True

def transcribe_audio(frames, model):
    """Convert audio frames to text using the Whisper model."""
    global awake, sleep_countdown, prompt, output
    audio_data = b"".join(frames)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    # Add VAD filtering and set minimum confidence threshold
    segments, _ = model.transcribe(audio_np, beam_size=5, vad_filter=True, temperature=0.0)
    
    for segment in segments:
        # Skip segments with low confidence
        if hasattr(segment, 'avg_logprob') and segment.avg_logprob < -0.7:
            continue
            
        final_text = segment.text.lower().replace(".", "")
        
        # Skip very short segments that are likely noise
        if len(final_text.strip()) < 2:
            continue
            
        line = f"[{segment.start:.2f}s - {segment.end:.2f}s] {final_text}\n"
        write_to_file(line)

        if not awake and wake_word_enabled:
            if WAKE_WORD.lower() in final_text.strip().lower():
                awake = True
                sleep_countdown = AWAKE_TIME
                show_prompt()
        else:
            clear_prompt()

            if len(final_text) < 8 and "enter" in final_text.strip().lower():
                pyautogui.press("enter") if SEND_KEYS else None
                output = output + "\n"
                show_prompt()
            elif len(final_text) < 8 and "period" in final_text.strip().lower():
                pyautogui.write(". ") if SEND_KEYS else None
                output = output + ". "
                show_prompt()
            elif len(final_text) < 8 and "delete" in final_text.strip().lower():
                debug(":" + output + ":")
                last_word = get_last_word(output)
                output = output[:-len(last_word)].rstrip()
                debug("[" + last_word + "]")
                for _ in range(len(last_word) + 1):  # +1 to remove the trailing space
                    pyautogui.press("backspace") if SEND_KEYS else None
                debug(">" + output + "<")
                show_prompt()

            else:
                pyautogui.write(final_text) if SEND_KEYS else None
                output = output + final_text
                show_prompt()

def show_prompt():
    global prompt
    prompt= ">"
    pyautogui.write(prompt) if SEND_KEYS else None

def clear_prompt():
    global prompt
    if prompt == ">":
        prompt = ""
        pyautogui.press("backspace") if SEND_KEYS else None

def get_last_word(text):
    words = text.split()
    return words[-1] if words else text

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
    if silence_duration > 0:
        silence_duration = 0
    frames.append(chunk)
    return speech_detected, speech_start_time, silence_duration

def handle_silence_detected(speech_detected, silence_duration, chunk, frames, speech_start_time):
    MAX_SILENCE_DURATION = 1.5  # Increased maximum silence duration before stopping
    if speech_detected:
        silence_duration += CHUNK_DURATION_MS / 1000
        frames.append(chunk)
        if silence_duration >= MAX_SILENCE_DURATION:
            if (time.time() - speech_start_time) >= MIN_SPEECH_DURATION:
                return 0
    return silence_duration

def save_recorded_audio(filename, frames, sample_rate):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)  # pyaudio.paInt16 = 2 bytes
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def write_to_file(msg):
    """Write a message to the output file."""
    print(msg, end="", flush=True)
    with open(OUTPUT_FILE, "a") as f:
        f.write(msg)

def debug(msg):
    if DEBUG:
        write_to_file(msg + "\n")  # Write debug messages to the file
        # keyboard.read_event(msg)

def main():
    parser = argparse.ArgumentParser(description="Real-time audio transcription with VAD and FasterWhisper.")
    parser.add_argument("-f", "--force", action="store_true", help="Force bypass the lock file test.")
    parser.add_argument("-w", "--wake-word", type=str, help="Set custom wake word (use 'none' to disable)")
    args = parser.parse_args()

    if os.path.exists(LOCK_FILE) and not args.force:
        print("Another instance is already running.\n")
        sys.exit(0)

    # Handle wake word settings
    global WAKE_WORD, wake_word_enabled, awake
    if args.wake_word:
        if args.wake_word.lower() == 'none':
            wake_word_enabled = False
            awake = True  # Start in awake state when wake word is disabled
        else:
            WAKE_WORD = args.wake_word
            wake_word_enabled = True

    # Create the lock file
    open(LOCK_FILE, "w").close()

    # Initialize FasterWhisper model
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    # Initialize WebRTC VAD with higher aggressiveness (0-3, 3 being most aggressive)
    vad = webrtcvad.Vad(0)

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
    print(f"Starting real-time transcription with VAD. " + 
          (f"Say '{WAKE_WORD}' to start sending output to the current window." if wake_word_enabled 
           else "Wake word is disabled, immediately sending output to current window."))

    try:
        while True:
            process_audio_buffer(audio_buffer, vad, model, max_duration=3)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Ensure resources are cleaned up properly
        debug("stop_recording")
        audio_buffer.stop_recording.set()
        debug("stop_stream")
        stream.stop_stream()
        debug("stream.close")
        stream.close()
        debug("p.terminate")
        p.terminate()
        if os.path.exists(LOCK_FILE):
            debug("os.remove")
            os.remove(LOCK_FILE)
        sys.exit(0)  # Ensure a clean exit

if __name__ == "__main__":
    main()