import pyaudio
import wave
import time
import webrtcvad
from faster_whisper import WhisperModel
import numpy as np

# Audio settings
SAMPLE_RATE = 16000  # WebRTC VAD requires 8kHz, 16kHz, 32kHz, or 48kHz
CHUNK_DURATION_MS = 30  # VAD works with 10ms, 20ms, or 30ms chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # Frames per chunk
MIN_SPEECH_DURATION = 0.5  # Minimum speech duration to process (in seconds)

def is_speech(frame, vad, sample_rate=SAMPLE_RATE):
    """Check if the audio frame contains speech using WebRTC VAD."""
    try:
        return vad.is_speech(frame, sample_rate)
    except Exception:
        return False

def record_chunk_with_vad(p, stream, vad, filename, max_duration=5, sample_rate=SAMPLE_RATE):
    """Record audio only when speech is detected, up to max_duration."""
    frames = []
    speech_detected = False
    speech_start_time = None
    total_duration = 0

    print("Listening for speech...")

    while total_duration < max_duration:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        total_duration += CHUNK_DURATION_MS / 1000  # Increment time

        if is_speech(data, vad):
            if not speech_detected:
                speech_detected = True
                speech_start_time = time.time()
                print("Speech detected!")
            frames.append(data)
        else:
            if speech_detected and (time.time() - speech_start_time) >= MIN_SPEECH_DURATION:
                # Stop recording once speech ends and minimum duration is met
                break
            elif speech_detected:
                frames.append(data)  # Keep recording briefly after speech ends

    if not frames:
        print("No speech detected in this chunk.")
        return False

    # Save the recorded audio
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    return True

def main():
    # Initialize FasterWhisper model
    model = WhisperModel("small", device="cpu", compute_type="int8")

    # Initialize WebRTC VAD (aggressiveness level: 0-3, 3 being most aggressive)
    vad = webrtcvad.Vad(1)  # Adjust aggressiveness as needed (1 is moderate)

    # Set up PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    print("Starting real-time transcription with VAD. Speak into your microphone...")

    try:
        while True:
            audio_file = "temp_chunk.wav"
            # Record a chunk with VAD, max 5 seconds
            if record_chunk_with_vad(p, stream, vad, audio_file, max_duration=5):
                # Transcribe the chunk if speech was detected
                segments, info = model.transcribe(audio_file, beam_size=5)
                for segment in segments:
                    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
            else:
                print("Skipping silent chunk.")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    main()