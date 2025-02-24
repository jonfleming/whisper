import pyaudio
import wave
import sys

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16  # Changed from paFloat32 for better compatibility
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
OUTPUT_FILENAME = "recorded_audio.wav"

try:
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # List all available audio devices
    print("\nAvailable audio input devices:")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:  # if it has input channels
            print(f"Index {i}: {dev_info['name']}")
    
    # Get the default input device index
    default_input = p.get_default_input_device_info()
    print(f"\nDefault input device: {default_input['name']}")
    
    # Open stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=default_input['index'])

    print("\n* Recording...")

    # Record audio
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    print("* Done recording")

    # Stop and close the stream
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

except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1)