"""
Live microphone test script for wakeword detection using Picovoice Porcupine.
Runs inference in a background thread to avoid audio callback delays.
"""

import numpy as np
import sounddevice as sd
import threading
import queue
import time
from pvporcupine import create, KEYWORD_PATHS

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
CHANNEL_INDEX = 1
CHANNELS = 6

import os

access_key = os.getenv("PICOVOICE_ACCESS_KEY")
if not access_key:
    raise RuntimeError("Missing Picovoice access key. Set PICOVOICE_ACCESS_KEY environment variable.")

porcupine = create(access_key=access_key, keywords=["jarvis"])  # Replace with your preferred wakeword
BUFFER_SIZE = CHUNK_SIZE  # Porcupine needs fixed size frames

audio_queue = queue.Queue()
stop_event = threading.Event()

# Print device list for diagnostics
print("🔍 Available audio input devices:")
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(f"  [{i}] {dev['name']} — {dev['max_input_channels']} channels")

print(f"🎤 Listening on channel {CHANNEL_INDEX} for wakeword...")

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Status:", status, flush=True)

    audio_chunk = indata[:, CHANNEL_INDEX]
    audio_chunk = (audio_chunk * 32768).astype(np.int16)  # Convert to int16 PCM for Porcupine
    audio_queue.put(audio_chunk.copy())

def inference_loop():
    buffer = np.zeros((0,), dtype=np.int16)

    while not stop_event.is_set():
        try:
            audio_chunk = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        buffer = np.concatenate((buffer, audio_chunk))
        while len(buffer) >= BUFFER_SIZE:
            frame = buffer[:BUFFER_SIZE]
            buffer = buffer[BUFFER_SIZE:]
            result = porcupine.process(frame)
            if result >= 0:
                print(f"🎯 WAKE WORD DETECTED! Index: {result}")

try:
    infer_thread = threading.Thread(target=inference_loop)
    infer_thread.start()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', blocksize=CHUNK_SIZE, callback=audio_callback):
        while not stop_event.is_set():
            time.sleep(0.1)

    infer_thread.join()
except KeyboardInterrupt:
    print("\nStopped by user.")
    stop_event.set()
except Exception as e:
    print(f"\nError: {e}")
    stop_event.set()
finally:
    porcupine.delete()
