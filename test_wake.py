"""
Live microphone test script for wakeword detection using OpenWakeWord.
Runs inference in a background thread to avoid audio callback delays.
"""

import numpy as np
import openwakeword
import sounddevice as sd
import threading
import queue
import time

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
BUFFER_SIZE = 8192
CHANNEL_INDEX = 0
CHANNELS = 6

oww = openwakeword.Model(wakeword_models=["hey_mycroft"], inference_framework="tflite")
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
audio_queue = queue.Queue()
stop_event = threading.Event()

# Print device list for diagnostics
print("🔍 Available audio input devices:")
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(f"  [{i}] {dev['name']} — {dev['max_input_channels']} channels")

print(f"\n🎤 Listening on channel {CHANNEL_INDEX} for wakeword...")

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Status:", status, flush=True)

    audio_chunk = indata[:, CHANNEL_INDEX]
    if len(audio_chunk) < CHUNK_SIZE:
        audio_chunk = np.pad(audio_chunk, (0, CHUNK_SIZE - len(audio_chunk)), mode='constant')

    audio_queue.put(audio_chunk.copy())

def inference_loop():
    global audio_buffer
    while not stop_event.is_set():
        try:
            audio_chunk = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
        audio_buffer[-CHUNK_SIZE:] = audio_chunk

        prediction = oww.predict(audio_buffer)
        if isinstance(prediction, dict):
            score = prediction.get("hey_mycroft", 0)
            print(f"hey_mycroft score: {score:.3f}", end="\r")
            if score > 0.1:
                print(f"\n\n🎯 WAKE WORD DETECTED! Confidence: {score:.2f}\n")
                stop_event.set()
                break

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
