"""
Simple Edge Device Client

Streams audio from microphone to server at 16kHz and plays audio from server at 24kHz.
Uses wakeword detection to start/stop conversations.

Usage:
    python -m audio.client.simple_client [--server=ws://localhost:8765]
"""

import os
import asyncio
import websockets
import json
import base64
import numpy as np
import sounddevice as sd
import wave
import io
import time
import argparse
import queue
import struct
import urllib.request
import zipfile
from pvporcupine import create

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
CHANNEL_INDEX = 1  # Use channel 1 from the 6-channel input
CHANNELS = 6  # Total number of input channels

class SimpleAudioClient:
    def __init__(self, server_url="ws://localhost:8765"):
        """Initialize audio client"""
        self.server_url = server_url
        self.websocket = None
        self.running = False
        self.hanging_up = False  # Track when we're in the process of hanging up
        self.sequence = 0  # Counter for message sequencing
        self.in_conversation = False  # Track if we're in an active conversation
        
        # Audio configuration
        self.speaker_sample_rate = 24000  # 24kHz for TTS audio
        
        # For handling chunked audio from server
        self.audio_chunks = {}  # Dictionary to store partial audio chunks
        
        # Initialize wakeword detector
        self.audio_queue = queue.Queue()
        self.stop_event = asyncio.Event()
        
        # Initialize Porcupine - required dependency
        access_key = os.getenv("PICOVOICE_ACCESS_KEY")
        if not access_key:
            raise RuntimeError("Missing Picovoice access key. Set PICOVOICE_ACCESS_KEY environment variable.")
        
        self.porcupine = create(access_key=access_key, keywords=["jarvis"])
        print("Wakeword detection initialized successfully")
        
        # Try to import the signal module for resampling
        self.has_signal = False
        try:
            from scipy import signal
            self.has_signal = True
            print("Audio resampling initialized successfully")
        except ImportError:
            print("⚠️  scipy.signal not available - audio resampling will be limited")
            print("For better audio quality, install scipy: pip install scipy")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice input stream"""
        if status:
            print("Status:", status, flush=True)

        audio_chunk = indata[:, CHANNEL_INDEX]
        audio_chunk = (audio_chunk * 32768).astype(np.int16)  # Convert to int16 PCM for Porcupine
        self.audio_queue.put(audio_chunk.copy())

    async def initialize_audio(self):
        """Initialize audio devices without connecting to server"""
        try:
            # Print device list for diagnostics
            print("🔍 Available audio input devices:")
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0:
                    print(f"  [{i}] {dev['name']} — {dev['max_input_channels']} channels")

            print(f"🎤 Listening on channel {CHANNEL_INDEX} for wakeword...")
            
            # Set up input stream with sounddevice
            self.input_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='float32',
                blocksize=CHUNK_SIZE,
                callback=self.audio_callback
            )
            self.input_stream.start()
            
            print("Audio devices initialized")
            return True
            
        except Exception as e:
            print(f"Audio initialization failed: {e}")
            self.cleanup()
            return False

    async def connect(self):
        """Connect to server and initialize audio devices"""
        try:
            # Connect to WebSocket server
            print(f"Connecting to {self.server_url}...")
            self.websocket = await websockets.connect(self.server_url)
            print("Connected to server")
            
            # Initialize speaker output stream
            print("Initializing speaker...")
            self.output_stream = sd.OutputStream(
                samplerate=self.speaker_sample_rate,
                channels=1,
                dtype='float32',
                blocksize=CHUNK_SIZE
            )
            self.output_stream.start()
            
            print("Audio devices initialized")
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            if self.websocket:
                await self.websocket.close()
            self.cleanup()
            return False

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up audio resources...")
            
        if hasattr(self, 'input_stream'):
            try:
                self.input_stream.stop()
                self.input_stream.close()
            except Exception as e:
                print(f"Error closing input stream: {e}")
            finally:
                self.input_stream = None
            
        if hasattr(self, 'output_stream'):
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except Exception as e:
                print(f"Error closing output stream: {e}")
            finally:
                self.output_stream = None
                
        if self.porcupine:
            try:
                self.porcupine.delete()
            except Exception as e:
                print(f"Error deleting Porcupine: {e}")
            finally:
                self.porcupine = None
        
        print("Audio resources cleaned up")

    async def run(self):
        """Run the client with wakeword detection"""
        if not await self.initialize_audio():
            return
            
        self.running = True
        
        print("\n========== EDGE DEVICE SIMULATOR ==========")
        print("Waiting for wake word 'jarvis'...")
        print("Receiving audio will play through speakers")
        print("\n⚠️  IMPORTANT: Use headphones to prevent echo!")
        print("==========================================")
        print("Press Ctrl+C to exit\n")
        
        try:
            while self.running:
                if not self.in_conversation:
                    # Wait for wake word
                    buffer = np.zeros((0,), dtype=np.int16)
                    
                    while self.running and not self.in_conversation:
                        try:
                            audio_chunk = self.audio_queue.get(timeout=0.5)
                        except queue.Empty:
                            continue

                        buffer = np.concatenate((buffer, audio_chunk))
                        while len(buffer) >= CHUNK_SIZE:
                            frame = buffer[:CHUNK_SIZE]
                            buffer = buffer[CHUNK_SIZE:]
                            result = self.porcupine.process(frame)
                            if result >= 0:
                                print("\n🎯 Wake word detected!")
                                self.in_conversation = True
                                break
                
                if self.in_conversation:
                    # Start a new conversation
                    if await self.connect():
                        print("Starting conversation...")
                        try:
                            send_task = asyncio.create_task(self.send_audio_loop())
                            receive_task = asyncio.create_task(self.receive_audio_loop())
                            
                            # Wait for conversation to end
                            while self.running and self.in_conversation:
                                if send_task.done() or receive_task.done():
                                    break
                                await asyncio.sleep(0.1)
                            
                            # Clean up tasks
                            if not send_task.done():
                                send_task.cancel()
                            if not receive_task.done():
                                receive_task.cancel()
                            
                            try:
                                await asyncio.wait_for(asyncio.gather(send_task, receive_task, return_exceptions=True), timeout=2.0)
                            except asyncio.TimeoutError:
                                print("Some tasks did not terminate gracefully")
                            
                        except Exception as e:
                            print(f"Error in conversation: {e}")
                        finally:
                            await self.disconnect()
                            self.in_conversation = False
                            print("\nWaiting for wake word 'jarvis'...")
                    else:
                        print("Failed to connect to server. Please check if the server is running.")
                        print("Retrying in 5 seconds... (Press Ctrl+C to exit)")
                        try:
                            await asyncio.sleep(5)
                        except asyncio.CancelledError:
                            break
                
        except asyncio.CancelledError:
            print("Tasks cancelled")
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Client shutting down...")
            self.running = False
            self.cleanup()
            print("Client shutdown complete")

    async def send_audio_loop(self):
        """Continuously read from microphone and send to server"""
        try:
            while self.running and self.websocket and self.in_conversation:
                # If we're hanging up, stop sending audio
                if self.hanging_up:
                    print("Hanging up - stopped sending audio")
                    self.in_conversation = False
                    break
                    
                # Check if the websocket is closed
                if hasattr(self.websocket, 'closed') and self.websocket.closed:
                    print("WebSocket connection closed - stopped sending audio")
                    self.in_conversation = False
                    break
                    
                # Read audio data from queue
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.5)
                    
                    # Simple audio level check
                    audio_level = np.abs(audio_chunk).mean()
                    
                    # Only warn if audio level is very low and not too frequently
                    if audio_level < 10 and self.sequence % 100 == 0:
                        print(f"Warning: Very low audio level ({audio_level:.1f}). Check your microphone.")
                    
                    # Create audio message
                    message = {
                        "type": "audio_stream",
                        "timestamp": time.time(),
                        "sequence": self.sequence,
                        "payload": {
                            "audio": base64.b64encode(audio_chunk.tobytes()).decode('utf-8')
                        }
                    }
                    self.sequence += 1
                    
                    # Send to server - wrap in try/except to handle closed connection
                    try:
                        await self.websocket.send(json.dumps(message))
                    except websockets.exceptions.ConnectionClosed:
                        print("Connection closed while sending audio - stopping")
                        self.in_conversation = False
                        break
                    
                    # Pace ourselves according to the chunk duration
                    chunk_duration = CHUNK_SIZE / SAMPLE_RATE
                    await asyncio.sleep(chunk_duration * 0.5)  # Sleep for half the chunk duration
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error capturing audio: {e}")
                    await asyncio.sleep(0.1)
                    
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed - audio sending stopped")
            self.in_conversation = False
        except Exception as e:
            print(f"Error in send loop: {e}")
            self.in_conversation = False

    def process_and_play_audio(self, audio_data):
        """Process audio data and play it"""
        try:
            # Check if it's a WAV file (has RIFF header)
            if audio_data[:4] == b'RIFF':
                # Parse WAV data
                with io.BytesIO(audio_data) as wav_buffer:
                    with wave.open(wav_buffer, 'rb') as wf:
                        # Get WAV properties
                        wav_rate = wf.getframerate()
                        wav_channels = wf.getnchannels()
                        
                        # Read audio data
                        pcm_data = wf.readframes(wf.getnframes())
                        
                        # Convert to numpy array
                        audio_np = np.frombuffer(pcm_data, dtype=np.int16)
                        
                        # Convert to mono if needed
                        if wav_channels > 1:
                            audio_np = audio_np.reshape(-1, wav_channels)
                            audio_np = np.mean(audio_np, axis=1, dtype=np.int16)
                        
                        # Resample if needed
                        if wav_rate != self.speaker_sample_rate:
                            audio_np = self.resample_audio(audio_np, wav_rate, self.speaker_sample_rate)
                        
                        # Convert to float32 for sounddevice
                        audio_np = audio_np.astype(np.float32) / 32768.0
            else:
                # Assume it's raw PCM at the speaker sample rate
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                audio_np = audio_np.astype(np.float32) / 32768.0
            
            # Play the audio directly
            if hasattr(self, 'output_stream'):
                print(f"▶️ Playing audio ({len(audio_np)} samples)")
                self.output_stream.write(audio_np)
                print("✅ Audio playback complete")
                
                # If we're hanging up and this was the final audio chunk, end conversation
                if self.hanging_up:
                    print("📞 Final hangup audio played, ending conversation...")
                    self.in_conversation = False
            else:
                print("Speaker stream unavailable - cannot play audio")
            
        except Exception as audio_err:
            print(f"Error processing audio: {audio_err}")
            import traceback
            traceback.print_exc()

    def resample_audio(self, audio_np, source_rate, target_rate):
        """Resample audio data using the best available method"""
        try:
            if self.has_signal:
                # Use scipy's high-quality resampling
                from scipy import signal
                resampled = signal.resample_poly(
                    audio_np,
                    target_rate,
                    source_rate
                )
                return np.int16(resampled)
            else:
                # Simple linear resampling if scipy is not available
                ratio = target_rate / source_rate
                if ratio > 1:
                    # Upsample by linear interpolation
                    indices = np.arange(0, len(audio_np) * ratio) / ratio
                    resampled = np.interp(indices, np.arange(len(audio_np)), audio_np)
                else:
                    # Downsample by averaging
                    indices = np.floor(np.arange(0, len(audio_np) * ratio) / ratio).astype(int)
                    resampled = audio_np[indices]
                return np.int16(resampled)
        except Exception as e:
            print(f"Error during resampling: {e}")
            # Return original audio if resampling fails
            return audio_np

    async def disconnect(self):
        """Disconnect from server and clean up"""
        print("Disconnecting from server...")
        
        try:
            if self.websocket and not (hasattr(self.websocket, 'closed') and self.websocket.closed):
                # Send disconnect message
                try:
                    message = {
                        "type": "disconnect",
                        "timestamp": time.time(),
                        "sequence": self.sequence,
                        "payload": {
                            "reason": "Client disconnecting"
                        }
                    }
                    self.sequence += 1
                    await self.websocket.send(json.dumps(message))
                    print("Sent disconnect message")
                except Exception as e:
                    print(f"Error sending disconnect message: {e}")
                
                # Close the websocket
                try:
                    await self.websocket.close()
                except Exception as e:
                    print(f"Error closing websocket: {e}")
        except Exception as e:
            print(f"Error during disconnect: {e}")
        finally:
            self.websocket = None
            
            # Close the speaker stream
            if hasattr(self, 'output_stream'):
                try:
                    self.output_stream.stop()
                    self.output_stream.close()
                except Exception as e:
                    print(f"Error closing speaker stream: {e}")
                finally:
                    self.output_stream = None
            
        print("Disconnected from server")

async def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simple Audio Edge Client")
    parser.add_argument("--server", default="ws://localhost:8765", help="WebSocket server URL")
    args = parser.parse_args()
    
    # Create and run client
    client = SimpleAudioClient(server_url=args.server)
    
    try:
        await client.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Make sure we clean up
        if client.websocket:
            await client.disconnect()
        client.cleanup()
    
if __name__ == "__main__":
    asyncio.run(main()) 