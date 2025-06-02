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
import subprocess
from pvporcupine import create
from led_control import LEDControl

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
        self.original_volume = None  # Store original volume level
        
        # Initialize LED control
        self.led_control = LEDControl()
        
        # Audio configuration
        self.speaker_sample_rate = 24000  # 24kHz for TTS audio
        self.audio_block_size = 1024  # Increased block size for more stable playback
        
        # For handling chunked audio from server
        self.audio_chunks = {}  # Dictionary to store partial audio chunks
        
        # Initialize wakeword detector
        self.audio_queue = queue.Queue()
        self.stop_event = asyncio.Event()
        
        # Initialize Porcupine - required dependency
        access_key = os.getenv("PICOVOICE_ACCESS_KEY")
        if not access_key:
            raise RuntimeError("Missing Picovoice access key. Set PICOVOICE_ACCESS_KEY environment variable.")
        
        self.porcupine = create(access_key=access_key, keywords=["porcupine"])
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

    def get_system_volume(self):
        """Get current system volume as a percentage"""
        try:
            # Get volume of main_out sink
            result = subprocess.run(['pactl', 'list', 'sinks'], 
                                 capture_output=True, text=True)
            
            # Find the main_out sink section
            sink_sections = result.stdout.split('\n\n')
            main_out_section = None
            for section in sink_sections:
                if 'Sink #4' in section or 'main_out' in section:
                    main_out_section = section
                    break
            
            if main_out_section:
                # Extract volume percentage
                for line in main_out_section.split('\n'):
                    if 'Volume:' in line:
                        # Parse the volume percentage from the line
                        # Format is typically: Volume: front-left: 65536 / 100% / 0.00 dB
                        volume_str = line.split('/')[1].strip()
                        volume_percent = int(volume_str.replace('%', ''))
                        return volume_percent
            
            print("Could not find main_out sink volume")
            return None
        except Exception as e:
            print(f"Error getting system volume: {e}")
            return None

    def set_system_volume(self, volume_percent):
        """Set system volume to specified percentage (0-100)"""
        try:
            # Set volume of main_out sink
            subprocess.run(['pactl', 'set-sink-volume', 'main_out', f'{volume_percent}%'])
            return True
        except Exception as e:
            print(f"Error setting system volume: {e}")
            return False

    async def connect(self):
        """Connect to server and initialize audio devices"""
        try:
            # Connect to WebSocket server
            print(f"Connecting to {self.server_url}...")
            self.websocket = await websockets.connect(self.server_url)
            print("Connected to server")
            
            # Store original volume and set to 55%
            self.original_volume = self.get_system_volume()
            if self.original_volume is not None:
                print(f"Setting volume to 75% (was {self.original_volume}%)")
                self.set_system_volume(75)
            
            # Initialize speaker output stream
            print("Initializing speaker...")
            self.output_stream = sd.OutputStream(
                samplerate=self.speaker_sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.audio_block_size,
                latency='low'  # Request low latency
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
        
        # Clean up LED control
        if hasattr(self, 'led_control'):
            self.led_control.cleanup()
        
        print("Audio resources cleaned up")

    async def run(self):
        """Run the client with wakeword detection"""
        if not await self.initialize_audio():
            return
            
        self.running = True
        
        print("\n========== EDGE DEVICE ==========")
        print("Waiting for wake word 'porcupine'...")
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
                                # Store the remaining buffer for sending
                                self.wakeword_audio = buffer.copy()
                                self.in_conversation = True
                                self.hanging_up = False  # Reset hanging up state
                                # Trigger LED wake state
                                self.led_control.wake_detected()
                                break
                
                if self.in_conversation:
                    # Start a new conversation
                    if await self.connect():
                        print("Starting conversation...")
                        try:
                            # Send the buffered audio first
                            if hasattr(self, 'wakeword_audio') and len(self.wakeword_audio) > 0:
                                message = {
                                    "type": "audio_stream",
                                    "timestamp": time.time(),
                                    "sequence": self.sequence,
                                    "payload": {
                                        "audio": base64.b64encode(self.wakeword_audio.tobytes()).decode('utf-8')
                                    }
                                }
                                self.sequence += 1
                                await self.websocket.send(json.dumps(message))
                                print("Sent buffered audio after wakeword")
                                delattr(self, 'wakeword_audio')  # Clear the buffer
                            
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
                            # Turn off LEDs when conversation ends
                            self.led_control.conversation_ended()
                            print("\nWaiting for wake word 'porcupine'...")
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

    async def receive_audio_loop(self):
        """Receive audio and other messages from server"""
        try:
            while self.running and self.websocket and self.in_conversation:
                # Check if the websocket is closed
                if hasattr(self.websocket, 'closed') and self.websocket.closed:
                    print("WebSocket connection closed - stopped receiving")
                    self.in_conversation = False
                    break
                
                # Wait for a message from the server with a timeout
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    # No message received in the timeout period, check if we're still running and try again
                    if not self.running or not self.in_conversation or (hasattr(self.websocket, 'closed') and self.websocket.closed):
                        break
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed while waiting for messages")
                    self.in_conversation = False
                    break
                
                try:
                    # Parse the message
                    msg = json.loads(message)
                    
                    # Handle audio playback messages
                    if msg["type"] == "audio_playback":
                        # Check if this message has the hangup flag
                        is_hangup = msg["payload"].get("is_hangup", False)
                        is_final = msg["payload"].get("is_final", False)
                        
                        # If this is a hangup message, mark it for proper handling
                        if is_hangup:
                            print("📞 Received audio with hangup flag")
                            self.hanging_up = True
                            
                            # If this is a final empty message with hangup flag, end conversation
                            if is_final and "audio" not in msg["payload"]:
                                print("📞 Received final hangup signal without audio, ending conversation...")
                                self.in_conversation = False
                                break
                        
                        # Check if there's audio data to play
                        if "audio" in msg["payload"]:
                            # Decode audio data from base64
                            audio_data = base64.b64decode(msg["payload"]["audio"])
                            
                            # Play it directly - no queuing
                            if audio_data and len(audio_data) > 0:
                                print(f"Playing audio: hangup={is_hangup}, final={is_final}, size={len(audio_data)}bytes")
                                self.process_and_play_audio(audio_data)
                                
                                # Check if this was the final audio chunk in a hangup sequence
                                if is_hangup and is_final:
                                    print("📞 Received final hangup audio, ending conversation...")
                                    self.in_conversation = False
                            else:
                                print("Received empty audio data")
                    
                    # Handle status messages (transcription, state updates)
                    elif msg["type"] == "status":
                        # Check if we're hanging up
                        if msg["payload"].get("state") == "hanging_up":
                            print("Server is hanging up")
                            self.hanging_up = True
                            
                        # Handle speech detection
                        if "is_speech" in msg["payload"]:
                            is_speech = msg["payload"]["is_speech"]
                            if is_speech:
                                print("🎤 Speech detected...")
                                
                        # Handle state transitions
                        if "state" in msg["payload"]:
                            state = msg["payload"]["state"]
                            print(f"🔄 State: {state}")
                            
                        # Handle transcription
                        if "transcription" in msg["payload"]:
                            text = msg["payload"]["transcription"]["text"]
                            print(f"🔊 You said: \"{text}\"")
                    
                    # Handle error messages
                    elif msg["type"] == "error":
                        error = msg["payload"].get("error", "Unknown error")
                        print(f"❌ Error: {error}")
                
                except json.JSONDecodeError:
                    print(f"Invalid JSON message")
                except KeyError as e:
                    print(f"Missing key in message: {e}")
                except Exception as e:
                    print(f"Error processing message: {e}")
                    import traceback
                    traceback.print_exc()
                
        except Exception as e:
            print(f"Error in receive loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
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
                try:
                    self.output_stream.write(audio_np)
                    print("✅ Audio playback complete")
                except Exception as e:
                    print(f"Warning: Audio playback error (non-critical): {e}")
                
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
            
            # Restore original volume
            if self.original_volume is not None:
                print(f"Restoring volume to {self.original_volume}%")
                self.set_system_volume(self.original_volume)
                self.original_volume = None
            
            # Reset conversation state
            self.hanging_up = False
            self.in_conversation = False
            
            # Clear the audio queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
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
