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
import pyaudio
import wave
import io
import time
import argparse
import queue
import struct
import urllib.request
import zipfile
import openwakeword

class SimpleAudioClient:
    def __init__(self, server_url="ws://localhost:8765"):
        """Initialize audio client"""
        self.server_url = server_url
        self.websocket = None
        self.pyaudio = None
        self.mic_stream = None
        self.speaker_stream = None
        self.running = False
        self.hanging_up = False  # Track when we're in the process of hanging up
        self.sequence = 0  # Counter for message sequencing
        self.in_conversation = False  # Track if we're in an active conversation
        
        # Audio configuration
        self.channels = 1  # Mono
        self.mic_sample_rate = 16000  # 16kHz for whisper
        self.speaker_sample_rate = 24000  # 24kHz for TTS audio
        self.chunk_size = 1024  # Samples per chunk
        
        # For handling chunked audio from server
        self.audio_chunks = {}  # Dictionary to store partial audio chunks
        
        # Initialize wakeword detector
        self.has_wakeword = False
        self.oww = None
        try:
            self.oww = openwakeword.Model(
                wakeword_models=["hey_jarvis"]
            )
            self.has_wakeword = True
            print("Wakeword detection initialized successfully")
        except ImportError:
            print("openwakeword not available - wakeword detection disabled")
        except Exception as e:
            print(f"Failed to initialize wakeword detection: {e}")
        
        # Try to import the signal module for resampling
        self.has_signal = False
        try:
            from scipy import signal
            self.has_signal = True
            print("Audio resampling initialized successfully")
        except ImportError:
            print("⚠️  scipy.signal not available - audio resampling will be limited")
            print("For better audio quality, install scipy: pip install scipy")
        
    async def initialize_audio(self):
        """Initialize audio devices without connecting to server"""
        try:
            # Initialize PyAudio
            self.pyaudio = pyaudio.PyAudio()
            
            # Set up microphone input stream
            print("Initializing microphone...")
            self.mic_stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.mic_sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Set up speaker output stream
            print("Initializing speaker...")
            self.speaker_stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.speaker_sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )
            
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
            
            # Initialize PyAudio
            self.pyaudio = pyaudio.PyAudio()
            
            # Set up microphone input stream
            print("Initializing microphone...")
            self.mic_stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.mic_sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Set up speaker output stream
            print("Initializing speaker...")
            self.speaker_stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.speaker_sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )
            
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
            
        if self.mic_stream:
            try:
                self.mic_stream.stop_stream()
                self.mic_stream.close()
            except Exception as e:
                print(f"Error closing mic stream: {e}")
            finally:
                self.mic_stream = None
            
        if self.speaker_stream:
            try:
                self.speaker_stream.stop_stream()
                self.speaker_stream.close()
            except Exception as e:
                print(f"Error closing speaker stream: {e}")
            finally:
                self.speaker_stream = None
            
        if self.pyaudio:
            try:
                self.pyaudio.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
            finally:
                self.pyaudio = None
        
        print("Audio resources cleaned up")
    
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
            
        print("Disconnected from server")
            
    async def run(self):
        """Run the client with wakeword detection"""
        if not await self.initialize_audio():
            return
            
        self.running = True
        
        print("\n========== EDGE DEVICE SIMULATOR ==========")
        if self.has_wakeword:
            print("Waiting for wake word 'hey_jarvis'...")
        else:
            print("Wakeword detection not available - press Enter to start conversation")
        print("Receiving audio will play through speakers")
        print("\n⚠️  IMPORTANT: Use headphones to prevent echo!")
        print("==========================================")
        print("Press Ctrl+C to exit\n")
        
        try:
            while self.running:
                if not self.in_conversation:
                    # Wait for wake word or user input
                    if self.has_wakeword:
                        await self.wait_for_wakeword()
                    else:
                        # If wakeword detection is not available, wait for Enter key
                        print("Press Enter to start conversation (or Ctrl+C to exit)...")
                        await asyncio.get_event_loop().run_in_executor(None, input)
                        self.in_conversation = True
                
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
                            if self.has_wakeword:
                                print("\nWaiting for wake word 'hey_jarvis'...")
                            else:
                                print("\nConversation ended. Press Enter to start a new one...")
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

    async def wait_for_wakeword(self):
        """Wait for wake word detection"""
        try:
            # Buffer for audio data
            audio_buffer = np.zeros(self.chunk_size * 2, dtype=np.float32)
            buffer_index = 0
            
            while self.running and not self.in_conversation:
                # Read audio data from microphone
                audio_data = self.mic_stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Convert to float32 and normalize
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Add to buffer
                audio_buffer[buffer_index:buffer_index + self.chunk_size] = audio_np
                buffer_index = (buffer_index + self.chunk_size) % len(audio_buffer)
                
                # Process with OpenWakeWord
                prediction = self.oww.predict(audio_buffer)
                
                # Check for wake word
                if prediction[0]["hey_jarvis"] > 0.5:  # Threshold can be adjusted
                    print("\n🎯 Wake word detected!")
                    print(f"Confidence: {prediction[0]['hey_jarvis']:.2f}")
                    self.in_conversation = True
                    break
                
                # Short sleep to prevent high CPU usage
                await asyncio.sleep(0.01)
                
        except Exception as e:
            print(f"Error in wakeword detection: {e}")
            # Don't stop the client, just disable wakeword detection
            self.has_wakeword = False
            self.oww = None
            print("Wakeword detection disabled due to error")

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
                    
                # Read audio data from microphone
                try:
                    audio_data = self.mic_stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Simple audio level check
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    audio_level = np.abs(audio_np).mean()
                    
                    # Only warn if audio level is very low and not too frequently
                    if audio_level < 10 and self.sequence % 100 == 0:
                        print(f"Warning: Very low audio level ({audio_level:.1f}). Check your microphone.")
                    
                    # Create audio message
                    message = {
                        "type": "audio_stream",
                        "timestamp": time.time(),
                        "sequence": self.sequence,
                        "payload": {
                            "audio": base64.b64encode(audio_data).decode('utf-8')
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
                    chunk_duration = self.chunk_size / self.mic_sample_rate
                    await asyncio.sleep(chunk_duration * 0.5)  # Sleep for half the chunk duration
                    
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
                        
                        # Convert back to bytes
                        pcm_data = audio_np.tobytes()
            else:
                # Assume it's raw PCM at the speaker sample rate
                pcm_data = audio_data
            
            # Play the audio directly
            if self.speaker_stream:
                print(f"▶️ Playing audio ({len(pcm_data)} bytes)")
                self.speaker_stream.write(pcm_data)
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