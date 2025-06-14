import asyncio
import websockets
import json
import base64
import time
import sounddevice as sd
import numpy as np
from scipy import signal
import io
import wave
import queue
import threading
import logging

class ServerConnection:
    """
    Manages the WebSocket connection to the server.
    - Connects and initializes speaker only when conversation starts
    - Releases resources immediately when conversation ends
    """
    def __init__(self, server_url, audio_processor):
        self.server_url = server_url
        self.audio_processor = audio_processor
        self.speaker_sample_rate = 24000
        self.output_queue = queue.Queue()
        self._output_stream = None
        self.logger = logging.getLogger(__name__)

    async def handle_conversation(self, initial_audio):
        """Manages the full conversation flow with the server."""
        websocket = None
        output_stream = None
        in_conversation = True
        hanging_up = False
        sequence = 0
        
        try:
            # Initialize connection and speaker for this conversation
            self.logger.info(f"Connecting to {self.server_url}...")
            websocket = await websockets.connect(self.server_url)
            self.logger.info("✅ Connected to server.")
            
            self.logger.info("Initializing speaker...")
            self._output_stream = sd.OutputStream(
                samplerate=self.speaker_sample_rate,
                channels=1,
                dtype='float32',
                blocksize=1024,
                latency='low',
                callback=self._output_callback
            )
            self._output_stream.start()
            self.logger.info("✅ Speaker initialized.")

            # Send the initial audio captured after the wakeword
            if initial_audio is not None and len(initial_audio) > 0:
                message = {
                    "type": "audio_stream",
                    "timestamp": time.time(),
                    "sequence": sequence,
                    "payload": {"audio": base64.b64encode(initial_audio.tobytes()).decode('utf-8')}
                }
                sequence += 1
                await websocket.send(json.dumps(message))
                self.logger.info(f"Sent initial audio chunk ({len(initial_audio)} bytes).")

            # Create tasks for sending and receiving audio
            send_task = asyncio.create_task(
                self._send_loop(websocket, sequence, in_conversation, hanging_up)
            )
            receive_task = asyncio.create_task(
                self._receive_loop(websocket, in_conversation, hanging_up)
            )
            
            # Wait for conversation to end
            done, pending = await asyncio.wait([send_task, receive_task], return_when=asyncio.FIRST_COMPLETED)
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            self.logger.error(f"❌ Error during conversation: {e}")
        finally:
            # Clean up resources immediately
            if websocket:
                try:
                    # Try to send disconnect message, but don't fail if connection is already closed
                    try:
                        disconnect_msg = {"type": "disconnect", "timestamp": time.time(), "sequence": 0}
                        await websocket.send(json.dumps(disconnect_msg))
                    except websockets.exceptions.ConnectionClosed:
                        pass  # Connection already closed, that's fine
                    
                    await websocket.close()
                    self.logger.info("✅ Connection closed.")
                except Exception as e:
                    self.logger.warning(f"Warning: Error closing websocket: {e}")
            
            if self._output_stream:
                try:
                    self._output_stream.stop()
                    self._output_stream.close()
                    self.logger.info("✅ Speaker closed.")
                except Exception as e:
                    self.logger.warning(f"Warning: Error closing speaker: {e}")
                finally:
                    self._output_stream = None
            
            # Clear any remaining audio in the output queue
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.logger.info("Conversation ended.")

    async def _send_loop(self, websocket, sequence, in_conversation, hanging_up):
        """Continuously sends audio from the microphone to the server."""
        try:
            while in_conversation and not hanging_up:
                try:
                    audio_chunk = await self.audio_processor.get_chunk(timeout=0.1)
                    if audio_chunk is not None:
                        message = {
                            "type": "audio_stream",
                            "timestamp": time.time(),
                            "sequence": sequence,
                            "payload": {"audio": base64.b64encode(audio_chunk.tobytes()).decode('utf-8')}
                        }
                        sequence += 1
                        await websocket.send(json.dumps(message))
                    await asyncio.sleep(0.01)
                except websockets.exceptions.ConnectionClosed:
                    self.logger.info("Connection closed during send.")
                    break
        except Exception as e:
            self.logger.error(f"Error in send loop: {e}")

    async def _receive_loop(self, websocket, in_conversation, hanging_up):
        """Continuously receives messages from the server."""
        try:
            while in_conversation and not hanging_up:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    msg = json.loads(message)
                    msg_type = msg.get('type', 'unknown')

                    if msg_type == "audio_playback":
                        hangup_received = self._handle_audio_playback(msg['payload'])
                        if hangup_received:
                            hanging_up = True
                            break
                    elif msg_type == "status":
                        self.logger.info(f"ℹ️ Server status: {msg['payload']}")
                    elif msg_type == "error":
                        self.logger.error(f"❌ Server error: {msg['payload']}")

                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    self.logger.info("Connection closed during receive.")
                    break
                except json.JSONDecodeError:
                    self.logger.warning(f"⚠️ Invalid JSON received")
                except Exception as e:
                    self.logger.error(f"❌ Error processing server message: {e}")
        except Exception as e:
            self.logger.error(f"Error in receive loop: {e}")

    def _handle_audio_playback(self, payload):
        """Handles incoming audio from the server. Returns True if hangup received."""
        is_hangup = payload.get("is_hangup", False)
        
        if is_hangup:
            self.logger.info("📞 Hangup signal received.")

        if "audio" in payload and len(payload["audio"]) > 0:
            audio_data = base64.b64decode(payload["audio"])
            self.logger.debug(f"Queuing audio ({len(audio_data)} bytes)...")
            self._queue_audio(audio_data)

        return is_hangup and payload.get("is_final", False)

    def _output_callback(self, outdata, frames, time, status):
        """Callback for the output stream - fills outdata with audio from the queue."""
        if status:
            self.logger.warning(f"Audio output status: {status}")
        
        try:
            # Get audio chunk from queue (non-blocking)
            audio_chunk = self.output_queue.get_nowait()
            
            # If we have audio data, copy it to the output buffer
            if len(audio_chunk) >= frames:
                outdata[:] = audio_chunk[:frames].reshape(-1, 1)
                # Put remaining audio back in queue if any
                if len(audio_chunk) > frames:
                    remaining = audio_chunk[frames:]
                    self.output_queue.put(remaining)
            else:
                # Pad with zeros if we don't have enough data
                outdata[:len(audio_chunk)] = audio_chunk.reshape(-1, 1)
                outdata[len(audio_chunk):] = 0
                
        except queue.Empty:
            # No audio available, output silence
            outdata.fill(0)

    def _queue_audio(self, audio_data):
        """Processes and queues audio data for playback."""
        try:
            processed_audio = self._process_audio_data(audio_data)
            
            # Split into chunks that match the callback frame size
            chunk_size = 1024  # Match blocksize
            for i in range(0, len(processed_audio), chunk_size):
                chunk = processed_audio[i:i + chunk_size]
                try:
                    self.output_queue.put_nowait(chunk)
                except queue.Full:
                    self.logger.warning("⚠️ Output queue full, dropping audio chunk")
                    break
                    
            self.logger.debug(f"✅ Queued {len(processed_audio)} audio samples")
        except Exception as e:
            self.logger.error(f"❌ Error queuing audio: {e}")

    def _process_audio_data(self, audio_data):
        """Decodes, resamples, and converts audio data to float32."""
        try:
            # Check if it's WAV format
            if audio_data[:4] == b'RIFF':
                with io.BytesIO(audio_data) as wav_buffer:
                    with wave.open(wav_buffer, 'rb') as wf:
                        wav_rate = wf.getframerate()
                        wav_channels = wf.getnchannels()
                        pcm_data = wf.readframes(wf.getnframes())
                        audio_np = np.frombuffer(pcm_data, dtype=np.int16)
                        
                        if wav_channels > 1:
                            audio_np = audio_np.reshape(-1, wav_channels)
                            audio_np = np.mean(audio_np, axis=1, dtype=np.int16)
                        
                        if wav_rate != self.speaker_sample_rate:
                            audio_np = signal.resample_poly(
                                audio_np,
                                self.speaker_sample_rate,
                                wav_rate
                            )
            else:
                # Assume raw PCM at 16kHz, resample to 24kHz
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                audio_np = signal.resample_poly(audio_np, 3, 2)  # 16kHz -> 24kHz
            
            # Convert to float32 for sounddevice
            audio_float = audio_np.astype(np.float32) / 32768.0
            return audio_float
            
        except Exception as e:
            self.logger.error(f"❌ Error processing audio: {e}")
            return np.array([], dtype=np.float32) 