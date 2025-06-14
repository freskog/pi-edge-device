
import os

# Set the latency to 20ms to ensure that reference audio is in sycn with microphone audio
os.environ["PULSE_LATENCY_MSEC"] = "20"

import sounddevice as sd
import numpy as np
import queue
import asyncio
import logging

class AudioProcessor:
    """
    Handles all audio input operations.
    - Manages the microphone input stream.
    - Puts incoming audio data into a queue.
    """
    def __init__(self, sample_rate=16000, channels=6, channel_index=0, chunk_size=512):
        self.sample_rate = sample_rate
        self.channels = channels
        self.channel_index = channel_index
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self._input_stream = None
        self.logger = logging.getLogger(__name__)

    def _audio_callback(self, indata, frames, time_info, status):
        """This callback is called by the sounddevice stream for each new audio chunk."""
        if status:
            self.logger.warning(f"Audio input status: {status}")

        # Select the correct channel and convert to int16 PCM
        audio_chunk = indata[:, self.channel_index]
        audio_chunk = (audio_chunk * 32768).astype(np.int16)
        try:
            self.audio_queue.put_nowait(audio_chunk.copy())
        except queue.Full:
            # Drop audio if queue is full to prevent blocking
            self.logger.warning("Audio input queue full, dropping chunk")

    def start(self):
        """Starts the microphone input stream."""
        self.logger.info("Starting audio stream...")
        try:
            # Print device list for diagnostics
            self.logger.info("🔍 Available audio input devices:")
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0:
                    self.logger.info(f"  [{i}] {dev['name']} — {dev['max_input_channels']} channels")

            self.logger.info(f"🎤 Listening on channel {self.channel_index}...")
            self._input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',
                blocksize=self.chunk_size,
                callback=self._audio_callback
            )
            self._input_stream.start()
            self.logger.info("✅ Audio stream started.")
        except Exception as e:
            self.logger.error(f"❌ Audio initialization failed: {e}")
            raise

    def stop(self):
        """Stops the microphone input stream."""
        if self._input_stream:
            self.logger.info("Stopping audio stream...")
            try:
                self._input_stream.stop()
                self._input_stream.close()
                self.logger.info("✅ Audio stream stopped.")
            except Exception as e:
                self.logger.error(f"❌ Error closing input stream: {e}")
            finally:
                self._input_stream = None
        self.clear_queue()

    def clear_queue(self):
        """Removes all items from the audio queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def flush_audio_buffers(self):
        """
        Properly flushes all audio buffers including hardware and sounddevice buffers.
        This stops new audio from coming in, clears everything, then resumes.
        """
        if not self._input_stream:
            return
            
        try:
            self.logger.debug("🔄 Stopping audio stream...")
            # Stop the stream to prevent new audio from coming in
            self._input_stream.stop()
            
            self.logger.debug("🧹 Clearing audio queue...")
            # Clear the Python queue
            self.clear_queue()
            
            self.logger.debug("▶️ Restarting audio stream...")
            # Restart the stream with fresh buffers
            self._input_stream.start()
            self.logger.debug("✅ Audio buffers flushed successfully.")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error flushing audio buffers: {e}")
            # If something goes wrong, try to restart the stream
            try:
                self.logger.info("🔧 Attempting to recover audio stream...")
                if self._input_stream:
                    self._input_stream.start()
                self.logger.info("✅ Audio stream recovered.")
            except Exception as recovery_error:
                self.logger.error(f"❌ Failed to recover audio stream: {recovery_error}")
                # Set stream to None so we know it's broken
                self._input_stream = None

    async def get_chunk(self, timeout=0.5):
        """
        Asynchronously gets the next audio chunk from the queue.
        Returns None if the queue is empty after the timeout.
        """
        loop = asyncio.get_event_loop()
        
        def _get_chunk():
            try:
                return self.audio_queue.get(timeout=timeout)
            except queue.Empty:
                return None
        
        # Run the blocking queue.get in a thread pool
        return await loop.run_in_executor(None, _get_chunk) 