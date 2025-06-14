import numpy as np
import webrtcvad
import time
import asyncio
import logging
from openwakeword.model import Model

class WakewordDetector:
    """
    Handles wakeword detection.
    - Uses VAD (Voice Activity Detection) to filter non-speech.
    - Uses openwakeword to detect the specified wakeword.
    - Always available, resets after each detection.
    """
    def __init__(self, sample_rate=16000, chunk_size=512):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.vad_frame_size = int(sample_rate * 0.03)  # 30ms for VAD
        self.vad = webrtcvad.Vad(2)  # Aggressiveness: 0-3 (3 is most permissive)
        self.model = Model(wakeword_models=["hey_mycroft"])
        self._last_detection_time = 0
        self.logger = logging.getLogger(__name__)
        self.logger.info("✅ Wakeword detector initialized.")

    async def wait_for_wakeword(self, audio_processor, confidence_threshold=0.7, detection_gap=2.0):
        """
        Listens to the audio stream from the AudioProcessor and returns when a wakeword is detected.
        Resets state after each detection.
        
        Returns:
            np.array: The audio buffer containing speech immediately following the wakeword.
        """
        # Reset state for new detection cycle
        post_wakeword_buffer = np.zeros((0,), dtype=np.int16)
        history_buffer = np.zeros((0,), dtype=np.int16)
        history_buffer_duration = 1.5  # seconds
        history_buffer_size = int(self.sample_rate * history_buffer_duration)

        self.logger.info("👂 Listening for 'hey mycroft'...")
        
        # Clear any stale audio from previous conversation
        audio_processor.clear_queue()

        while True:
            audio_chunk = await audio_processor.get_chunk()
            if audio_chunk is None:
                continue

            # Add to history buffer (rolling window)
            history_buffer = np.concatenate((history_buffer, audio_chunk))
            if len(history_buffer) > history_buffer_size:
                history_buffer = history_buffer[-history_buffer_size:]

            # Process in chunks
            processing_buffer = np.concatenate((post_wakeword_buffer, audio_chunk))
            while len(processing_buffer) >= self.chunk_size:
                frame = processing_buffer[:self.chunk_size]
                processing_buffer = processing_buffer[self.chunk_size:]

                # VAD check - skip if no speech detected
                vad_frame = frame[:self.vad_frame_size]
                if not self.vad.is_speech(vad_frame.tobytes(), self.sample_rate):
                    continue

                # Wakeword prediction
                prediction = self.model.predict(frame)
                confidence = prediction.get("hey_mycroft", 0.0)

                now = time.time()
                if confidence > confidence_threshold and (now - self._last_detection_time) > detection_gap:
                    self.logger.info(f"🎯 Wakeword detected! (Confidence: {confidence:.2f})")
                    self._last_detection_time = now
                    
                    # Return recent audio history plus any remaining processing buffer
                    # This captures the command that follows the wakeword
                    return np.concatenate((history_buffer, processing_buffer))

            post_wakeword_buffer = processing_buffer

    def reset(self):
        """Reset the detector state after a conversation ends."""
        # No persistent state to reset - each wait_for_wakeword call starts fresh
        pass 