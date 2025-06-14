import asyncio
import argparse
import signal
import sys
import logging
from audio import AudioProcessor
from wakeword import WakewordDetector
from connection import ServerConnection
# from led_control import LEDControl # Assuming led_control.py exists and is correct

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle Ctrl+C more aggressively."""
    logger = logging.getLogger(__name__)
    logger.warning(f"🛑 Received signal {signum}. Force exiting...")
    sys.exit(1)

async def main(args):
    """
    Main function to run the audio client.
    - Initializes always-available components (mic, wakeword detector)
    - Per-conversation components (speaker, server connection) are handled in ServerConnection
    """
    # Set up logging
    logger = setup_logging()
    
    # Set up signal handler for more responsive Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚀 Starting voice assistant...")
    
    # --- Initialize always-available components ---
    try:
        logger.info("Initializing always-available components...")
        
        # Audio processor - always running
        audio_processor = AudioProcessor(
            sample_rate=16000,
            channels=6,
            channel_index=args.channel,
            chunk_size=512
        )
        audio_processor.start()
        
        # Wakeword detector - always available
        logger.info("Loading wakeword detector...")
        wakeword_detector = WakewordDetector(
            sample_rate=audio_processor.sample_rate,
            chunk_size=audio_processor.chunk_size
        )
        
        # Server connection factory - creates connection per conversation
        def create_server_connection():
            return ServerConnection(args.server, audio_processor)
        
        # led_control = LEDControl()
        
        logger.info("✅ Always-available components initialized")

        # --- Main conversation loop ---
        conversation_count = 0
        
        while True:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Ready for conversation #{conversation_count + 1}")
                logger.info(f"{'='*50}")
                
                # 1. Listen for wakeword (always-available components)
                # led_control.set_listening()
                initial_audio = await wakeword_detector.wait_for_wakeword(audio_processor)
                
                # 2. Handle conversation (per-conversation components)
                # led_control.set_active()
                logger.info("🗣️ Starting conversation...")
                
                # Create fresh connection for this conversation
                server_connection = create_server_connection()
                await server_connection.handle_conversation(initial_audio)
                
                # Connection and speaker are automatically cleaned up in handle_conversation
                # led_control.set_off()
                
                conversation_count += 1
                
                # 3. Important: Clear stale audio thoroughly
                logger.debug("💤 Clearing stale audio...")
                audio_processor.clear_queue()
                await asyncio.sleep(0.1)  # Brief pause to let any in-flight audio settle
                audio_processor.clear_queue()
                logger.info("Ready to listen for next wakeword...")
                
            except Exception as e:
                logger.error(f"❌ Error in conversation loop: {e}")
                logger.info("Clearing audio and continuing to listen for wakeword...")
                audio_processor.clear_queue()
                await asyncio.sleep(2.0)

    except KeyboardInterrupt:
        logger.warning("\n🛑 User interruption detected.")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("\nShutting down...")
        if 'audio_processor' in locals():
            audio_processor.stop()
        # if 'led_control' in locals():
        #     led_control.set_off()
        logger.info("✅ Cleanup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Assistant Edge Client")
    parser.add_argument("--server", default="ws://localhost:8765", help="WebSocket server URL")
    parser.add_argument("--channel", type=int, default=0, help="Index of the microphone channel to use.")
    args = parser.parse_args()
    
    asyncio.run(main(args)) 