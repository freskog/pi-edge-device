"""
LED Ring Control Module

Controls the LED ring on the ReSpeaker 4-Mic Array (USB) v2.0 with different states:
- OFF: LEDs are off (default state)
- WAKE: LEDs flash briefly when wake word is detected
- ACTIVE: LEDs are on during active conversation

Requires:
pip install pixel-ring
"""

import time
import threading
import usb.core
from pixel_ring import usb_pixel_ring_v2

class LEDControl:
    def __init__(self):
        """Initialize LED control"""
        self.state = "OFF"
        self._lock = threading.Lock()
        
        # Initialize USB device and PixelRing
        try:
            # Find the USB device
            self.dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
            if self.dev is None:
                raise ValueError("ReSpeaker USB Mic Array not found")
            
            # Create PixelRing controller
            self.pixel_ring = usb_pixel_ring_v2.PixelRing(self.dev)
            # Set initial brightness
            self.pixel_ring.set_brightness(10)
            # Turn off LEDs initially
            self.pixel_ring.off()
            print("Found ReSpeaker 4-Mic Array")
            print("LED control initialized")
        except Exception as e:
            print(f"Error initializing LED control: {e}")
            print("⚠️  Warning: Could not initialize LED control")
            print("   LED control will be disabled")
            self.pixel_ring = None
        
    def _set_led_state(self, state):
        """Set LED state using PixelRing"""
        if self.pixel_ring is None:
            print(f"LED State (disabled): {state}")
            return
            
        try:
            if state == "OFF":
                # Turn all LEDs off
                self.pixel_ring.off()
                print("LED State: OFF")
            elif state == "WAKE":
                # Flash effect - blue color
                self.pixel_ring.set_color(r=0, g=0, b=255)
                print("LED State: WAKE (flash)")
            elif state == "ACTIVE":
                # Solid blue color
                self.pixel_ring.set_color(r=0, g=0, b=255)
                print("LED State: ACTIVE")
            
        except Exception as e:
            print(f"Error setting LED state: {e}")
    
    def wake_detected(self):
        """Handle wake word detection"""
        with self._lock:
            self._set_led_state("WAKE")
            # Flash briefly then turn on
            time.sleep(0.5)
            self._set_led_state("ACTIVE")
            self.state = "ACTIVE"
    
    def conversation_started(self):
        """Handle conversation start"""
        with self._lock:
            self._set_led_state("ACTIVE")
            self.state = "ACTIVE"
    
    def conversation_ended(self):
        """Handle conversation end"""
        with self._lock:
            self._set_led_state("OFF")
            self.state = "OFF"
    
    def cleanup(self):
        """Clean up LED resources"""
        with self._lock:
            self._set_led_state("OFF")
            self.state = "OFF" 