"""
LED Ring Control Module

Controls the LED ring on the ReSpeaker 4-Mic Array (USB) v2.0 with two states:
- OFF: LEDs are off (default state)
- ACTIVE: LEDs are solid blue during active conversation

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
        
    def set_active(self):
        """Set LED to active (solid blue)"""
        with self._lock:
            if self.pixel_ring is None:
                print("LED State (disabled): ACTIVE")
                return
            try:
                self.pixel_ring.set_color(r=0, g=0, b=255)
                print("LED State: ACTIVE")
                self.state = "ACTIVE"
            except Exception as e:
                print(f"Error setting LED state: {e}")
    
    def set_tool_active(self):
        """Set LED to tool active state (flashing blue)"""
        with self._lock:
            if self.pixel_ring is None:
                print("LED State (disabled): TOOL_ACTIVE")
                return
            try:
                # Start a flashing pattern
                self.pixel_ring.think()  # Use think() for flashing pattern
                print("LED State: TOOL_ACTIVE")
                self.state = "TOOL_ACTIVE"
            except Exception as e:
                print(f"Error setting LED state: {e}")
    
    def set_error(self):
        """Set LED to error state (solid red)"""
        with self._lock:
            if self.pixel_ring is None:
                print("LED State (disabled): ERROR")
                return
            try:
                self.pixel_ring.set_color(r=255, g=0, b=0)
                print("LED State: ERROR")
                self.state = "ERROR"
            except Exception as e:
                print(f"Error setting LED state: {e}")
    
    def set_off(self):
        """Turn off LEDs"""
        with self._lock:
            if self.pixel_ring is None:
                print("LED State (disabled): OFF")
                return
            try:
                self.pixel_ring.off()
                print("LED State: OFF")
                self.state = "OFF"
            except Exception as e:
                print(f"Error setting LED state: {e}")
    
    def cleanup(self):
        """Clean up LED resources"""
        self.set_off() 