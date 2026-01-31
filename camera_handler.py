"""
Camera Handler
Manages webcam interface using OpenCV
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class CameraHandler:
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        Initialize camera handler
        
        Args:
            camera_index: Camera device index (usually 0 for default webcam)
            width: Frame width
            height: Frame height
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.capture = None
        self.is_active = False
    
    def start(self) -> bool:
        """
        Start the camera
        
        Returns:
            True if successful, False otherwise
        """
        if self.is_active:
            return True
        
        self.capture = cv2.VideoCapture(self.camera_index)
        
        if not self.capture.isOpened():
            return False
        
        # Set camera properties
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.is_active = True
        return True
    
    def stop(self):
        """Stop the camera and release resources"""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.is_active = False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera
        
        Returns:
            (success, frame) tuple where frame is in RGB format
        """
        if not self.is_active or self.capture is None:
            return False, None
        
        ret, frame = self.capture.read()
        
        if not ret:
            return False, None
        
        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB (face_recognition uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return True, frame_rgb
    
    def get_bgr_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame in BGR format (for OpenCV display)
        
        Returns:
            (success, frame) tuple where frame is in BGR format
        """
        if not self.is_active or self.capture is None:
            return False, None
        
        ret, frame = self.capture.read()
        
        if not ret:
            return False, None
        
        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        return True, frame
    
    def is_running(self) -> bool:
        """Check if camera is currently active"""
        return self.is_active
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop()
