"""
Face Recognition Engine (OpenCV Alternative)
Uses OpenCV's built-in LBPH face recognition with improvements
"""

import cv2
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
from database_manager import DatabaseManager


class FaceRecognitionEngine:
    def __init__(self, db_manager: DatabaseManager, tolerance: float = 100.0):
        """
        Initialize face recognition engine using OpenCV LBPH
        
        Args:
            db_manager: Database manager instance
            tolerance: Recognition threshold (lower = more strict)
                      Default 100.0 is balanced
        """
        self.db_manager = db_manager
        self.tolerance = tolerance
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_to_name = {}
        self.name_to_label = {}
        self.trained = False
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load all known faces from database and train recognizer"""
        persons = self.db_manager.get_all_persons()
        
        if not persons or len(persons) == 0:
            self.trained = False
            return 0
        
        # Collect all training data
        faces = []
        labels = []
        
        self.label_to_name.clear()
        self.name_to_label.clear()
        
        label_id = 0
        for name, count, _ in persons:
            self.label_to_name[label_id] = name
            self.name_to_label[name] = label_id
            
            # Get encodings for this person (stored as images for LBPH)
            person_encodings = self.db_manager.get_person_encodings(name)
            for encoding in person_encodings:
                # Encoding is stored as image data for LBPH
                faces.append(encoding)
                labels.append(label_id)
            
            label_id += 1
        
        if len(faces) > 0:
            self.recognizer.train(faces, np.array(labels))
            self.trained = True
            return len(faces)
        
        self.trained = False
        return 0
    
    def detect_faces(self, frame: np.ndarray, model: str = "hog") -> List[Tuple]:
        """
        Detect faces in a frame
        
        Args:
            frame: RGB or grayscale image array
            model: Ignored (for compatibility)
        
        Returns:
            List of face locations (top, right, bottom, left)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        # Convert from (x, y, w, h) to (top, right, bottom, left)
        face_locations = []
        for (x, y, w, h) in faces:
            top = y
            right = x + w
            bottom = y + h
            left = x
            face_locations.append((top, right, bottom, left))
        
        return face_locations
    
    def recognize_faces(self, frame: np.ndarray, model: str = "hog") -> List[Tuple[Tuple, str, float]]:
        """
        Detect and recognize all faces in a frame
        
        Args:
            frame: RGB image array
            model: Ignored (for compatibility)
        
        Returns:
            List of (face_location, name, confidence) tuples
        """
        if not self.trained:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        face_locations = self.detect_faces(frame)
        
        if not face_locations:
            return []
        
        results = []
        
        for face_location in face_locations:
            top, right, bottom, left = face_location
            
            # Extract and resize face
            face_roi = gray[top:bottom, left:right]
            face_resized = cv2.resize(face_roi, (200, 200))
            
            # Recognize
            label, confidence = self.recognizer.predict(face_resized)
            
            name = "Unknown"
            conf_percentage = 0.0
            
            # Lower confidence value means better match for LBPH
            if confidence < self.tolerance and label in self.label_to_name:
                name = self.label_to_name[label]
                # Convert LBPH confidence to percentage (inverse relationship)
                conf_percentage = max(0, 100 - confidence)
            
            results.append((face_location, name, conf_percentage))
        
        return results
    
    def register_face(self, frame: np.ndarray, name: str) -> Tuple[bool, str]:
        """
        Register a new face encoding
        
        Args:
            frame: RGB image array containing a face
            name: Person's name
        
        Returns:
            (success, message) tuple
        """
        # Detect faces in frame
        face_locations = self.detect_faces(frame)
        
        if len(face_locations) == 0:
            return False, "No face detected"
        
        if len(face_locations) > 1:
            return False, "Multiple faces detected - only one person should be visible"
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Extract face
        top, right, bottom, left = face_locations[0]
        face_roi = gray[top:bottom, left:right]
        face_resized = cv2.resize(face_roi, (200, 200))
        
        # Add person if not exists
        person_id = self.db_manager.add_person(name)
        
        if person_id is None:
            return False, "Failed to add person to database"
        
        # Store face image as encoding (for LBPH)
        self.db_manager.add_face_encoding(person_id, face_resized)
        
        # Reload and retrain
        self.load_known_faces()
        
        return True, "Face registered successfully"
    
    def delete_person(self, name: str) -> bool:
        """Delete a person and reload known faces"""
        success = self.db_manager.delete_person(name)
        if success:
            self.load_known_faces()
        return success
    
    def clear_all_faces(self):
        """Clear all registered faces"""
        self.db_manager.clear_all_data()
        self.load_known_faces()
    
    def get_statistics(self) -> dict:
        """Get recognition statistics"""
        return {
            "total_persons": self.db_manager.get_person_count(),
            "total_encodings": len(self.label_to_name),
            "tolerance": self.tolerance
        }
