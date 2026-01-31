"""
Face Recognition Engine
Core facial recognition logic using face_recognition library
"""

import face_recognition
import numpy as np
from typing import List, Tuple, Optional
from database_manager import DatabaseManager


class FaceRecognitionEngine:
    def __init__(self, db_manager: DatabaseManager, tolerance: float = 0.6):
        """
        Initialize face recognition engine
        
        Args:
            db_manager: Database manager instance
            tolerance: Recognition threshold (lower = more strict)
                      Default 0.6 is good balance
                      0.5 = strict, 0.7 = loose
        """
        self.db_manager = db_manager
        self.tolerance = tolerance
        self.known_encodings = []
        self.known_names = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load all known face encodings from database"""
        encodings_data = self.db_manager.get_all_encodings()
        self.known_names = []
        self.known_encodings = []
        
        for name, encoding in encodings_data:
            self.known_names.append(name)
            self.known_encodings.append(encoding)
        
        return len(self.known_encodings)
    
    def detect_faces(self, frame: np.ndarray, model: str = "hog") -> List[Tuple]:
        """
        Detect faces in a frame
        
        Args:
            frame: RGB image array
            model: "hog" (faster, CPU) or "cnn" (more accurate, GPU recommended)
        
        Returns:
            List of face locations (top, right, bottom, left)
        """
        face_locations = face_recognition.face_locations(frame, model=model)
        return face_locations
    
    def encode_faces(self, frame: np.ndarray, face_locations: List[Tuple]) -> List[np.ndarray]:
        """
        Generate 128-dimensional encodings for detected faces
        
        Args:
            frame: RGB image array
            face_locations: List of face locations from detect_faces()
        
        Returns:
            List of face encodings (128-d arrays)
        """
        encodings = face_recognition.face_encodings(frame, face_locations)
        return encodings
    
    def recognize_faces(self, frame: np.ndarray, model: str = "hog") -> List[Tuple[Tuple, str, float]]:
        """
        Detect and recognize all faces in a frame
        
        Args:
            frame: RGB image array
            model: Detection model ("hog" or "cnn")
        
        Returns:
            List of (face_location, name, confidence) tuples
        """
        if not self.known_encodings:
            return []
        
        # Detect faces
        face_locations = self.detect_faces(frame, model)
        
        if not face_locations:
            return []
        
        # Encode faces
        face_encodings = self.encode_faces(frame, face_locations)
        
        results = []
        
        for face_location, face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_encodings, 
                face_encoding, 
                tolerance=self.tolerance
            )
            
            name = "Unknown"
            confidence = 0.0
            
            # Calculate face distances (lower = better match)
            face_distances = face_recognition.face_distance(
                self.known_encodings, 
                face_encoding
            )
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
                    # Convert distance to confidence percentage
                    # Distance ranges from 0 (perfect match) to 1+ (no match)
                    confidence = max(0, (1 - face_distances[best_match_index]) * 100)
            
            results.append((face_location, name, confidence))
        
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
        
        # Encode the face
        face_encodings = self.encode_faces(frame, face_locations)
        
        if len(face_encodings) == 0:
            return False, "Failed to encode face"
        
        # Add person if not exists
        person_id = self.db_manager.add_person(name)
        
        if person_id is None:
            return False, "Failed to add person to database"
        
        # Store encoding
        self.db_manager.add_face_encoding(person_id, face_encodings[0])
        
        # Reload known faces
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
            "total_encodings": len(self.known_encodings),
            "tolerance": self.tolerance
        }
