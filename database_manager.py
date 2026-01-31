"""
Database Manager for Facial Recognition System
Handles storage and retrieval of face encodings using SQLite
"""

import sqlite3
import pickle
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional


class DatabaseManager:
    def __init__(self, db_path: str = "face_recognition.db"):
        """Initialize database connection and create tables if needed"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                registered_date TEXT NOT NULL
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_encodings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                encoding BLOB NOT NULL,
                captured_date TEXT NOT NULL,
                FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE
            )
        ''')
        
        self.conn.commit()
    
    def add_person(self, name: str) -> int:
        """
        Add a new person to the database
        Returns: person_id
        """
        try:
            self.cursor.execute(
                "INSERT INTO persons (name, registered_date) VALUES (?, ?)",
                (name, datetime.now().isoformat())
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            # Person already exists, return their ID
            self.cursor.execute("SELECT id FROM persons WHERE name = ?", (name,))
            result = self.cursor.fetchone()
            return result[0] if result else None
    
    def add_face_encoding(self, person_id: int, encoding: np.ndarray):
        """Add a face encoding for a person"""
        encoding_blob = pickle.dumps(encoding)
        self.cursor.execute(
            "INSERT INTO face_encodings (person_id, encoding, captured_date) VALUES (?, ?, ?)",
            (person_id, encoding_blob, datetime.now().isoformat())
        )
        self.conn.commit()
    
    def get_all_encodings(self) -> List[Tuple[str, np.ndarray]]:
        """
        Get all face encodings with person names
        Returns: List of (name, encoding) tuples
        """
        self.cursor.execute('''
            SELECT p.name, fe.encoding
            FROM face_encodings fe
            JOIN persons p ON fe.person_id = p.id
        ''')
        
        results = []
        for row in self.cursor.fetchall():
            name = row[0]
            encoding = pickle.loads(row[1])
            results.append((name, encoding))
        
        return results
    
    def get_person_encodings(self, name: str) -> List[np.ndarray]:
        """Get all encodings for a specific person"""
        self.cursor.execute('''
            SELECT fe.encoding
            FROM face_encodings fe
            JOIN persons p ON fe.person_id = p.id
            WHERE p.name = ?
        ''', (name,))
        
        return [pickle.loads(row[0]) for row in self.cursor.fetchall()]
    
    def delete_person(self, name: str) -> bool:
        """Delete a person and all their face encodings"""
        self.cursor.execute("DELETE FROM persons WHERE name = ?", (name,))
        self.conn.commit()
        return self.cursor.rowcount > 0
    
    def get_all_persons(self) -> List[Tuple[str, int, str]]:
        """
        Get all registered persons
        Returns: List of (name, encoding_count, registered_date) tuples
        """
        self.cursor.execute('''
            SELECT p.name, COUNT(fe.id), p.registered_date
            FROM persons p
            LEFT JOIN face_encodings fe ON p.id = fe.person_id
            GROUP BY p.id, p.name, p.registered_date
            ORDER BY p.registered_date DESC
        ''')
        
        return self.cursor.fetchall()
    
    def clear_all_data(self):
        """Delete all persons and encodings"""
        self.cursor.execute("DELETE FROM face_encodings")
        self.cursor.execute("DELETE FROM persons")
        self.conn.commit()
    
    def get_person_count(self) -> int:
        """Get total number of registered persons"""
        self.cursor.execute("SELECT COUNT(*) FROM persons")
        return self.cursor.fetchone()[0]
    
    def person_exists(self, name: str) -> bool:
        """Check if a person is already registered"""
        self.cursor.execute("SELECT COUNT(*) FROM persons WHERE name = ?", (name,))
        return self.cursor.fetchone()[0] > 0
    
    def close(self):
        """Close database connection"""
        self.conn.close()
