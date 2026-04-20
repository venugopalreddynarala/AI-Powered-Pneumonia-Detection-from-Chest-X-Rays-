"""
Database module for patient tracking and history.
Uses SQLite for lightweight, file-based storage.
Tracks patients, predictions, feedback, and model versions.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class Database:
    """
    SQLite database for patient records and prediction history.
    """
    
    def __init__(self, db_path: str = 'data/xray_system.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        """Create all database tables."""
        cursor = self.conn.cursor()
        
        # Patients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE NOT NULL,
                name TEXT,
                age INTEGER,
                gender TEXT,
                contact TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                filename TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                prediction_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT,
                affected_area REAL,
                uncertainty REAL,
                model_version TEXT,
                gradcam_path TEXT,
                report TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        ''')
        
        # Feedback table (radiologist feedback for active learning)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                reviewer_id TEXT,
                is_correct BOOLEAN,
                correct_label INTEGER,
                comments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        ''')
        
        # Model versions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT UNIQUE NOT NULL,
                architecture TEXT,
                accuracy REAL,
                f1_score REAL,
                training_config TEXT,
                weights_path TEXT,
                is_active BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sessions table (for user tracking)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                user_id TEXT,
                ip_address TEXT,
                user_agent TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    # ==================== Patient operations ====================
    
    def add_patient(self, patient_id: str, name: str = None,
                    age: int = None, gender: str = None,
                    contact: str = None, notes: str = None) -> Dict:
        """Add a new patient record."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                '''INSERT INTO patients (patient_id, name, age, gender, contact, notes)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (patient_id, name, age, gender, contact, notes)
            )
            self.conn.commit()
            return {'status': 'success', 'patient_id': patient_id}
        except sqlite3.IntegrityError:
            return {'status': 'exists', 'patient_id': patient_id}
    
    def get_patient(self, patient_id: str) -> Optional[Dict]:
        """Get patient by ID."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def list_patients(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """List all patients."""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT * FROM patients ORDER BY created_at DESC LIMIT ? OFFSET ?',
            (limit, offset)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def search_patients(self, query: str) -> List[Dict]:
        """Search patients by name or ID."""
        cursor = self.conn.cursor()
        search_term = f'%{query}%'
        cursor.execute(
            '''SELECT * FROM patients 
               WHERE patient_id LIKE ? OR name LIKE ?
               ORDER BY created_at DESC''',
            (search_term, search_term)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def update_patient(self, patient_id: str, **kwargs) -> bool:
        """Update patient fields."""
        allowed_fields = {'name', 'age', 'gender', 'contact', 'notes'}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return False
        
        set_clause = ', '.join(f'{k} = ?' for k in updates)
        values = list(updates.values()) + [patient_id]
        
        cursor = self.conn.cursor()
        cursor.execute(
            f'UPDATE patients SET {set_clause}, updated_at = CURRENT_TIMESTAMP '
            f'WHERE patient_id = ?',
            values
        )
        self.conn.commit()
        return cursor.rowcount > 0
    
    # ==================== Prediction operations ====================
    
    def add_prediction(self, filename: str, prediction: int,
                       prediction_label: str, confidence: float,
                       patient_id: str = None, severity: str = None,
                       affected_area: float = None, uncertainty: float = None,
                       model_version: str = None, gradcam_path: str = None,
                       report: str = None, metadata: Dict = None) -> int:
        """Add a prediction record. Returns the prediction ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            '''INSERT INTO predictions 
               (patient_id, filename, prediction, prediction_label, confidence,
                severity, affected_area, uncertainty, model_version, gradcam_path,
                report, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (patient_id, filename, prediction, prediction_label, confidence,
             severity, affected_area, uncertainty, model_version, gradcam_path,
             report, json.dumps(metadata) if metadata else None)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def get_prediction(self, prediction_id: int) -> Optional[Dict]:
        """Get prediction by ID."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_patient_history(self, patient_id: str, limit: int = 50) -> List[Dict]:
        """Get all predictions for a patient, most recent first."""
        cursor = self.conn.cursor()
        cursor.execute(
            '''SELECT * FROM predictions 
               WHERE patient_id = ? 
               ORDER BY created_at DESC LIMIT ?''',
            (patient_id, limit)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_recent_predictions(self, limit: int = 20) -> List[Dict]:
        """Get most recent predictions across all patients."""
        cursor = self.conn.cursor()
        cursor.execute(
            '''SELECT p.*, pt.name as patient_name 
               FROM predictions p
               LEFT JOIN patients pt ON p.patient_id = pt.patient_id
               ORDER BY p.created_at DESC LIMIT ?''',
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    # ==================== Feedback operations ====================
    
    def add_feedback(self, prediction_id: int, reviewer_id: str = None,
                     is_correct: bool = None, correct_label: int = None,
                     comments: str = None) -> int:
        """Add radiologist feedback for a prediction."""
        cursor = self.conn.cursor()
        cursor.execute(
            '''INSERT INTO feedback 
               (prediction_id, reviewer_id, is_correct, correct_label, comments)
               VALUES (?, ?, ?, ?, ?)''',
            (prediction_id, reviewer_id, is_correct, correct_label, comments)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def get_feedback_for_prediction(self, prediction_id: int) -> List[Dict]:
        """Get all feedback for a specific prediction."""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT * FROM feedback WHERE prediction_id = ? ORDER BY created_at DESC',
            (prediction_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics for model improvement."""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as total FROM feedback')
        total = cursor.fetchone()['total']
        
        cursor.execute('SELECT COUNT(*) as correct FROM feedback WHERE is_correct = 1')
        correct = cursor.fetchone()['correct']
        
        cursor.execute('SELECT COUNT(*) as incorrect FROM feedback WHERE is_correct = 0')
        incorrect = cursor.fetchone()['incorrect']
        
        return {
            'total_feedback': total,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': correct / total if total > 0 else 0
        }
    
    # ==================== Model version operations ====================
    
    def add_model_version(self, version: str, architecture: str = None,
                          accuracy: float = None, f1_score: float = None,
                          training_config: Dict = None,
                          weights_path: str = None) -> int:
        """Register a new model version."""
        cursor = self.conn.cursor()
        cursor.execute(
            '''INSERT INTO model_versions 
               (version, architecture, accuracy, f1_score, training_config, weights_path)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (version, architecture, accuracy, f1_score,
             json.dumps(training_config) if training_config else None, weights_path)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def set_active_model(self, version: str) -> bool:
        """Set a model version as the active one."""
        cursor = self.conn.cursor()
        cursor.execute('UPDATE model_versions SET is_active = 0')
        cursor.execute(
            'UPDATE model_versions SET is_active = 1 WHERE version = ?',
            (version,)
        )
        self.conn.commit()
        return cursor.rowcount > 0
    
    def get_active_model(self) -> Optional[Dict]:
        """Get the currently active model version."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM model_versions WHERE is_active = 1')
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def list_model_versions(self) -> List[Dict]:
        """List all registered model versions."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM model_versions ORDER BY created_at DESC')
        return [dict(row) for row in cursor.fetchall()]
    
    # ==================== Dashboard statistics ====================
    
    def get_dashboard_stats(self) -> Dict:
        """Get comprehensive dashboard statistics."""
        cursor = self.conn.cursor()
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) as count FROM predictions')
        total_predictions = cursor.fetchone()['count']
        
        # Pneumonia vs Normal
        cursor.execute(
            'SELECT prediction_label, COUNT(*) as count FROM predictions GROUP BY prediction_label'
        )
        prediction_counts = {row['prediction_label']: row['count'] for row in cursor.fetchall()}
        
        # Severity distribution
        cursor.execute(
            'SELECT severity, COUNT(*) as count FROM predictions WHERE severity IS NOT NULL GROUP BY severity'
        )
        severity_dist = {row['severity']: row['count'] for row in cursor.fetchall()}
        
        # Average confidence
        cursor.execute('SELECT AVG(confidence) as avg_conf FROM predictions')
        avg_confidence = cursor.fetchone()['avg_conf'] or 0
        
        # Total patients
        cursor.execute('SELECT COUNT(*) as count FROM patients')
        total_patients = cursor.fetchone()['count']
        
        # Today's predictions
        cursor.execute(
            "SELECT COUNT(*) as count FROM predictions WHERE date(created_at) = date('now')"
        )
        today_count = cursor.fetchone()['count']
        
        return {
            'total_predictions': total_predictions,
            'total_patients': total_patients,
            'today_predictions': today_count,
            'prediction_counts': prediction_counts,
            'severity_distribution': severity_dist,
            'avg_confidence': round(avg_confidence, 4),
            'feedback_stats': self.get_feedback_stats(),
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()


# Global database instance
_db = None


def get_database(db_path: str = 'data/xray_system.db') -> Database:
    """Get or create the global database instance."""
    global _db
    if _db is None:
        _db = Database(db_path=db_path)
    return _db


if __name__ == "__main__":
    db = Database(db_path='data/xray_system_test.db')
    
    # Test operations
    db.add_patient('P001', name='Test Patient', age=45, gender='M')
    db.add_prediction('xray_001.jpg', 1, 'Pneumonia', 0.87,
                      patient_id='P001', severity='Moderate', affected_area=32.5)
    db.add_feedback(1, reviewer_id='DR001', is_correct=True, comments='Looks correct')
    db.add_model_version('v1.0', architecture='DenseNet121', accuracy=0.92)
    
    stats = db.get_dashboard_stats()
    print("Dashboard stats:", json.dumps(stats, indent=2))
    
    history = db.get_patient_history('P001')
    print(f"\nPatient P001 history: {len(history)} predictions")
    
    db.close()
    
    # Cleanup test DB
    os.remove('data/xray_system_test.db')
    print("\nDatabase module tested successfully!")
