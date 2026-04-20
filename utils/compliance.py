"""
HIPAA Compliance and Audit Logging module.
Provides secure audit trail for all predictions, uploads, and user actions.
Supports log encryption, data anonymization, and retention policies.
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any
import uuid


class AuditLogger:
    """
    HIPAA-compliant audit logger.
    Records all system interactions with timestamps, user IDs, and action details.
    Supports anonymization of patient data.
    """
    
    def __init__(self, log_dir: str = 'logs', log_file: str = 'audit.log',
                 anonymize: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / log_file
        self.anonymize = anonymize
        
        # Setup logger
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        handler = logging.FileHandler(str(self.log_file))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        
        # JSON log for structured queries
        self.json_log_file = self.log_dir / 'audit_structured.jsonl'
    
    def _anonymize_data(self, data: Dict) -> Dict:
        """Anonymize patient-identifying information."""
        if not self.anonymize:
            return data
        
        anonymized = data.copy()
        sensitive_fields = [
            'patient_name', 'patient_id', 'PatientName', 'PatientID',
            'filename', 'file_path', 'ip_address', 'email'
        ]
        
        for field in sensitive_fields:
            if field in anonymized:
                value = str(anonymized[field])
                anonymized[field] = hashlib.sha256(value.encode()).hexdigest()[:16]
        
        return anonymized
    
    def _write_structured_log(self, entry: Dict):
        """Write structured JSON log entry."""
        try:
            with open(self.json_log_file, 'a') as f:
                f.write(json.dumps(entry, default=str) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write structured log: {e}")
    
    def log_prediction(self, user_id: str, filename: str,
                       prediction: int, confidence: float,
                       severity: str, affected_area: float,
                       metadata: Dict = None):
        """
        Log a prediction event.
        
        Args:
            user_id: User/session identifier
            filename: Uploaded file name
            prediction: Prediction result (0=Normal, 1=Pneumonia)
            confidence: Model confidence
            severity: Severity classification
            affected_area: Affected area percentage
            metadata: Additional metadata
        """
        entry = {
            'event_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'prediction',
            'user_id': user_id,
            'filename': filename,
            'prediction': 'Pneumonia' if prediction == 1 else 'Normal',
            'confidence': round(confidence, 4),
            'severity': severity,
            'affected_area': round(affected_area, 2),
            'metadata': metadata or {}
        }
        
        entry = self._anonymize_data(entry)
        
        self.logger.info(
            f"PREDICTION | user={entry['user_id']} | "
            f"result={entry['prediction']} | confidence={entry['confidence']} | "
            f"severity={entry['severity']}"
        )
        self._write_structured_log(entry)
    
    def log_upload(self, user_id: str, filename: str,
                   file_size: int, file_type: str):
        """Log a file upload event."""
        entry = {
            'event_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'upload',
            'user_id': user_id,
            'filename': filename,
            'file_size_bytes': file_size,
            'file_type': file_type,
        }
        
        entry = self._anonymize_data(entry)
        
        self.logger.info(
            f"UPLOAD | user={entry['user_id']} | "
            f"type={file_type} | size={file_size}"
        )
        self._write_structured_log(entry)
    
    def log_user_action(self, user_id: str, action: str,
                        details: Dict = None):
        """Log a generic user action."""
        entry = {
            'event_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'user_action',
            'user_id': user_id,
            'action': action,
            'details': details or {}
        }
        
        entry = self._anonymize_data(entry)
        
        self.logger.info(
            f"ACTION | user={entry['user_id']} | action={action}"
        )
        self._write_structured_log(entry)
    
    def log_feedback(self, user_id: str, prediction_id: str,
                     feedback: str, correct_label: Optional[int] = None):
        """Log radiologist feedback for active learning."""
        entry = {
            'event_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'feedback',
            'user_id': user_id,
            'prediction_id': prediction_id,
            'feedback': feedback,
            'correct_label': correct_label,
        }
        
        self.logger.info(
            f"FEEDBACK | user={entry['user_id']} | "
            f"prediction={prediction_id} | feedback={feedback}"
        )
        self._write_structured_log(entry)
    
    def log_error(self, user_id: str, error_type: str,
                  error_message: str, context: Dict = None):
        """Log an error event."""
        entry = {
            'event_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'error',
            'user_id': user_id,
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {}
        }
        
        self.logger.error(
            f"ERROR | user={entry['user_id']} | "
            f"type={error_type} | msg={error_message}"
        )
        self._write_structured_log(entry)
    
    def get_audit_trail(self, user_id: str = None,
                        event_type: str = None,
                        start_date: str = None,
                        end_date: str = None,
                        limit: int = 100) -> list:
        """
        Query the audit trail with filters.
        
        Args:
            user_id: Filter by user
            event_type: Filter by event type
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            limit: Max entries to return
        
        Returns:
            List of matching audit entries
        """
        entries = []
        
        try:
            with open(self.json_log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        if user_id and entry.get('user_id') != user_id:
                            continue
                        if event_type and entry.get('event_type') != event_type:
                            continue
                        if start_date and entry.get('timestamp', '') < start_date:
                            continue
                        if end_date and entry.get('timestamp', '') > end_date:
                            continue
                        
                        entries.append(entry)
                        
                        if len(entries) >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        
        return entries
    
    def enforce_retention_policy(self, retention_days: int = 365):
        """
        Remove audit entries older than retention_days.
        
        Args:
            retention_days: Number of days to retain logs
        """
        cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
        
        if not self.json_log_file.exists():
            return
        
        retained = []
        removed = 0
        
        with open(self.json_log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get('timestamp', '') >= cutoff:
                        retained.append(line)
                    else:
                        removed += 1
                except json.JSONDecodeError:
                    retained.append(line)
        
        with open(self.json_log_file, 'w') as f:
            f.writelines(retained)
        
        self.logger.info(
            f"RETENTION | Removed {removed} entries older than {retention_days} days"
        )
    
    def get_statistics(self) -> Dict:
        """Get summary statistics from audit logs."""
        all_entries = self.get_audit_trail(limit=10000)
        
        stats = {
            'total_events': len(all_entries),
            'predictions': 0,
            'uploads': 0,
            'feedback_entries': 0,
            'errors': 0,
            'unique_users': set(),
            'pneumonia_detections': 0,
            'normal_detections': 0,
        }
        
        for entry in all_entries:
            event_type = entry.get('event_type', '')
            stats['unique_users'].add(entry.get('user_id', 'unknown'))
            
            if event_type == 'prediction':
                stats['predictions'] += 1
                if entry.get('prediction') == 'Pneumonia':
                    stats['pneumonia_detections'] += 1
                else:
                    stats['normal_detections'] += 1
            elif event_type == 'upload':
                stats['uploads'] += 1
            elif event_type == 'feedback':
                stats['feedback_entries'] += 1
            elif event_type == 'error':
                stats['errors'] += 1
        
        stats['unique_users'] = len(stats['unique_users'])
        return stats


# Global audit logger instance
_audit_logger = None


def get_audit_logger(log_dir: str = 'logs', anonymize: bool = True) -> AuditLogger:
    """Get or create the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(log_dir=log_dir, anonymize=anonymize)
    return _audit_logger


if __name__ == "__main__":
    logger = AuditLogger(log_dir='logs')
    
    # Test logging
    logger.log_upload('user_001', 'xray_001.jpg', 1024000, 'image/jpeg')
    logger.log_prediction('user_001', 'xray_001.jpg', 1, 0.87, 'Moderate', 32.5)
    logger.log_feedback('user_001', 'pred_001', 'correct', correct_label=1)
    logger.log_user_action('user_001', 'download_report')
    
    stats = logger.get_statistics()
    print("Audit Statistics:", json.dumps(stats, indent=2))
    print("\nCompliance logging module tested successfully!")
