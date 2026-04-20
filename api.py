"""
FastAPI REST API for the X-Ray Pneumonia Detection System.
Provides programmatic access to predictions, patient management,
and system features via RESTful endpoints.
"""

import os
import sys
import uuid
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

# FastAPI imports
try:
    from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
    from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("FastAPI not installed. Install with: pip install fastapi uvicorn")

import torch
import numpy as np


# ==================== Pydantic Models ====================

if HAS_FASTAPI:
    
    class Token(BaseModel):
        access_token: str
        token_type: str
        role: str
        username: str
    
    class UserCreate(BaseModel):
        username: str
        password: str
        role: str = 'viewer'
        full_name: str = ''
    
    class UserResponse(BaseModel):
        username: str
        role: str
        full_name: str
        is_active: bool
        created_at: str
    
    class PredictionResponse(BaseModel):
        prediction_id: int
        filename: str
        prediction: int
        prediction_label: str
        confidence: float
        severity: Optional[str] = None
        affected_area: Optional[float] = None
        uncertainty: Optional[float] = None
        recommendations: Optional[List[str]] = None
    
    class PatientCreate(BaseModel):
        patient_id: str
        name: Optional[str] = None
        age: Optional[int] = None
        gender: Optional[str] = None
        contact: Optional[str] = None
        notes: Optional[str] = None
    
    class FeedbackCreate(BaseModel):
        prediction_id: int
        is_correct: Optional[bool] = None
        correct_label: Optional[int] = None
        comments: Optional[str] = None
    
    class BatchResponse(BaseModel):
        total_images: int
        successful: int
        errors: int
        pneumonia_count: int
        normal_count: int
        results: List[Dict]
    
    # ==================== App Setup ====================
    
    app = FastAPI(
        title="X-Ray Pneumonia Detection API",
        description="AI-powered chest X-ray analysis with Grad-CAM explainability",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)
    
    # ==================== Global State ====================
    
    _model = None
    _device = None
    
    
    def get_model():
        """Load model lazily on first request."""
        global _model, _device
        if _model is None:
            from config import get_config
            config = get_config()
            _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            from utils.ensemble import build_single_model
            _model = build_single_model(
                architecture=config['model']['architecture'],
                num_classes=config['dataset']['num_classes'],
                pretrained=False
            )
            
            weights_path = config['model']['save_path']
            if os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=_device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    _model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    _model.load_state_dict(checkpoint, strict=False)
            
            _model.to(_device)
            _model.eval()
        
        return _model, _device
    
    
    async def get_current_user(token: str = Depends(oauth2_scheme)):
        """Validate token and return current user."""
        if token is None:
            return None
        
        from auth import verify_token
        payload = verify_token(token)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    
    
    def require_auth(user=Depends(get_current_user)):
        """Require authenticated user."""
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        return user
    
    
    # ==================== Auth Endpoints ====================
    
    @app.post("/auth/login", response_model=Token)
    async def login(form_data: OAuth2PasswordRequestForm = Depends()):
        """Authenticate and receive an access token."""
        from auth import get_user_manager, create_access_token
        
        manager = get_user_manager()
        user = manager.authenticate(form_data.username, form_data.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        token = create_access_token({
            'sub': user['username'],
            'role': user['role'],
        })
        
        return Token(
            access_token=token,
            token_type="bearer",
            role=user['role'],
            username=user['username']
        )
    
    
    @app.post("/auth/register", response_model=UserResponse)
    async def register(user_data: UserCreate):
        """Register a new user (admin only in production)."""
        from auth import get_user_manager
        
        manager = get_user_manager()
        result = manager.create_user(
            user_data.username, user_data.password,
            user_data.role, user_data.full_name
        )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result['message'])
        
        user = manager.get_user(user_data.username)
        return UserResponse(**user)
    
    
    # ==================== Prediction Endpoints ====================
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: UploadFile = File(...),
        patient_id: Optional[str] = None,
    ):
        """
        Upload an X-ray image and get a pneumonia prediction.
        Returns prediction, confidence, severity, and recommendations.
        """
        model, device = get_model()
        
        # Save uploaded file temporarily
        suffix = Path(file.filename or 'upload.jpg').suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            from utils.data_prep import preprocess_single_image
            from utils.recommendations import SeverityClassifier, generate_recommendations
            from utils.gradcam import generate_gradcam_visualization
            import cv2
            
            # Preprocess and predict
            input_tensor = preprocess_single_image(tmp_path).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                confidence = probs[0].cpu().numpy()
                prediction = output.argmax(dim=1).item()
            
            # Severity classification
            severity_classifier = SeverityClassifier()
            result_data = {
                'filename': file.filename,
                'prediction': prediction,
                'prediction_label': 'Pneumonia' if prediction == 1 else 'Normal',
                'confidence': float(confidence[prediction]),
            }
            
            if prediction == 1:
                original_img = cv2.imread(tmp_path)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                original_img = cv2.resize(original_img, (224, 224))
                
                try:
                    _, _, cam_intensity = generate_gradcam_visualization(
                        model, input_tensor, original_img
                    )
                    affected_area = severity_classifier.get_affected_area_percentage(
                        cam_intensity * np.ones((224, 224))
                    )
                    severity = severity_classifier.classify(
                        float(confidence[1]), affected_area
                    )
                    result_data['severity'] = severity
                    result_data['affected_area'] = float(affected_area)
                except Exception:
                    result_data['severity'] = 'Unknown'
                
                recs = generate_recommendations(
                    'Pneumonia', result_data.get('severity', 'Moderate'),
                    result_data.get('affected_area', 30.0),
                    float(confidence[1])
                )
                result_data['recommendations'] = recs.get('actions', []) if isinstance(recs, dict) else []
            
            # Save to database
            try:
                from database import get_database
                db = get_database()
                pred_id = db.add_prediction(
                    filename=file.filename,
                    prediction=prediction,
                    prediction_label=result_data['prediction_label'],
                    confidence=result_data['confidence'],
                    patient_id=patient_id,
                    severity=result_data.get('severity'),
                    affected_area=result_data.get('affected_area'),
                )
                result_data['prediction_id'] = pred_id
            except Exception:
                result_data['prediction_id'] = 0
            
            # Audit log
            try:
                from utils.compliance import get_audit_logger
                logger = get_audit_logger()
                logger.log_prediction(
                    'api_user', file.filename,
                    prediction, result_data['confidence'],
                    result_data.get('severity', 'N/A'),
                    result_data.get('affected_area', 0)
                )
            except Exception:
                pass
            
            return PredictionResponse(**result_data)
        
        finally:
            os.unlink(tmp_path)
    
    
    @app.post("/predict/batch", response_model=BatchResponse)
    async def batch_predict(files: List[UploadFile] = File(...)):
        """Process multiple X-ray images in a batch."""
        model, device = get_model()
        
        # Save all files temporarily
        tmp_paths = []
        for file in files:
            suffix = Path(file.filename or 'upload.jpg').suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_paths.append((tmp.name, file.filename))
        
        try:
            from utils.batch_processor import process_batch
            
            image_paths = [p[0] for p in tmp_paths]
            results = process_batch(model, image_paths, device, include_gradcam=False)
            
            # Map back filenames
            for i, result in enumerate(results.get('results', [])):
                if i < len(tmp_paths):
                    result['filename'] = tmp_paths[i][1]
            
            return BatchResponse(
                total_images=results['summary']['total_images'],
                successful=results['summary']['successful'],
                errors=results['summary']['errors'],
                pneumonia_count=results['summary']['pneumonia_count'],
                normal_count=results['summary']['normal_count'],
                results=results['results']
            )
        finally:
            for path, _ in tmp_paths:
                try:
                    os.unlink(path)
                except Exception:
                    pass
    
    
    # ==================== Patient Endpoints ====================
    
    @app.post("/patients")
    async def create_patient(patient: PatientCreate):
        """Create a new patient record."""
        from database import get_database
        db = get_database()
        result = db.add_patient(**patient.dict())
        
        if result['status'] == 'exists':
            raise HTTPException(status_code=409, detail="Patient ID already exists")
        
        return result
    
    
    @app.get("/patients")
    async def list_patients(limit: int = 100, offset: int = 0):
        """List all patients."""
        from database import get_database
        db = get_database()
        return db.list_patients(limit=limit, offset=offset)
    
    
    @app.get("/patients/{patient_id}")
    async def get_patient(patient_id: str):
        """Get patient details."""
        from database import get_database
        db = get_database()
        patient = db.get_patient(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        return patient
    
    
    @app.get("/patients/{patient_id}/history")
    async def get_patient_history(patient_id: str, limit: int = 50):
        """Get prediction history for a patient."""
        from database import get_database
        db = get_database()
        history = db.get_patient_history(patient_id, limit=limit)
        return history
    
    
    # ==================== Feedback Endpoints ====================
    
    @app.post("/feedback")
    async def submit_feedback(feedback: FeedbackCreate):
        """Submit radiologist feedback for a prediction."""
        from database import get_database
        db = get_database()
        
        prediction = db.get_prediction(feedback.prediction_id)
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        feedback_id = db.add_feedback(
            prediction_id=feedback.prediction_id,
            reviewer_id='api_user',
            is_correct=feedback.is_correct,
            correct_label=feedback.correct_label,
            comments=feedback.comments,
        )
        
        return {'feedback_id': feedback_id, 'status': 'success'}
    
    
    # ==================== Dashboard Endpoints ====================
    
    @app.get("/dashboard/stats")
    async def get_dashboard_stats():
        """Get comprehensive dashboard statistics."""
        from database import get_database
        db = get_database()
        return db.get_dashboard_stats()
    
    
    @app.get("/models")
    async def list_models():
        """List all registered model versions."""
        from database import get_database
        db = get_database()
        return db.list_model_versions()
    
    
    # ==================== Health Check ====================
    
    @app.get("/health")
    async def health_check():
        """API health check endpoint."""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'gpu_available': torch.cuda.is_available(),
        }
    
    
    @app.get("/")
    async def root():
        """API root - documentation redirect."""
        return {
            'message': 'X-Ray Pneumonia Detection API',
            'version': '2.0.0',
            'docs': '/docs',
            'health': '/health',
        }


def start_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server."""
    if not HAS_FASTAPI:
        print("FastAPI is not installed. Run: pip install fastapi uvicorn")
        return
    
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_api()
