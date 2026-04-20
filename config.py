"""
Configuration file for the AI Pneumonia Detection System.
Modify these settings to customize the behavior of training, evaluation, and deployment.

Features:
 - Multi-class classification (Normal, Bacterial Pneumonia, Viral Pneumonia)
 - Ensemble models (DenseNet121, EfficientNet-B4, ResNet50)
 - Cross-validation, class-imbalance handling, attention mechanisms
 - DICOM support, uncertainty quantification, lung segmentation
 - Multiple XAI methods (Grad-CAM, LIME, SHAP, Integrated Gradients)
 - NLP report generation, multilingual support
 - HIPAA/compliance logging, ONNX export, batch processing
 - Federated learning, patient tracking, authentication
 - FastAPI REST API, Docker, CI/CD
"""

from pathlib import Path

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "chest_xray"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
DB_DIR = PROJECT_ROOT / "db"
EXPORTS_DIR = PROJECT_ROOT / "exports"

# ==================== DATASET CONFIGURATION ====================
DATASET_CONFIG = {
    'kaggle_dataset': 'paultimothymooney/chest-xray-pneumonia',
    'image_size': (224, 224),
    # Multi-class: Normal, Bacterial Pneumonia, Viral Pneumonia
    'num_classes': 3,
    'class_names': ['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia'],
    # Legacy binary mode for backward compatibility
    'binary_mode': True,
    'binary_class_names': ['Normal', 'Pneumonia'],
    'binary_num_classes': 2,
    'data_split': {
        'train': 0.8,
        'val': 0.1,
        'test': 0.1
    }
}

# ==================== MODEL CONFIGURATION ====================
MODEL_CONFIG = {
    'architecture': 'densenet121',
    'pretrained': True,
    'freeze_layers': True,
    'dropout_rate': 0.3,
    'num_classes': 2,
    # Attention mechanism: None, 'cbam', or 'se'
    'attention': None,
    # Available architectures for ensemble
    'available_architectures': ['densenet121', 'efficientnet_b4', 'resnet50'],
}

# ==================== ENSEMBLE CONFIGURATION ====================
ENSEMBLE_CONFIG = {
    'enabled': False,
    'models': ['densenet121', 'efficientnet_b4', 'resnet50'],
    'strategy': 'soft_voting',   # 'soft_voting', 'hard_voting', 'weighted_average'
    'weights': [0.4, 0.35, 0.25],  # Per-model weights for weighted_average
}

# ==================== CROSS-VALIDATION CONFIGURATION ====================
CROSS_VALIDATION_CONFIG = {
    'enabled': False,
    'k_folds': 5,
    'stratified': True,
    'save_fold_models': True,
}

# ==================== CLASS IMBALANCE CONFIGURATION ====================
CLASS_IMBALANCE_CONFIG = {
    'use_weighted_loss': True,
    'use_oversampling': False,
    'auto_compute_weights': True,
    'manual_weights': [1.0, 1.5],  # fallback if auto is off
}

# ==================== TRAINING CONFIGURATION ====================
TRAINING_CONFIG = {
    'epochs': 20,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss_function': 'cross_entropy',
    'scheduler': {
        'type': 'reduce_on_plateau',
        'mode': 'min',
        'factor': 0.5,
        'patience': 3,
        'verbose': True
    },
    'early_stopping': {
        'enabled': True,
        'patience': 5,
        'min_delta': 0.001
    },
    'save_best_only': True,
    'num_workers': 4,
    'pin_memory': True,
    'mixed_precision': False,
    'gradient_accumulation_steps': 1,
}

# ==================== UNCERTAINTY QUANTIFICATION ====================
UNCERTAINTY_CONFIG = {
    'enabled': True,
    'method': 'mc_dropout',          # 'mc_dropout', 'temperature_scaling', 'deep_ensemble'
    'mc_dropout_iterations': 30,
    'confidence_threshold': 0.6,
    'temperature': 1.5,              # For temperature scaling
}

# ==================== DICOM CONFIGURATION ====================
DICOM_CONFIG = {
    'enabled': True,
    'allowed_extensions': ['dcm', 'dicom', 'DCM', 'DICOM'],
    'extract_metadata': True,
    'windowing': {
        'center': 40,
        'width': 400
    },
}

# ==================== DATA AUGMENTATION ====================
AUGMENTATION_CONFIG = {
    'train': {
        'random_horizontal_flip': True,
        'random_rotation': 10,  # degrees
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2
        },
        'normalize': {
            'mean': [0.485, 0.456, 0.406],  # ImageNet stats
            'std': [0.229, 0.224, 0.225]
        }
    },
    'val_test': {
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
}

# ==================== GRAD-CAM CONFIGURATION ====================
GRADCAM_CONFIG = {
    'target_layer': 'denseblock4.denselayer16.conv2',
    'colormap': 'jet',
    'overlay_alpha': 0.4,
    'threshold': 0.5
}

# ==================== MULTI-XAI CONFIGURATION ====================
XAI_CONFIG = {
    'methods': ['gradcam', 'lime', 'shap', 'integrated_gradients'],
    'default_method': 'gradcam',
    'lime': {
        'num_samples': 1000,
        'num_features': 10,
        'batch_size': 10,
    },
    'shap': {
        'max_evals': 500,
        'batch_size': 50,
    },
    'integrated_gradients': {
        'n_steps': 50,
        'internal_batch_size': 5,
    },
}

# ==================== LUNG SEGMENTATION CONFIGURATION ====================
SEGMENTATION_CONFIG = {
    'enabled': False,
    'model': 'unet',                    # 'unet' or 'pretrained'
    'pretrained_weights': None,          # path to segmentation model weights
    'threshold': 0.5,
    'apply_before_classification': True,
}

# ==================== 3D VISUALIZATION CONFIGURATION ====================
VISUALIZATION_3D_CONFIG = {
    'lung_model_path': 'data/realistic_human_lungs.glb',
    'colorscale': [
        [0, 'rgb(0,255,0)'],    # Green - Normal
        [0.5, 'rgb(255,255,0)'],  # Yellow - Mild
        [1, 'rgb(255,0,0)']     # Red - Severe
    ],
    'camera_position': {'x': 1.5, 'y': 1.5, 'z': 1.2},
    'figure_height': 600
}

# ==================== SEVERITY CLASSIFICATION ====================
SEVERITY_CONFIG = {
    'confidence_weight': 0.6,  # Weight for model confidence
    'area_weight': 0.4,        # Weight for affected area
    'thresholds': {
        'normal': 0.5,      # Below this = normal
        'mild': 0.4,        # 0.4-0.65 = mild
        'moderate': 0.65,   # 0.65-0.85 = moderate
        'severe': 0.85      # Above this = severe
    },
    'area_thresholds': {  # Percentage of affected lung area
        'mild': 20,
        'moderate': 40,
        'severe': 60
    }
}

# ==================== METRICS CONFIGURATION ====================
METRICS_CONFIG = {
    'primary_metrics': [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'auc_roc'
    ],
    'bleu_samples': 50,  # Number of samples for BLEU score calculation
    'save_visualizations': True,
    'generate_pdf_report': True
}

# ==================== STREAMLIT APP CONFIGURATION ====================
APP_CONFIG = {
    'title': 'AI-Powered Pneumonia Detection System',
    'icon': '🫁',
    'layout': 'wide',
    'theme': {
        'primary_color': '#1E88E5',
        'background_color': '#FFFFFF',
        'secondary_background_color': '#F5F5F5',
        'text_color': '#262730'
    },
    'max_upload_size': 10,
    'allowed_extensions': ['jpg', 'jpeg', 'png', 'dcm', 'dicom'],
    'cache_model': True,
    'enable_batch_upload': True,
    'enable_multi_image_comparison': True,
    'enable_patient_tracking': True,
}

# ==================== NLP REPORT GENERATION ====================
NLP_REPORT_CONFIG = {
    'enabled': True,
    'backend': 'template',               # 'template', 'openai', 'local_llm'
    'openai_model': 'gpt-4',
    'openai_api_key': None,              # Set via environment variable OPENAI_API_KEY
    'local_llm_model': None,             # Path to local GGUF/ONNX LLM model
    'report_format': 'radiology',        # 'radiology', 'patient_friendly', 'both'
    'max_tokens': 500,
}

# ==================== MULTILINGUAL CONFIGURATION ====================
MULTILINGUAL_CONFIG = {
    'enabled': True,
    'default_language': 'en',
    'supported_languages': ['en', 'es', 'fr', 'de', 'zh', 'hi', 'ar', 'pt', 'ja', 'ko'],
    'language_names': {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'zh': 'Chinese', 'hi': 'Hindi', 'ar': 'Arabic', 'pt': 'Portuguese',
        'ja': 'Japanese', 'ko': 'Korean'
    },
}

# ==================== COMPLIANCE / HIPAA LOGGING ====================
COMPLIANCE_CONFIG = {
    'enabled': True,
    'audit_log_file': LOGS_DIR / 'audit.log',
    'log_predictions': True,
    'log_uploads': True,
    'log_user_actions': True,
    'anonymize_patient_data': True,
    'retention_days': 365,
    'encryption_key': None,  # Set via env var AUDIT_ENCRYPTION_KEY
}

# ==================== MODEL EXPORT (ONNX) ====================
ONNX_CONFIG = {
    'enabled': True,
    'export_path': EXPORTS_DIR / 'model.onnx',
    'opset_version': 14,
    'dynamic_axes': True,
    'optimize': True,
}

# ==================== BATCH PROCESSING ====================
BATCH_CONFIG = {
    'enabled': True,
    'max_batch_size': 100,
    'output_format': 'csv',           # 'csv', 'json', 'pdf'
    'include_gradcam': True,
    'parallel_workers': 4,
}

# ==================== FEDERATED LEARNING ====================
FEDERATED_CONFIG = {
    'enabled': False,
    'num_rounds': 10,
    'min_clients': 2,
    'aggregation': 'fedavg',          # 'fedavg', 'fedprox'
    'server_address': 'localhost:8080',
    'differential_privacy': True,
    'noise_multiplier': 1.0,
    'max_grad_norm': 1.0,
}

# ==================== DATABASE / PATIENT TRACKING ====================
DATABASE_CONFIG = {
    'enabled': True,
    'engine': 'sqlite',                # 'sqlite', 'postgresql'
    'sqlite_path': DB_DIR / 'patients.db',
    'postgresql_url': None,            # Set via env var DATABASE_URL
    'track_scans': True,
    'track_progression': True,
}

# ==================== AUTHENTICATION & RBAC ====================
AUTH_CONFIG = {
    'enabled': False,
    'secret_key': 'change-me-in-production',  # Set via env var AUTH_SECRET_KEY
    'algorithm': 'HS256',
    'access_token_expire_minutes': 60,
    'roles': ['admin', 'doctor', 'radiologist', 'patient'],
    'default_role': 'doctor',
}

# ==================== REST API CONFIGURATION ====================
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'workers': 4,
    'cors_origins': ['*'],
    'rate_limit': '100/minute',
    'api_key_required': False,
}

# ==================== DEPLOYMENT CONFIGURATION ====================
DEPLOYMENT_CONFIG = {
    'streamlit_cloud': {
        'python_version': '3.9',
        'port': 8501
    },
    'docker': {
        'base_image': 'python:3.9-slim',
        'port': 8501,
        'api_port': 8000,
    },
    'huggingface': {
        'space_name': 'pneumonia-detection',
        'sdk': 'streamlit',
        'sdk_version': '1.28.0'
    }
}

# ==================== LOGGING CONFIGURATION ====================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_to_file': True,
    'log_dir': LOGS_DIR
}

# ==================== PERFORMANCE OPTIMIZATION ====================
PERFORMANCE_CONFIG = {
    'use_gpu': True,
    'mixed_precision': False,
    'gradient_accumulation_steps': 1,
    'dataloader_workers': 4,
    'prefetch_factor': 2
}

# ==================== MEDICAL RECOMMENDATIONS ====================
RECOMMENDATIONS_CONFIG = {
    'normal': {
        'urgency': 'Low',
        'follow_up': 'Routine checkup in 6-12 months',
        'actions': [
            'No signs of pneumonia detected',
            'Maintain regular health checkups',
            'Practice good respiratory hygiene',
            'Stay up to date with vaccinations'
        ]
    },
    'mild': {
        'urgency': 'Moderate',
        'follow_up': 'Consult physician within 24-48 hours',
        'actions': [
            'Rest and stay hydrated',
            'Monitor symptoms closely',
            'Consider consulting a physician',
            'Avoid strenuous activities'
        ]
    },
    'moderate': {
        'urgency': 'High',
        'follow_up': 'Immediate medical consultation required (within 12-24 hours)',
        'actions': [
            '⚠️ Consult a physician immediately',
            'Likely requires prescription medication',
            'Rest and avoid physical exertion',
            'Monitor vital signs regularly'
        ]
    },
    'severe': {
        'urgency': 'Critical',
        'follow_up': '🚑 EMERGENCY - Go to ER immediately or call 911',
        'actions': [
            '🚨 SEEK IMMEDIATE MEDICAL ATTENTION',
            'This may require hospitalization',
            'Supplemental oxygen may be necessary',
            'Do NOT delay - go to emergency department'
        ]
    }
}

# ==================== EXPORT FUNCTIONS ====================

def get_config(section: str = None):
    """
    Get configuration dictionary.
    
    Args:
        section: Specific section to retrieve (e.g., 'TRAINING_CONFIG')
                If None, returns all configs
    
    Returns:
        Configuration dictionary
    """
    if section:
        return globals().get(section.upper(), {})
    
    # Return all configs
    return {
        'DATASET': DATASET_CONFIG,
        'MODEL': MODEL_CONFIG,
        'ENSEMBLE': ENSEMBLE_CONFIG,
        'CROSS_VALIDATION': CROSS_VALIDATION_CONFIG,
        'CLASS_IMBALANCE': CLASS_IMBALANCE_CONFIG,
        'TRAINING': TRAINING_CONFIG,
        'UNCERTAINTY': UNCERTAINTY_CONFIG,
        'DICOM': DICOM_CONFIG,
        'AUGMENTATION': AUGMENTATION_CONFIG,
        'GRADCAM': GRADCAM_CONFIG,
        'XAI': XAI_CONFIG,
        'SEGMENTATION': SEGMENTATION_CONFIG,
        'VISUALIZATION_3D': VISUALIZATION_3D_CONFIG,
        'SEVERITY': SEVERITY_CONFIG,
        'METRICS': METRICS_CONFIG,
        'APP': APP_CONFIG,
        'NLP_REPORT': NLP_REPORT_CONFIG,
        'MULTILINGUAL': MULTILINGUAL_CONFIG,
        'COMPLIANCE': COMPLIANCE_CONFIG,
        'ONNX': ONNX_CONFIG,
        'BATCH': BATCH_CONFIG,
        'FEDERATED': FEDERATED_CONFIG,
        'DATABASE': DATABASE_CONFIG,
        'AUTH': AUTH_CONFIG,
        'API': API_CONFIG,
        'DEPLOYMENT': DEPLOYMENT_CONFIG,
        'LOGGING': LOGGING_CONFIG,
        'PERFORMANCE': PERFORMANCE_CONFIG,
        'RECOMMENDATIONS': RECOMMENDATIONS_CONFIG
    }


def print_config():
    """Print all configuration settings."""
    import json
    configs = get_config()
    print("="*70)
    print("CONFIGURATION SETTINGS")
    print("="*70)
    for section, config in configs.items():
        print(f"\n{section}:")
        print(json.dumps(config, indent=2))


if __name__ == "__main__":
    print_config()
