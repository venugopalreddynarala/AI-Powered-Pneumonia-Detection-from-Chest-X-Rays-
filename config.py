"""
Configuration file for the AI Pneumonia Detection System.
Modify these settings to customize the behavior of training, evaluation, and deployment.
"""

from pathlib import Path

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "chest_xray"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ==================== DATASET CONFIGURATION ====================
DATASET_CONFIG = {
    'kaggle_dataset': 'paultimothymooney/chest-xray-pneumonia',
    'image_size': (224, 224),
    'num_classes': 2,
    'class_names': ['Normal', 'Pneumonia'],
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
    'freeze_layers': True,  # Freeze early layers for transfer learning
    'dropout_rate': 0.3,
    'num_classes': 2
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
    'pin_memory': True
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
    'colormap': 'jet',  # OpenCV colormap
    'overlay_alpha': 0.4,  # Transparency of heatmap overlay
    'threshold': 0.5  # Threshold for affected area calculation
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
    'max_upload_size': 10,  # MB
    'allowed_extensions': ['jpg', 'jpeg', 'png'],
    'cache_model': True
}

# ==================== DEPLOYMENT CONFIGURATION ====================
DEPLOYMENT_CONFIG = {
    'streamlit_cloud': {
        'python_version': '3.9',
        'port': 8501
    },
    'docker': {
        'base_image': 'python:3.9-slim',
        'port': 8501
    },
    'huggingface': {
        'space_name': 'pneumonia-detection',
        'sdk': 'streamlit',
        'sdk_version': '1.28.0'
    }
}

# ==================== LOGGING CONFIGURATION ====================
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_to_file': True,
    'log_dir': PROJECT_ROOT / 'logs'
}

# ==================== PERFORMANCE OPTIMIZATION ====================
PERFORMANCE_CONFIG = {
    'use_gpu': True,
    'mixed_precision': False,  # FP16 training (requires GPU)
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
        'TRAINING': TRAINING_CONFIG,
        'AUGMENTATION': AUGMENTATION_CONFIG,
        'GRADCAM': GRADCAM_CONFIG,
        'VISUALIZATION_3D': VISUALIZATION_3D_CONFIG,
        'SEVERITY': SEVERITY_CONFIG,
        'METRICS': METRICS_CONFIG,
        'APP': APP_CONFIG,
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
