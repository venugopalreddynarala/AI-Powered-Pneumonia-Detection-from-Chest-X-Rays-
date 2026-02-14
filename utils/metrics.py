"""
Performance metrics computation and visualization.
Handles accuracy, precision, recall, F1, AUC-ROC, BLEU, and confusion matrix.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def compute_classification_metrics(y_true: np.ndarray, 
                                   y_pred: np.ndarray,
                                   y_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities for positive class
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_proba),
        'specificity': 0.0,  # Will calculate below
        'sensitivity': recall_score(y_true, y_pred, zero_division=0)  # Same as recall
    }
    
    # Calculate specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return metrics, cm


def compute_bleu_score(reference_text: str, generated_text: str) -> float:
    """
    Compute BLEU score for text comparison.
    
    Args:
        reference_text: Ground truth text
        generated_text: Generated/predicted text
        
    Returns:
        BLEU score (0-1)
    """
    # Tokenize
    reference = [reference_text.lower().split()]
    candidate = generated_text.lower().split()
    
    # Use smoothing to avoid zero scores
    smoothing = SmoothingFunction().method1
    
    try:
        score = sentence_bleu(reference, candidate, smoothing_function=smoothing)
    except:
        score = 0.0
    
    return score


def generate_severity_caption(severity: str, confidence: float, affected_area: float) -> str:
    """
    Generate text caption for prediction.
    
    Args:
        severity: Severity level (Mild/Moderate/Severe)
        confidence: Model confidence
        affected_area: Percentage of affected lung area
        
    Returns:
        Generated caption
    """
    caption = f"{severity} pneumonia detected with {confidence*100:.1f}% confidence. "
    caption += f"Approximately {affected_area:.1f}% of lung tissue shows infiltration."
    return caption


def compute_average_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Compute average BLEU score across multiple predictions.
    
    Args:
        predictions: List of generated captions
        references: List of reference captions
        
    Returns:
        Average BLEU score
    """
    scores = []
    for pred, ref in zip(predictions, references):
        score = compute_bleu_score(ref, pred)
        scores.append(score)
    
    return np.mean(scores) if scores else 0.0


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str] = ['Normal', 'Pneumonia'],
                         save_path: str = None) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(y_true: np.ndarray, 
                  y_proba: np.ndarray,
                  save_path: str = None) -> plt.Figure:
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: Ground truth labels
        y_proba: Prediction probabilities
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Pneumonia Detection', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_metrics_bar_chart(metrics: Dict[str, float],
                           save_path: str = None) -> plt.Figure:
    """
    Plot metrics as bar chart.
    
    Args:
        metrics: Dictionary of metric names and values
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Filter out non-primary metrics
    primary_metrics = {
        k: v for k, v in metrics.items() 
        if k in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = list(primary_metrics.keys())
    metric_values = list(primary_metrics.values())
    
    # Create bars with color gradient
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metric_names)))
    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_metrics_to_csv(metrics: Dict[str, float], 
                        save_path: str = 'metrics.csv'):
    """
    Save metrics to CSV file.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save CSV file
    """
    df = pd.DataFrame([metrics])
    df.to_csv(save_path, index=False)
    print(f"Metrics saved to {save_path}")


def generate_classification_report(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   class_names: List[str] = ['Normal', 'Pneumonia']) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Classification report as string
    """
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names,
                                   digits=4)
    return report


def calculate_affected_area(heatmap: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculate percentage of affected area from Grad-CAM heatmap.
    
    Args:
        heatmap: Grad-CAM heatmap (normalized 0-1)
        threshold: Intensity threshold for affected regions
        
    Returns:
        Percentage of affected area
    """
    affected_pixels = (heatmap > threshold).sum()
    total_pixels = heatmap.size
    percentage = (affected_pixels / total_pixels) * 100
    return percentage


def evaluate_model_performance(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_proba: np.ndarray,
                               save_dir: str = 'results') -> Dict:
    """
    Complete model evaluation with all metrics and visualizations.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        save_dir: Directory to save results
        
    Returns:
        Dictionary with all metrics and figure paths
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute metrics
    metrics, cm = compute_classification_metrics(y_true, y_pred, y_proba)
    
    # Generate visualizations
    cm_fig = plot_confusion_matrix(cm, save_path=f'{save_dir}/confusion_matrix.png')
    roc_fig = plot_roc_curve(y_true, y_proba, save_path=f'{save_dir}/roc_curve.png')
    bar_fig = plot_metrics_bar_chart(metrics, save_path=f'{save_dir}/metrics_bar.png')
    
    # Generate classification report
    report = generate_classification_report(y_true, y_pred)
    with open(f'{save_dir}/classification_report.txt', 'w') as f:
        f.write(report)
    
    # Save metrics to CSV
    save_metrics_to_csv(metrics, f'{save_dir}/metrics.csv')
    
    # Close figures to free memory
    plt.close('all')
    
    results = {
        'metrics': metrics,
        'confusion_matrix': cm,
        'report': report,
        'figures': {
            'confusion_matrix': f'{save_dir}/confusion_matrix.png',
            'roc_curve': f'{save_dir}/roc_curve.png',
            'metrics_bar': f'{save_dir}/metrics_bar.png'
        }
    }
    
    return results


if __name__ == "__main__":
    # Test metrics module
    print("Testing metrics computation...")
    
    # Simulate predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_proba = np.random.rand(100)
    
    # Compute metrics
    metrics, cm = compute_classification_metrics(y_true, y_pred, y_proba)
    
    print("\n✓ Metrics computed successfully!")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
