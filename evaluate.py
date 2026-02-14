"""
Model evaluation script.
Evaluates trained model on test set and generates comprehensive metrics.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch

from utils.data_prep import create_dataloaders
from utils.metrics import (
    compute_classification_metrics, 
    evaluate_model_performance,
    compute_bleu_score,
    generate_severity_caption
)
from utils.gradcam import generate_gradcam_visualization, get_severity_from_heatmap, tensor_to_numpy_image
from utils.recommendations import SeverityClassifier
from train import build_model


def load_trained_model(weights_path: str, device: torch.device) -> nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        weights_path: Path to model weights
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model = build_model(num_classes=2, pretrained=False)
    
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {weights_path}")
    print(f"Best validation accuracy: {checkpoint['val_acc']:.4f}")
    print(f"Epoch: {checkpoint['epoch']}")
    
    return model


def evaluate_test_set(model: nn.Module,
                     test_loader: torch.utils.data.DataLoader,
                     device: torch.device) -> dict:
    """
    Run inference on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run inference on
        
    Returns:
        Dictionary with predictions and ground truth
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating on test set...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of pneumonia class
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }


def evaluate_with_gradcam_bleu(model: nn.Module,
                               test_loader: torch.utils.data.DataLoader,
                               device: torch.device,
                               num_samples: int = 50) -> dict:
    """
    Evaluate with Grad-CAM visualization and BLEU score.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
        num_samples: Number of samples for BLEU evaluation
        
    Returns:
        Dictionary with BLEU scores and visualizations
    """
    model.eval()
    severity_classifier = SeverityClassifier()
    
    bleu_scores = []
    sample_count = 0
    
    print("\nGenerating Grad-CAM visualizations and computing BLEU scores...")
    
    for inputs, labels in tqdm(test_loader, desc="Grad-CAM + BLEU"):
        if sample_count >= num_samples:
            break
        
        for i in range(inputs.size(0)):
            if sample_count >= num_samples:
                break
            
            input_tensor = inputs[i:i+1].to(device)
            label = labels[i].item()
            
            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.softmax(output, dim=1)
                confidence = prob[0, 1].item()  # Pneumonia confidence
            
            # Generate Grad-CAM
            try:
                original_img = tensor_to_numpy_image(inputs[i])
                heatmap, overlaid, cam_intensity = generate_gradcam_visualization(
                    model, input_tensor, original_img
                )
                
                # Get severity and affected area
                affected_area = severity_classifier.get_affected_area_percentage(
                    cam_intensity * np.ones((224, 224))  # Simplified
                )
                severity, _ = get_severity_from_heatmap(
                    cam_intensity * np.ones((224, 224)), confidence
                )
                
                # Generate captions
                predicted_caption = generate_severity_caption(severity, confidence, affected_area)
                
                # Reference caption based on ground truth
                true_severity = "Severe" if label == 1 else "Normal"
                reference_caption = f"{true_severity} pneumonia with typical presentation."
                
                # Compute BLEU score
                bleu = compute_bleu_score(reference_caption, predicted_caption)
                bleu_scores.append(bleu)
                
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
            
            sample_count += 1
    
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    
    return {
        'bleu_scores': bleu_scores,
        'average_bleu': avg_bleu
    }


def generate_pdf_report(results: dict, save_path: str = 'evaluation_report.pdf'):
    """
    Generate PDF report with evaluation results.
    
    Args:
        results: Dictionary with evaluation results
        save_path: Path to save PDF report
    """
    doc = SimpleDocTemplate(save_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph("<b>AI Pneumonia Detection System</b><br/>Evaluation Report", 
                     styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.3*inch))
    
    # Metrics Summary
    story.append(Paragraph("<b>Performance Metrics</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    metrics = results['metrics']
    metrics_data = [
        ['Metric', 'Score'],
        ['Accuracy', f"{metrics['accuracy']:.4f}"],
        ['Precision', f"{metrics['precision']:.4f}"],
        ['Recall', f"{metrics['recall']:.4f}"],
        ['F1-Score', f"{metrics['f1_score']:.4f}"],
        ['AUC-ROC', f"{metrics['auc_roc']:.4f}"],
        ['BLEU Score', f"{results.get('bleu_score', 0):.4f}"]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Add visualizations if available
    if 'figures' in results:
        story.append(Paragraph("<b>Visualizations</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        for fig_name, fig_path in results['figures'].items():
            if os.path.exists(fig_path):
                story.append(Paragraph(f"<b>{fig_name.replace('_', ' ').title()}</b>", 
                                     styles['Heading3']))
                img = Image(fig_path, width=5*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
    
    # Build PDF
    doc.build(story)
    print(f"\nPDF report saved to {save_path}")


def generate_evaluation_report(model_path: str = 'models/model_weights.pth',
                               data_dir: str = 'data/chest_xray',
                               output_dir: str = 'results'):
    """
    Complete evaluation pipeline.
    
    Args:
        model_path: Path to trained model weights
        data_dir: Path to dataset
        output_dir: Directory to save results
    """
    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = load_trained_model(model_path, device)
    
    # Load test data
    print("\nLoading test dataset...")
    dataloaders = create_dataloaders(data_dir, batch_size=32)
    test_loader = dataloaders['test']
    
    # Evaluate on test set
    test_results = evaluate_test_set(model, test_loader, device)
    
    # Compute metrics
    print("\nComputing metrics...")
    results = evaluate_model_performance(
        test_results['labels'],
        test_results['predictions'],
        test_results['probabilities'],
        save_dir=output_dir
    )
    
    # Evaluate with Grad-CAM and BLEU
    gradcam_results = evaluate_with_gradcam_bleu(
        model, test_loader, device, num_samples=50
    )
    
    results['bleu_score'] = gradcam_results['average_bleu']
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print("\nClassification Metrics:")
    for metric, value in results['metrics'].items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    print(f"\nBLEU Score: {results['bleu_score']:.4f}")
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    print("\n" + "="*70)
    print("Files saved to:", output_dir)
    print("  - metrics.csv")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - metrics_bar.png")
    print("  - classification_report.txt")
    
    # Generate PDF report
    try:
        pdf_path = os.path.join(output_dir, 'evaluation_report.pdf')
        generate_pdf_report(results, pdf_path)
    except Exception as e:
        print(f"\nNote: Could not generate PDF report: {e}")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate pneumonia detection model')
    parser.add_argument('--model_path', type=str, default='models/model_weights.pth',
                       help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str, default='data/chest_xray',
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Run evaluation
    generate_evaluation_report(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
