"""
Batch Processing module.
Processes multiple X-ray images at once, generating predictions,
Grad-CAM heatmaps, and reports for all images in a batch.
Supports CSV, JSON, and PDF output.
"""

import os
import csv
import json
import zipfile
import tempfile
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm

from utils.data_prep import preprocess_single_image
from utils.gradcam import generate_gradcam_visualization
from utils.recommendations import SeverityClassifier, generate_recommendations


def process_batch(model: torch.nn.Module, image_paths: List[str],
                  device: torch.device,
                  include_gradcam: bool = True,
                  output_dir: str = 'batch_results') -> Dict:
    """
    Process a batch of X-ray images.
    
    Args:
        model: Trained model
        image_paths: List of image file paths
        device: Compute device
        include_gradcam: Whether to generate Grad-CAM for each image
        output_dir: Directory to save results
    
    Returns:
        Dictionary with batch results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    severity_classifier = SeverityClassifier()
    
    results = []
    
    print(f"\nProcessing batch of {len(image_paths)} images...")
    
    for img_path in tqdm(image_paths, desc="Batch processing"):
        try:
            result = _process_single(
                model, img_path, device, severity_classifier,
                include_gradcam, output_dir
            )
            results.append(result)
        except Exception as e:
            results.append({
                'filename': os.path.basename(img_path),
                'status': 'error',
                'error': str(e)
            })
    
    # Summary statistics
    successful = [r for r in results if r.get('status') != 'error']
    summary = {
        'total_images': len(image_paths),
        'successful': len(successful),
        'errors': len(results) - len(successful),
        'pneumonia_count': sum(1 for r in successful if r.get('prediction') == 1),
        'normal_count': sum(1 for r in successful if r.get('prediction') == 0),
        'timestamp': datetime.now().isoformat(),
    }
    
    if successful:
        confidences = [r['confidence'] for r in successful if 'confidence' in r]
        summary['avg_confidence'] = float(np.mean(confidences))
    
    batch_output = {
        'summary': summary,
        'results': results
    }
    
    print(f"\nBatch complete: {summary['successful']}/{summary['total_images']} successful")
    print(f"  Pneumonia: {summary['pneumonia_count']}, Normal: {summary['normal_count']}")
    
    return batch_output


def _process_single(model, img_path, device, severity_classifier,
                    include_gradcam, output_dir) -> Dict:
    """Process a single image in the batch."""
    filename = os.path.basename(img_path)
    
    # Preprocess and predict
    input_tensor = preprocess_single_image(img_path).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence = probs[0].cpu().numpy()
        prediction = output.argmax(dim=1).item()
    
    result = {
        'filename': filename,
        'filepath': img_path,
        'status': 'success',
        'prediction': prediction,
        'prediction_label': 'Pneumonia' if prediction == 1 else 'Normal',
        'confidence': float(confidence[prediction]),
        'normal_prob': float(confidence[0]),
        'pneumonia_prob': float(confidence[1]),
    }
    
    # Grad-CAM and severity
    if include_gradcam and prediction == 1:
        try:
            original_img = cv2.imread(img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            original_img = cv2.resize(original_img, (224, 224))
            
            heatmap, overlaid, cam_intensity = generate_gradcam_visualization(
                model, input_tensor, original_img
            )
            
            affected_area = severity_classifier.get_affected_area_percentage(
                cam_intensity * np.ones((224, 224))
            )
            severity = severity_classifier.classify(confidence[1], affected_area)
            
            result['severity'] = severity
            result['affected_area'] = float(affected_area)
            result['cam_intensity'] = float(cam_intensity)
            
            # Save overlaid image
            overlay_path = os.path.join(output_dir, f'gradcam_{filename}')
            cv2.imwrite(overlay_path, cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
            result['gradcam_path'] = overlay_path
            
        except Exception as e:
            result['gradcam_error'] = str(e)
    
    return result


def save_batch_csv(batch_results: Dict, output_path: str = 'batch_results/results.csv'):
    """Save batch results to CSV."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    results = batch_results['results']
    
    if not results:
        return
    
    fieldnames = [
        'filename', 'status', 'prediction_label', 'confidence',
        'normal_prob', 'pneumonia_prob', 'severity', 'affected_area'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"CSV results saved to {output_path}")


def save_batch_json(batch_results: Dict, output_path: str = 'batch_results/results.json'):
    """Save batch results to JSON."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Remove non-serializable fields
    clean = json.loads(json.dumps(batch_results, default=str))
    
    with open(output_path, 'w') as f:
        json.dump(clean, f, indent=2)
    
    print(f"JSON results saved to {output_path}")


def create_batch_zip(batch_results: Dict, output_dir: str = 'batch_results',
                     zip_path: str = None) -> str:
    """
    Create a ZIP file containing all batch results and Grad-CAM images.
    
    Returns:
        Path to the ZIP file
    """
    if zip_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_path = os.path.join(output_dir, f'batch_results_{timestamp}.zip')
    
    os.makedirs(os.path.dirname(zip_path) or '.', exist_ok=True)
    
    # First save CSV and JSON
    csv_path = os.path.join(output_dir, 'results.csv')
    json_path = os.path.join(output_dir, 'results.json')
    save_batch_csv(batch_results, csv_path)
    save_batch_json(batch_results, json_path)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add CSV and JSON
        if os.path.exists(csv_path):
            zf.write(csv_path, 'results.csv')
        if os.path.exists(json_path):
            zf.write(json_path, 'results.json')
        
        # Add Grad-CAM images
        for result in batch_results.get('results', []):
            gradcam_path = result.get('gradcam_path')
            if gradcam_path and os.path.exists(gradcam_path):
                zf.write(gradcam_path, f'gradcam/{os.path.basename(gradcam_path)}')
    
    file_size = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"ZIP archive saved to {zip_path} ({file_size:.1f} MB)")
    
    return zip_path


def extract_images_from_zip(zip_path: str, extract_dir: str = None) -> List[str]:
    """
    Extract images from an uploaded ZIP file.
    
    Returns:
        List of extracted image paths
    """
    if extract_dir is None:
        extract_dir = tempfile.mkdtemp(prefix='batch_')
    
    os.makedirs(extract_dir, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm'}
    extracted = []
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            ext = Path(name).suffix.lower()
            if ext in image_extensions and not name.startswith('__MACOSX'):
                zf.extract(name, extract_dir)
                extracted.append(os.path.join(extract_dir, name))
    
    print(f"Extracted {len(extracted)} images from {zip_path}")
    return extracted


if __name__ == "__main__":
    print("Batch processing module loaded successfully")
    print("Features: batch prediction, CSV/JSON/ZIP export, Grad-CAM per image")
