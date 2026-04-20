"""
Multi-XAI (Explainable AI) Methods module.
Implements LIME, SHAP, and Integrated Gradients alongside existing Grad-CAM.
Provides a unified interface for generating explanations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ====================== INTEGRATED GRADIENTS ======================

class IntegratedGradients:
    """
    Integrated Gradients attribution method.
    Computes feature importance by integrating gradients along a straight
    path from a baseline (e.g., black image) to the input.
    
    Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
    
    def attribute(self, input_tensor: torch.Tensor,
                  target_class: Optional[int] = None,
                  n_steps: int = 50,
                  baseline: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Compute Integrated Gradients attribution.
        
        Args:
            input_tensor: Input image (1, C, H, W)
            target_class: Target class (None = predicted)
            n_steps: Number of interpolation steps
            baseline: Baseline input (default: zeros)
        
        Returns:
            Attribution map (H, W)
        """
        self.model.eval()
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Get predicted class if not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, n_steps + 1, device=input_tensor.device)
        
        # Accumulate gradients
        integrated_grads = torch.zeros_like(input_tensor)
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)
            
            output = self.model(interpolated)
            score = output[0, target_class]
            
            self.model.zero_grad()
            score.backward()
            
            if interpolated.grad is not None:
                integrated_grads += interpolated.grad.detach()
        
        # Average and multiply by (input - baseline)
        integrated_grads = (integrated_grads / (n_steps + 1)) * (input_tensor - baseline)
        
        # Sum across channels and take absolute value
        attr_map = integrated_grads.squeeze(0).sum(dim=0).abs().cpu().numpy()
        
        # Normalize to [0, 1]
        if attr_map.max() > attr_map.min():
            attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min())
        
        return attr_map


# ====================== LIME ======================

class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for image classification.
    Perturbs superpixels and fits a linear model to explain individual predictions.
    
    Reference: Ribeiro et al., "Why Should I Trust You?", KDD 2016
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cpu')
    
    def _get_superpixels(self, image: np.ndarray, n_segments: int = 50) -> np.ndarray:
        """Generate superpixel segmentation."""
        try:
            from skimage.segmentation import slic
            segments = slic(image, n_segments=n_segments, compactness=10, sigma=1)
        except ImportError:
            # Fallback: grid-based segmentation
            h, w = image.shape[:2]
            grid_size = max(1, int(np.sqrt(n_segments)))
            segments = np.zeros((h, w), dtype=int)
            cell_h, cell_w = h // grid_size, w // grid_size
            for i in range(grid_size):
                for j in range(grid_size):
                    segments[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = i * grid_size + j
        return segments
    
    def explain(self, input_tensor: torch.Tensor, original_image: np.ndarray,
                target_class: Optional[int] = None,
                num_samples: int = 1000,
                num_features: int = 10) -> np.ndarray:
        """
        Generate LIME explanation for an image prediction.
        
        Args:
            input_tensor: Preprocessed input tensor (1, C, H, W)
            original_image: Original image (H, W, 3), uint8
            target_class: Target class to explain
            num_samples: Number of perturbed samples
            num_features: Number of top features (superpixels) to show
        
        Returns:
            Attribution map (H, W) normalized [0, 1]
        """
        self.model.eval()
        
        # Get prediction
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor.to(self.device))
                target_class = output.argmax(dim=1).item()
        
        # Get superpixels
        img_resized = cv2.resize(original_image, (224, 224))
        segments = self._get_superpixels(img_resized)
        n_superpixels = segments.max() + 1
        
        # Generate perturbed samples
        perturbations = np.random.binomial(1, 0.5, (num_samples, n_superpixels))
        predictions = []
        
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        for perm in perturbations:
            perturbed = img_resized.copy()
            for sp_idx in range(n_superpixels):
                if perm[sp_idx] == 0:
                    perturbed[segments == sp_idx] = 128  # Gray out
            
            tensor = preprocess(Image.fromarray(perturbed)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
                prob = F.softmax(out, dim=1)[0, target_class].item()
                predictions.append(prob)
        
        predictions = np.array(predictions)
        
        # Fit weighted linear model
        from sklearn.linear_model import Ridge
        
        # Distance weights (closer perturbations get higher weight)
        original_perm = np.ones(n_superpixels)
        distances = np.sqrt(((perturbations - original_perm) ** 2).sum(axis=1))
        kernel_width = 0.25 * n_superpixels
        weights = np.exp(-distances ** 2 / (2 * kernel_width ** 2))
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(perturbations, predictions, sample_weight=weights)
        
        # Get feature importance
        importance = ridge.coef_
        
        # Map back to image space
        attr_map = np.zeros(segments.shape, dtype=np.float64)
        for sp_idx in range(n_superpixels):
            attr_map[segments == sp_idx] = importance[sp_idx]
        
        # Normalize to [0, 1]
        attr_map = np.abs(attr_map)
        if attr_map.max() > 0:
            attr_map = attr_map / attr_map.max()
        
        return attr_map


# ====================== SHAP (Gradient SHAP) ======================

class GradientSHAP:
    """
    Gradient SHAP approximation for image attribution.
    Uses random baselines and computes expected gradients.
    
    Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
    
    def attribute(self, input_tensor: torch.Tensor,
                  target_class: Optional[int] = None,
                  n_samples: int = 50,
                  stdev: float = 0.09) -> np.ndarray:
        """
        Compute Gradient SHAP attribution.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class
            n_samples: Number of random baseline samples
            stdev: Noise standard deviation for baselines
        
        Returns:
            Attribution map (H, W)
        """
        self.model.eval()
        
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        device = input_tensor.device
        attributions = torch.zeros_like(input_tensor)
        
        for _ in range(n_samples):
            # Random baseline with noise
            baseline = torch.randn_like(input_tensor) * stdev
            
            # Random alpha
            alpha = torch.rand(1, device=device)
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)
            
            output = self.model(interpolated)
            score = output[0, target_class]
            
            self.model.zero_grad()
            score.backward()
            
            if interpolated.grad is not None:
                grad = interpolated.grad.detach()
                attributions += grad * (input_tensor - baseline)
        
        attributions = attributions / n_samples
        
        # Sum across channels
        attr_map = attributions.squeeze(0).sum(dim=0).abs().cpu().numpy()
        
        # Normalize
        if attr_map.max() > attr_map.min():
            attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min())
        
        return attr_map


# ====================== UNIFIED INTERFACE ======================

def generate_explanation(model: torch.nn.Module, input_tensor: torch.Tensor,
                        original_image: np.ndarray,
                        method: str = 'gradcam',
                        target_class: Optional[int] = None,
                        device: torch.device = None,
                        **kwargs) -> Dict:
    """
    Unified interface for generating XAI explanations.
    
    Args:
        model: Trained model
        input_tensor: Preprocessed input tensor
        original_image: Original image as numpy array
        method: One of 'gradcam', 'lime', 'shap', 'integrated_gradients'
        target_class: Target class to explain
        device: Compute device
        **kwargs: Method-specific parameters
    
    Returns:
        Dictionary with 'attribution_map', 'heatmap_overlay', 'method'
    """
    if device is None:
        device = next(model.parameters()).device
    
    input_tensor = input_tensor.to(device)
    
    if method == 'integrated_gradients':
        ig = IntegratedGradients(model)
        attr_map = ig.attribute(
            input_tensor, target_class,
            n_steps=kwargs.get('n_steps', 50)
        )
    
    elif method == 'lime':
        lime = LIMEExplainer(model, device)
        attr_map = lime.explain(
            input_tensor, original_image, target_class,
            num_samples=kwargs.get('num_samples', 1000),
            num_features=kwargs.get('num_features', 10)
        )
    
    elif method == 'shap':
        shap = GradientSHAP(model)
        attr_map = shap.attribute(
            input_tensor, target_class,
            n_samples=kwargs.get('n_samples', 50)
        )
    
    elif method == 'gradcam':
        # Use existing Grad-CAM from gradcam.py
        from utils.gradcam import generate_gradcam_visualization
        heatmap, overlaid, intensity = generate_gradcam_visualization(
            model, input_tensor, original_image, target_class
        )
        return {
            'method': 'gradcam',
            'attribution_map': heatmap,
            'heatmap_overlay': overlaid,
            'intensity': intensity,
        }
    
    else:
        raise ValueError(f"Unknown XAI method: {method}")
    
    # Resize attribution map to image size
    h, w = original_image.shape[:2]
    attr_resized = cv2.resize(attr_map, (w, h))
    
    # Create heatmap overlay
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * attr_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(
        np.uint8(original_image), 0.6, heatmap_colored, 0.4, 0
    )
    
    return {
        'method': method,
        'attribution_map': attr_resized,
        'heatmap_overlay': overlay,
        'intensity': float(attr_resized.mean()),
    }


def compare_xai_methods(model: torch.nn.Module, input_tensor: torch.Tensor,
                        original_image: np.ndarray,
                        methods: list = None,
                        target_class: Optional[int] = None,
                        device: torch.device = None):
    """
    Run multiple XAI methods and return a matplotlib comparison figure.
    
    Returns:
        matplotlib.figure.Figure with side-by-side XAI visualizations
    """
    import matplotlib.pyplot as plt

    if methods is None:
        methods = ['gradcam', 'integrated_gradients', 'shap']
    
    results = {}
    for method in methods:
        try:
            results[method] = generate_explanation(
                model, input_tensor, original_image,
                method=method, target_class=target_class, device=device
            )
        except Exception as e:
            results[method] = {'error': str(e), 'method': method}

    # Build side-by-side comparison figure
    n_cols = len(results) + 1  # +1 for original image
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    # Show original image first
    axes[0].imshow(original_image)
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    method_labels = {
        'gradcam': 'Grad-CAM',
        'integrated_gradients': 'Integrated\nGradients',
        'lime': 'LIME',
        'shap': 'Gradient SHAP',
    }

    for idx, (method, res) in enumerate(results.items(), start=1):
        ax = axes[idx]
        if 'error' in res:
            ax.text(0.5, 0.5, f"Error:\n{res['error'][:60]}",
                    ha='center', va='center', fontsize=8, color='red',
                    transform=ax.transAxes, wrap=True)
            ax.set_facecolor('#1a1a1a')
        elif 'heatmap_overlay' in res:
            ax.imshow(res['heatmap_overlay'])
        elif 'visualization' in res:
            ax.imshow(res['visualization'])
        label = method_labels.get(method, method)
        intensity = res.get('intensity', None)
        title = label
        if intensity is not None:
            title += f'\n(mean={intensity:.3f})'
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    fig.suptitle('XAI Methods Comparison', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Multi-XAI methods module loaded successfully")
    print("Methods: Integrated Gradients, LIME, Gradient SHAP, Grad-CAM")
