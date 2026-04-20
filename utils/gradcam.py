"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
Generates heatmaps showing which regions of the X-ray influenced the model's prediction.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN visualization."""
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: Trained CNN model
            target_layer: Layer to compute gradients (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor: torch.Tensor, 
                     target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None = use predicted class)
            
        Returns:
            Heatmap as numpy array (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU (only positive influences)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def generate_heatmap(self, input_tensor: torch.Tensor,
                        target_class: Optional[int] = None,
                        size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Generate colored heatmap overlay.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            size: Output size (H, W)
            
        Returns:
            Colored heatmap as numpy array (H, W, 3)
        """
        cam = self.generate_cam(input_tensor, target_class)
        
        # Resize CAM to match input size
        cam_resized = cv2.resize(cam, size)
        
        # Apply colormap (COLORMAP_JET: blue -> red)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap


def get_target_layer(model, model_name: str = "densenet121"):
    """
    Get the appropriate target layer for Grad-CAM.
    
    Args:
        model: PyTorch model
        model_name: Name of the model architecture
        
    Returns:
        Target layer for Grad-CAM
    """
    if "densenet" in model_name.lower():
        # Last dense block for DenseNet
        return model.features.denseblock4.denselayer16.conv2
    elif "resnet" in model_name.lower():
        # Last layer of layer4 for ResNet
        return model.layer4[-1].conv2
    elif "vgg" in model_name.lower():
        # Last conv layer for VGG
        return model.features[-1]
    else:
        # Default: try to find last conv layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        raise ValueError(f"Could not find target layer for {model_name}")


def overlay_heatmap(original_img: np.ndarray, 
                   heatmap: np.ndarray,
                   alpha: float = 0.4) -> np.ndarray:
    """
    Overlay heatmap on original image.
    
    Args:
        original_img: Original image (H, W, 3) in RGB, range [0, 255]
        heatmap: Heatmap image (H, W, 3) in RGB, range [0, 255]
        alpha: Transparency of heatmap overlay
        
    Returns:
        Overlaid image as numpy array
    """
    # Ensure both are uint8
    original_img = np.uint8(original_img)
    heatmap = np.uint8(heatmap)
    
    # Blend images
    overlaid = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
    
    return overlaid


def get_severity_from_heatmap(heatmap: np.ndarray, 
                              confidence: float) -> Tuple[str, str]:
    """
    Determine severity level and color based on heatmap intensity and confidence.
    
    Args:
        heatmap: Grad-CAM heatmap (normalized [0, 1])
        confidence: Model prediction confidence
        
    Returns:
        Tuple of (severity_level, color_hex)
    """
    # Calculate affected area percentage (high intensity regions)
    high_intensity = (heatmap > 0.6).sum() / heatmap.size * 100
    
    # Determine severity
    if confidence < 0.6:
        severity = "Normal/Uncertain"
        color = "#00FF00"  # Green
    elif high_intensity < 15:
        severity = "Mild"
        color = "#FFFF00"  # Yellow
    elif high_intensity < 35:
        severity = "Moderate"
        color = "#FFA500"  # Orange
    else:
        severity = "Severe"
        color = "#FF0000"  # Red
    
    return severity, color


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert normalized tensor to displayable numpy image.
    
    Args:
        tensor: Image tensor (C, H, W) or (1, C, H, W)
        
    Returns:
        Numpy array (H, W, 3) in range [0, 255]
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    
    # Clip to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and reorder dimensions
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    
    return image


def generate_gradcam_visualization(model: torch.nn.Module,
                                  input_tensor: torch.Tensor,
                                  original_image: np.ndarray,
                                  target_class: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Complete Grad-CAM visualization pipeline.
    
    Args:
        model: Trained model
        input_tensor: Preprocessed input tensor
        original_image: Original image as numpy array
        target_class: Target class (None = predicted class)
        
    Returns:
        Tuple of (heatmap, overlaid_image, cam_intensity, cam_raw)
        - heatmap: JET-colorized heatmap (H, W, 3) uint8
        - overlaid_image: heatmap blended on original (H, W, 3) uint8
        - cam_intensity: scalar mean activation in high regions
        - cam_raw: raw activation map (H, W) float in [0, 1]
    """
    # Get target layer
    target_layer = get_target_layer(model, "densenet121")
    
    # Create Grad-CAM instance
    gradcam = GradCAM(model, target_layer)
    
    # Generate raw CAM and colorized heatmap
    cam = gradcam.generate_cam(input_tensor, target_class)
    heatmap = gradcam.generate_heatmap(input_tensor, target_class, 
                                       size=(original_image.shape[1], original_image.shape[0]))
    
    # Resize raw CAM to image dimensions for downstream use
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    
    # Overlay on original image
    overlaid = overlay_heatmap(original_image, heatmap, alpha=0.4)
    
    # Calculate average intensity in high-activation regions
    cam_intensity = cam[cam > 0.5].mean() if (cam > 0.5).any() else cam.mean()
    
    return heatmap, overlaid, float(cam_intensity), cam_resized


if __name__ == "__main__":
    print("Grad-CAM module loaded successfully")
    print("Key functions:")
    print("  - GradCAM: Main class for generating activation maps")
    print("  - overlay_heatmap: Overlay heatmap on original image")
    print("  - get_severity_from_heatmap: Determine severity level")
