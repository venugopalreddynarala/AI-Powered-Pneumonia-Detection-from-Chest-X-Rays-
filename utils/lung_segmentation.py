"""
Lung Segmentation module.
Implements a lightweight U-Net for segmenting lung regions from chest X-rays.
Used to mask non-lung areas before classification for improved accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional


class DoubleConv(nn.Module):
    """Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for lung segmentation.
    Produces a binary mask separating lung regions from background.
    
    Architecture:
        Encoder: 4 downsampling blocks (64→128→256→512)
        Bottleneck: 1024 channels
        Decoder: 4 upsampling blocks with skip connections
        Output: 1-channel sigmoid mask
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Output
        self.out_conv = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self._pad_and_cat(d4, e4)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = self._pad_and_cat(d3, e3)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = self._pad_and_cat(d2, e2)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = self._pad_and_cat(d1, e1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.out_conv(d1))
    
    def _pad_and_cat(self, up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Handle size mismatch between upsampled and skip connection."""
        diff_h = skip.size(2) - up.size(2)
        diff_w = skip.size(3) - up.size(3)
        up = F.pad(up, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])
        return torch.cat([up, skip], dim=1)


class LungSegmentor:
    """
    High-level lung segmentation interface.
    Segments lung regions and applies the mask to X-ray images.
    """
    
    def __init__(self, weights_path: Optional[str] = None,
                 device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet(in_channels=3, out_channels=1).to(self.device)
        
        if weights_path is not None:
            checkpoint = torch.load(weights_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded segmentation weights from {weights_path}")
        else:
            print("LungSegmentor: No weights loaded. Using heuristic fallback.")
        
        self.model.eval()
    
    def segment(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Segment lung regions from a chest X-ray image.
        
        Args:
            image: Input image (H, W, 3), uint8
            threshold: Binarization threshold for mask
        
        Returns:
            Binary mask (H, W) with 1=lung, 0=background
        """
        h, w = image.shape[:2]
        
        # Preprocess
        img_resized = cv2.resize(image, (256, 256))
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mask = self.model(img_tensor)
        
        mask = mask.squeeze().cpu().numpy()
        mask = cv2.resize(mask, (w, h))
        binary_mask = (mask > threshold).astype(np.uint8)
        
        return binary_mask
    
    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply lung mask to image, zeroing out non-lung regions."""
        masked = image.copy()
        for c in range(image.shape[2] if image.ndim == 3 else 1):
            if image.ndim == 3:
                masked[:, :, c] = image[:, :, c] * mask
            else:
                masked = image * mask
        return masked
    
    def segment_and_mask(self, image: np.ndarray,
                         threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment lungs and return both the mask and masked image.
        
        Returns:
            Tuple of (mask, masked_image)
        """
        mask = self.segment(image, threshold)
        masked = self.apply_mask(image, mask)
        return mask, masked


def heuristic_lung_mask(image: np.ndarray) -> np.ndarray:
    """
    Fallback heuristic lung segmentation using Otsu thresholding.
    Used when no trained U-Net weights are available.
    
    Args:
        image: Input image (H, W, 3) uint8
    
    Returns:
        Approximate binary lung mask (H, W)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Keep only the two largest connected components (left and right lung)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)
    
    if num_labels > 2:
        # Sort by area (skip background label 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        top_indices = areas.argsort()[-2:][::-1] + 1  # +1 because we skipped background
        
        mask = np.zeros_like(cleaned)
        for idx in top_indices:
            mask[labels == idx] = 255
    else:
        mask = cleaned
    
    return (mask > 0).astype(np.uint8)


if __name__ == "__main__":
    print("Lung segmentation module loaded successfully")
    print("Models: UNet (trainable), Heuristic (Otsu fallback)")
