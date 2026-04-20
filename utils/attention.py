"""
Attention mechanisms for CNN models.
Implements CBAM (Convolutional Block Attention Module) and SE (Squeeze-and-Excitation) blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Adaptively recalibrates channel-wise feature responses by modelling 
    interdependencies between channels.
    
    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze: global average pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: FC → ReLU → FC → Sigmoid
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """Channel attention module for CBAM."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Applies channel attention followed by spatial attention.
    
    Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        x = x * self.channel_attention(x)
        # Spatial attention
        x = x * self.spatial_attention(x)
        return x


class AttentionWrapper(nn.Module):
    """
    Wraps a backbone model (e.g. DenseNet121) with an attention module
    inserted after the feature extractor and before the classifier.
    """
    
    def __init__(self, backbone: nn.Module, attention_type: str = 'cbam',
                 feature_channels: int = 1024):
        """
        Args:
            backbone: Base CNN model (e.g. DenseNet121)
            attention_type: 'cbam' or 'se'
            feature_channels: Number of channels in backbone's final feature map
        """
        super().__init__()
        self.features = backbone.features
        
        if attention_type == 'cbam':
            self.attention = CBAMBlock(feature_channels)
        elif attention_type == 'se':
            self.attention = SEBlock(feature_channels)
        else:
            self.attention = nn.Identity()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = backbone.classifier
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = F.relu(features, inplace=True)
        features = self.attention(features)
        out = self.pool(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def add_attention_to_model(model: nn.Module, attention_type: str = 'cbam',
                           model_name: str = 'densenet121') -> nn.Module:
    """
    Factory function to add attention mechanism to an existing model.
    
    Args:
        model: Base CNN model
        attention_type: 'cbam', 'se', or None
        model_name: Architecture name for determining feature channels
    
    Returns:
        Model wrapped with attention
    """
    if attention_type is None:
        return model
    
    # Determine feature channels based on architecture
    channel_map = {
        'densenet121': 1024,
        'resnet50': 2048,
        'efficientnet_b4': 1792,
    }
    feature_channels = channel_map.get(model_name, 1024)
    
    return AttentionWrapper(model, attention_type, feature_channels)


if __name__ == "__main__":
    print("Attention mechanisms module loaded successfully")
    print("  - SEBlock: Squeeze-and-Excitation")
    print("  - CBAMBlock: Convolutional Block Attention Module")
    print("  - AttentionWrapper: Wraps backbone with attention")
