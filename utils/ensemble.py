"""
Ensemble model module.
Combines multiple CNN architectures (DenseNet121, EfficientNet-B4, ResNet50)
for improved prediction via voting/averaging strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path


def build_single_model(architecture: str, num_classes: int = 2,
                       pretrained: bool = True, dropout_rate: float = 0.3) -> nn.Module:
    """
    Build a single model by architecture name.
    
    Args:
        architecture: One of 'densenet121', 'efficientnet_b4', 'resnet50'
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout_rate: Dropout rate for classifier head
    
    Returns:
        PyTorch model
    """
    if architecture == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        # Freeze early layers, unfreeze denseblock3, transition3, denseblock4, norm5
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.features.denseblock3.parameters():
            param.requires_grad = True
        for param in model.features.transition3.parameters():
            param.requires_grad = True
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        if hasattr(model.features, 'norm5'):
            for param in model.features.norm5.parameters():
                param.requires_grad = True
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )
        
    elif architecture == 'efficientnet_b4':
        try:
            model = models.efficientnet_b4(pretrained=pretrained)
        except Exception:
            model = models.efficientnet_b4(weights='DEFAULT' if pretrained else None)
        # Freeze early layers
        for param in model.features[:-2].parameters():
            param.requires_grad = False
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    elif architecture == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        # Freeze early layers
        for param in list(model.parameters())[:-20]:
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{architecture}] Total: {total:,} | Trainable: {trainable:,}")
    
    return model


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple CNN models with configurable aggregation strategy.
    
    Strategies:
        - soft_voting: Average class probabilities
        - hard_voting: Majority vote on predicted classes
        - weighted_average: Weighted average of probabilities
    """
    
    def __init__(self, models_list: List[nn.Module],
                 strategy: str = 'soft_voting',
                 weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models_list)
        self.strategy = strategy
        
        if weights is None:
            self.weights = [1.0 / len(models_list)] * len(models_list)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for model in self.models:
            out = model(x)
            outputs.append(out)
        
        if self.strategy == 'soft_voting':
            probs = [F.softmax(out, dim=1) for out in outputs]
            avg_probs = torch.stack(probs).mean(dim=0)
            return avg_probs
        
        elif self.strategy == 'hard_voting':
            preds = [out.argmax(dim=1) for out in outputs]
            stacked = torch.stack(preds, dim=0)  # (num_models, batch)
            # mode returns (values, indices)
            majority, _ = stacked.mode(dim=0)
            # Return one-hot-like output
            num_classes = outputs[0].size(1)
            result = torch.zeros(x.size(0), num_classes, device=x.device)
            result.scatter_(1, majority.unsqueeze(1), 1.0)
            return result
        
        elif self.strategy == 'weighted_average':
            probs = [F.softmax(out, dim=1) for out in outputs]
            weighted = torch.zeros_like(probs[0])
            for prob, w in zip(probs, self.weights):
                weighted += w * prob
            return weighted
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def predict_with_details(self, x: torch.Tensor) -> Dict:
        """Get detailed predictions from each sub-model."""
        details = {}
        all_probs = []
        
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                out = model(x)
                prob = F.softmax(out, dim=1)
                all_probs.append(prob)
                details[f'model_{i}'] = {
                    'logits': out.cpu().numpy(),
                    'probabilities': prob.cpu().numpy(),
                    'prediction': out.argmax(dim=1).cpu().numpy()
                }
        
        # Ensemble result
        ensemble_prob = self.forward(x)
        details['ensemble'] = {
            'probabilities': ensemble_prob.cpu().detach().numpy(),
            'prediction': ensemble_prob.argmax(dim=1).cpu().detach().numpy()
        }
        
        return details


def build_ensemble(architectures: List[str] = None,
                   num_classes: int = 2,
                   strategy: str = 'soft_voting',
                   weights: Optional[List[float]] = None,
                   pretrained: bool = True) -> EnsembleModel:
    """
    Build an ensemble model from multiple architectures.
    
    Args:
        architectures: List of architecture names
        num_classes: Number of output classes
        strategy: Aggregation strategy
        weights: Per-model weights
        pretrained: Use pretrained weights
    
    Returns:
        EnsembleModel instance
    """
    if architectures is None:
        architectures = ['densenet121', 'efficientnet_b4', 'resnet50']
    
    print(f"Building ensemble: {architectures}")
    print(f"Strategy: {strategy}")
    
    models_list = []
    for arch in architectures:
        model = build_single_model(arch, num_classes, pretrained)
        models_list.append(model)
    
    ensemble = EnsembleModel(models_list, strategy, weights)
    
    total_params = sum(p.numel() for p in ensemble.parameters())
    print(f"\nEnsemble total parameters: {total_params:,}")
    
    return ensemble


def load_ensemble_weights(ensemble: EnsembleModel,
                          weight_paths: List[str],
                          device: torch.device) -> EnsembleModel:
    """Load pre-trained weights for each model in the ensemble."""
    for i, (model, path) in enumerate(zip(ensemble.models, weight_paths)):
        if Path(path).exists():
            checkpoint = torch.load(path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded weights for model {i} from {path}")
        else:
            print(f"Warning: weights not found at {path} for model {i}")
    
    return ensemble.to(device)


if __name__ == "__main__":
    print("Ensemble module loaded successfully")
    print("Supported architectures: densenet121, efficientnet_b4, resnet50")
    print("Strategies: soft_voting, hard_voting, weighted_average")
