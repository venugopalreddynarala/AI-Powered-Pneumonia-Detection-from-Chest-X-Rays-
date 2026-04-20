"""
Uncertainty Quantification module.
Implements Monte Carlo Dropout, Temperature Scaling, and confidence calibration
for reliable prediction confidence estimates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.
    Runs multiple forward passes with dropout enabled to estimate
    prediction uncertainty.
    """
    
    def __init__(self, model: nn.Module, num_iterations: int = 30):
        """
        Args:
            model: Trained model with dropout layers
            num_iterations: Number of stochastic forward passes
        """
        super().__init__()
        self.model = model
        self.num_iterations = num_iterations
    
    def _enable_dropout(self):
        """Enable dropout layers during inference for MC sampling."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def predict_with_uncertainty(self, input_tensor: torch.Tensor) -> Dict:
        """
        Run multiple forward passes with dropout to estimate uncertainty.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
        
        Returns:
            Dictionary with prediction, mean probability, uncertainty metrics
        """
        self.model.eval()
        self._enable_dropout()
        
        all_probs = []
        
        with torch.no_grad():
            for _ in range(self.num_iterations):
                output = self.model(input_tensor)
                prob = F.softmax(output, dim=1)
                all_probs.append(prob.cpu().numpy())
        
        all_probs = np.array(all_probs)  # (num_iter, batch, num_classes)
        
        # Mean prediction
        mean_probs = all_probs.mean(axis=0)  # (batch, num_classes)
        
        # Predictive entropy (total uncertainty)
        predictive_entropy = -np.sum(
            mean_probs * np.log(mean_probs + 1e-10), axis=-1
        )
        
        # Expected entropy (aleatoric uncertainty)
        expected_entropy = -np.mean(
            np.sum(all_probs * np.log(all_probs + 1e-10), axis=-1),
            axis=0
        )
        
        # Mutual information (epistemic uncertainty)
        mutual_info = predictive_entropy - expected_entropy
        
        # Standard deviation across iterations
        std_probs = all_probs.std(axis=0)
        
        # Prediction
        prediction = mean_probs.argmax(axis=-1)
        confidence = mean_probs.max(axis=-1)
        
        return {
            'prediction': prediction,
            'mean_probabilities': mean_probs,
            'confidence': confidence,
            'std': std_probs,
            'predictive_entropy': predictive_entropy,
            'aleatoric_uncertainty': expected_entropy,
            'epistemic_uncertainty': mutual_info,
            'all_probabilities': all_probs,
            'is_uncertain': bool(mutual_info[0] > 0.1) if len(mutual_info) > 0 else False,
        }


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for calibrating model confidence.
    Learns a single temperature parameter that scales logits
    to produce better-calibrated probabilities.
    
    Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
    """
    
    def __init__(self, model: nn.Module, temperature: float = 1.5):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.tensor(temperature))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits / self.temperature
    
    def calibrate(self, val_loader, device: torch.device,
                  max_iter: int = 50, lr: float = 0.01):
        """
        Learn optimal temperature on validation set using NLL loss.
        
        Args:
            val_loader: Validation data loader
            device: Compute device
            max_iter: Maximum optimization iterations
            lr: Learning rate for temperature optimization
        """
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        # Collect all logits and labels
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                logits = self.model(inputs)
                all_logits.append(logits.cpu())
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        def eval_fn():
            optimizer.zero_grad()
            scaled_logits = all_logits / self.temperature
            loss = nll_criterion(scaled_logits, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_fn)
        
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self.temperature.item()


def compute_calibration_metrics(probabilities: np.ndarray,
                                labels: np.ndarray,
                                n_bins: int = 10) -> Dict:
    """
    Compute Expected Calibration Error (ECE) and reliability diagram data.
    
    Args:
        probabilities: Predicted probabilities for positive class
        labels: Ground truth binary labels
        n_bins: Number of bins for calibration
    
    Returns:
        Dictionary with ECE, bin accuracies, bin confidences
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (probabilities >= bin_boundaries[i]) & (probabilities < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = labels[mask].mean()
            bin_conf = probabilities[mask].mean()
            bin_accs.append(float(bin_acc))
            bin_confs.append(float(bin_conf))
            bin_counts.append(int(mask.sum()))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)
    
    # ECE: weighted average of |accuracy - confidence| per bin
    total_samples = sum(bin_counts)
    ece = sum(
        (count / total_samples) * abs(acc - conf)
        for acc, conf, count in zip(bin_accs, bin_confs, bin_counts)
        if count > 0
    )
    
    return {
        'ece': float(ece),
        'bin_accuracies': bin_accs,
        'bin_confidences': bin_confs,
        'bin_counts': bin_counts,
        'bin_boundaries': bin_boundaries.tolist(),
    }


def predict_with_uncertainty(model: nn.Module, input_tensor: torch.Tensor,
                             method: str = 'mc_dropout',
                             num_iterations: int = 30,
                             temperature: float = 1.5) -> Dict:
    """
    High-level function for uncertainty-aware prediction.
    
    Args:
        model: Trained model
        input_tensor: Input tensor
        method: 'mc_dropout' or 'temperature_scaling'
        num_iterations: MC Dropout iterations
        temperature: Temperature scaling parameter
    
    Returns:
        Prediction dictionary with uncertainty metrics
    """
    if method == 'mc_dropout':
        mc = MCDropout(model, num_iterations)
        return mc.predict_with_uncertainty(input_tensor)
    
    elif method == 'temperature_scaling':
        ts = TemperatureScaling(model, temperature)
        ts.eval()
        with torch.no_grad():
            output = ts(input_tensor)
            prob = F.softmax(output, dim=1).cpu().numpy()
        
        return {
            'prediction': prob.argmax(axis=-1),
            'mean_probabilities': prob,
            'confidence': prob.max(axis=-1),
            'temperature': temperature,
            'is_uncertain': bool(prob.max() < 0.6),
        }
    
    else:
        # Fallback: standard prediction
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            prob = F.softmax(output, dim=1).cpu().numpy()
        
        return {
            'prediction': prob.argmax(axis=-1),
            'mean_probabilities': prob,
            'confidence': prob.max(axis=-1),
            'is_uncertain': False,
        }


if __name__ == "__main__":
    print("Uncertainty quantification module loaded successfully")
    print("Methods: MC Dropout, Temperature Scaling, Calibration Metrics")
