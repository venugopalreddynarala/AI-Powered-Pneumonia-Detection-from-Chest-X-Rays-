"""
Federated Learning module.
Implements Federated Averaging (FedAvg) and FedProx for privacy-preserving
distributed model training across multiple institutions.
Supports optional differential privacy.
"""

import os
import copy
import json
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Callable
from pathlib import Path


class FederatedServer:
    """
    Central federated learning server.
    Aggregates model updates from multiple clients without accessing raw data.
    """
    
    def __init__(self, global_model: nn.Module, 
                 aggregation: str = 'fedavg',
                 num_rounds: int = 10):
        """
        Args:
            global_model: The global model to be trained
            aggregation: 'fedavg' or 'fedprox'
            num_rounds: Number of federated rounds
        """
        self.global_model = global_model
        self.aggregation = aggregation
        self.num_rounds = num_rounds
        self.history = []
    
    def aggregate(self, client_models: List[nn.Module],
                  client_weights: Optional[List[float]] = None) -> nn.Module:
        """
        Aggregate client model updates using FedAvg.
        
        Args:
            client_models: List of trained client models
            client_weights: Data-proportional weights per client
        
        Returns:
            Updated global model
        """
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        else:
            total = sum(client_weights)
            client_weights = [w / total for w in client_weights]
        
        # Average model parameters
        global_dict = self.global_model.state_dict()
        
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
            for client_model, weight in zip(client_models, client_weights):
                client_dict = client_model.state_dict()
                global_dict[key] += weight * client_dict[key].float()
        
        self.global_model.load_state_dict(global_dict)
        return self.global_model
    
    def get_global_model(self) -> nn.Module:
        """Return a copy of the current global model."""
        return copy.deepcopy(self.global_model)
    
    def run_federation(self, clients: list,
                       val_loader=None,
                       device: torch.device = None) -> Dict:
        """
        Run the full federated learning process.
        
        Args:
            clients: List of FederatedClient instances
            val_loader: Optional validation data loader (for server-side eval)
            device: Compute device
        
        Returns:
            Training history
        """
        device = device or torch.device('cpu')
        
        print(f"\n{'='*70}")
        print(f"FEDERATED LEARNING - {self.aggregation.upper()}")
        print(f"Clients: {len(clients)} | Rounds: {self.num_rounds}")
        print(f"{'='*70}")
        
        for round_idx in range(self.num_rounds):
            print(f"\n--- Round {round_idx + 1}/{self.num_rounds} ---")
            
            # Distribute global model to clients
            global_model = self.get_global_model()
            
            # Client training
            client_models = []
            client_weights_list = []
            round_losses = []
            
            for i, client in enumerate(clients):
                print(f"  Client {i+1} training...")
                trained_model, metrics = client.local_train(
                    global_model, device
                )
                client_models.append(trained_model)
                client_weights_list.append(client.get_data_size())
                round_losses.append(metrics.get('loss', 0))
            
            # Aggregate
            self.aggregate(client_models, client_weights_list)
            
            # Evaluate on validation set
            round_result = {
                'round': round_idx + 1,
                'avg_client_loss': float(np.mean(round_losses)),
            }
            
            if val_loader:
                val_acc = self._evaluate(val_loader, device)
                round_result['val_accuracy'] = val_acc
                print(f"  Global Val Accuracy: {val_acc:.4f}")
            
            self.history.append(round_result)
            print(f"  Avg Client Loss: {round_result['avg_client_loss']:.4f}")
        
        print(f"\n{'='*70}")
        print("FEDERATED LEARNING COMPLETE")
        print(f"{'='*70}")
        
        return {
            'num_rounds': self.num_rounds,
            'num_clients': len(clients),
            'history': self.history,
        }
    
    def _evaluate(self, val_loader, device) -> float:
        """Evaluate global model on validation set."""
        self.global_model.eval()
        self.global_model.to(device)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.global_model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return correct / total if total > 0 else 0.0


class FederatedClient:
    """
    Federated learning client (represents one institution/hospital).
    Trains locally on private data and sends model updates to the server.
    """
    
    def __init__(self, train_loader, client_id: str = 'client_0',
                 local_epochs: int = 3, learning_rate: float = 0.001,
                 mu: float = 0.01, use_dp: bool = False,
                 noise_multiplier: float = 1.0, max_grad_norm: float = 1.0):
        """
        Args:
            train_loader: Local training data loader
            client_id: Client identifier
            local_epochs: Number of local training epochs
            learning_rate: Local learning rate
            mu: FedProx proximal term weight
            use_dp: Enable differential privacy
            noise_multiplier: DP noise multiplier
            max_grad_norm: DP max gradient norm
        """
        self.train_loader = train_loader
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.mu = mu
        self.use_dp = use_dp
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
    
    def get_data_size(self) -> int:
        """Return the size of local dataset."""
        return len(self.train_loader.dataset)
    
    def local_train(self, global_model: nn.Module,
                    device: torch.device) -> tuple:
        """
        Train locally using the client's private data.
        
        Args:
            global_model: Current global model
            device: Compute device
        
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        model = copy.deepcopy(global_model).to(device)
        model.train()
        
        # Keep reference to global params for FedProx
        global_params = {name: param.clone().detach() 
                        for name, param in global_model.named_parameters()}
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.learning_rate
        )
        
        total_loss = 0
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # FedProx proximal term
                if self.mu > 0:
                    proximal_term = 0
                    for name, param in model.named_parameters():
                        if name in global_params and param.requires_grad:
                            proximal_term += ((param - global_params[name]) ** 2).sum()
                    loss += (self.mu / 2) * proximal_term
                
                loss.backward()
                
                # Differential privacy: clip gradients and add noise
                if self.use_dp:
                    self._apply_dp(model, optimizer)
                
                optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        return model, {'loss': avg_loss, 'samples': total_samples}
    
    def _apply_dp(self, model: nn.Module, optimizer):
        """Apply differential privacy (gradient clipping + noise)."""
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_grad_norm
        )
        
        # Add Gaussian noise
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * (
                    self.noise_multiplier * self.max_grad_norm
                )
                param.grad += noise


def create_federated_splits(dataset, num_clients: int = 3,
                            iid: bool = True) -> List:
    """
    Split a dataset into federated client subsets.
    
    Args:
        dataset: Full training dataset
        num_clients: Number of clients
        iid: If True, random IID split; if False, non-IID (sorted by label)
    
    Returns:
        List of Subset objects, one per client
    """
    from torch.utils.data import Subset
    
    n = len(dataset)
    
    if iid:
        indices = np.random.permutation(n)
    else:
        # Non-IID: sort by label
        labels = []
        for i in range(n):
            _, label = dataset[i]
            labels.append(label)
        indices = np.array(labels).argsort()
    
    # Split indices evenly
    splits = np.array_split(indices, num_clients)
    
    subsets = [Subset(dataset, split.tolist()) for split in splits]
    
    for i, subset in enumerate(subsets):
        print(f"  Client {i}: {len(subset)} samples")
    
    return subsets


if __name__ == "__main__":
    print("Federated learning module loaded successfully")
    print("Features: FedAvg, FedProx, Differential Privacy, IID/Non-IID splits")
