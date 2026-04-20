"""
K-Fold Cross-Validation module.
Provides stratified k-fold cross-validation for robust model evaluation.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Callable
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_dataset_labels(dataset) -> np.ndarray:
    """Extract all labels from a dataset."""
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    return np.array(labels)


def stratified_k_fold_split(dataset, k: int = 5, 
                            random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate stratified k-fold indices.
    
    Args:
        dataset: PyTorch dataset with (data, label) items
        k: Number of folds
        random_state: Random seed
    
    Returns:
        List of (train_indices, val_indices) tuples
    """
    labels = get_dataset_labels(dataset)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    
    folds = []
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        folds.append((train_idx, val_idx))
    
    return folds


def cross_validate(model_builder: Callable, dataset, k: int = 5,
                   epochs: int = 10, batch_size: int = 32,
                   learning_rate: float = 0.001, device: torch.device = None,
                   save_dir: str = 'models/cv_folds') -> Dict:
    """
    Perform k-fold cross-validation.
    
    Args:
        model_builder: Function that returns a fresh model instance
        dataset: Full training dataset
        k: Number of folds
        epochs: Training epochs per fold
        batch_size: Batch size
        learning_rate: Learning rate
        device: Compute device
        save_dir: Directory to save fold models
    
    Returns:
        Dictionary with per-fold and aggregated results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(save_dir, exist_ok=True)
    
    folds = stratified_k_fold_split(dataset, k)
    
    fold_results = []
    all_val_accs = []
    all_val_losses = []
    
    print(f"\n{'='*70}")
    print(f"STARTING {k}-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{k} ---")
        print(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        # Create data subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, 
                                  shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, 
                                shuffle=False, num_workers=2, pin_memory=True)
        
        # Build fresh model
        model = model_builder().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=learning_rate)
        
        best_val_acc = 0.0
        fold_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(train_loader, desc=f'Fold {fold_idx+1} Epoch {epoch}',
                                       leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_loss = running_loss / total
            train_acc = correct / total
            
            # Validation
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss_sum += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss = val_loss_sum / val_total
            val_acc = val_correct / val_total
            
            fold_history['train_loss'].append(train_loss)
            fold_history['train_acc'].append(train_acc)
            fold_history['val_loss'].append(val_loss)
            fold_history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'fold': fold_idx,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, os.path.join(save_dir, f'fold_{fold_idx}_best.pth'))
        
        print(f"  Fold {fold_idx + 1} Best Val Acc: {best_val_acc:.4f}")
        
        all_val_accs.append(best_val_acc)
        all_val_losses.append(min(fold_history['val_loss']))
        fold_results.append({
            'fold': fold_idx,
            'best_val_acc': best_val_acc,
            'best_val_loss': min(fold_history['val_loss']),
            'history': fold_history
        })
    
    # Aggregate results
    results = {
        'k': k,
        'fold_results': fold_results,
        'mean_val_acc': float(np.mean(all_val_accs)),
        'std_val_acc': float(np.std(all_val_accs)),
        'mean_val_loss': float(np.mean(all_val_losses)),
        'std_val_loss': float(np.std(all_val_losses)),
        'all_val_accs': [float(a) for a in all_val_accs],
    }
    
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"  Mean Val Accuracy: {results['mean_val_acc']:.4f} ± {results['std_val_acc']:.4f}")
    print(f"  Mean Val Loss:     {results['mean_val_loss']:.4f} ± {results['std_val_loss']:.4f}")
    print(f"  Per-fold accuracies: {[f'{a:.4f}' for a in all_val_accs]}")
    
    # Save results
    results_path = os.path.join(save_dir, 'cv_results.json')
    serializable = {k: v for k, v in results.items() if k != 'fold_results'}
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    return results


def plot_cv_results(results: Dict, save_path: str = None):
    """Plot cross-validation results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of per-fold accuracy
    folds = range(1, results['k'] + 1)
    accs = results['all_val_accs']
    
    axes[0].bar(folds, accs, color='steelblue', edgecolor='black')
    axes[0].axhline(y=results['mean_val_acc'], color='red', linestyle='--',
                     label=f"Mean: {results['mean_val_acc']:.4f}")
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Validation Accuracy')
    axes[0].set_title(f'{results["k"]}-Fold Cross-Validation Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    axes[1].boxplot(accs, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightblue'))
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('Accuracy Distribution Across Folds')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Cross-validation module loaded successfully")
    print("Features: Stratified K-Fold, per-fold training, result aggregation")
