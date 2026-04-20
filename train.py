"""
Training pipeline for pneumonia detection.
Supports DenseNet121, EfficientNet-B4, ResNet50, ensembles,
attention mechanisms, cross-validation, and class imbalance handling.
"""

import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import numpy as np

from utils.data_prep import create_dataloaders


def build_model(num_classes: int = 2, pretrained: bool = True,
                architecture: str = 'densenet121', 
                use_attention: bool = False) -> nn.Module:
    """
    Build a classification model with optional attention mechanism.
    
    Args:
        num_classes: Number of output classes (2=binary, 3=multi-class)
        pretrained: Use ImageNet pretrained weights
        architecture: 'densenet121', 'efficientnet_b4', 'resnet50'
        use_attention: Add CBAM attention mechanism
        
    Returns:
        Modified model
    """
    try:
        from utils.ensemble import build_single_model
        model = build_single_model(architecture, num_classes, pretrained)
    except ImportError:
        # Fallback to original DenseNet121
        model = models.densenet121(pretrained=pretrained)
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    # Add attention mechanism if requested
    if use_attention:
        try:
            from utils.attention import add_attention_to_model
            model = add_attention_to_model(model, architecture)
            print(f"CBAM attention added to {architecture}")
        except ImportError:
            print("Attention module not found, skipping attention")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model built: {architecture} | Classes: {num_classes}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    return model


def compute_class_weights(data_dir: str, device: torch.device) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequency.
    Addresses class imbalance in the dataset.
    """
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        train_dir = os.path.join(data_dir, 'chest_xray', 'train')
    
    dataset = ImageFolder(train_dir, transform=transforms.ToTensor())
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets)
    total = sum(class_counts)
    weights = total / (len(class_counts) * class_counts)
    weights = torch.FloatTensor(weights).to(device)
    
    print(f"Class distribution: {dict(zip(dataset.classes, class_counts.tolist()))}")
    print(f"Class weights: {weights.cpu().numpy()}")
    
    return weights


def train_epoch(model: nn.Module, 
                dataloader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int) -> tuple:
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
            criterion: nn.Module,
            device: torch.device,
            epoch: int) -> tuple:
    """
    Validate model on validation set.
    
    Args:
        model: Neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def plot_training_history(history: dict, save_path: str = 'training_history.png'):
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")


def train_pipeline(data_dir: str = 'data/chest_xray',
                  epochs: int = 20,
                  batch_size: int = 32,
                  learning_rate: float = 0.001,
                  save_dir: str = 'models',
                  architecture: str = 'densenet121',
                  num_classes: int = 2,
                  use_attention: bool = False,
                  use_class_weights: bool = True,
                  use_ensemble: bool = False,
                  ensemble_models: list = None,
                  use_cross_validation: bool = False,
                  cv_folds: int = 5,
                  scheduler_type: str = 'plateau',
                  model_version: str = None,
                  resume_path: str = None):
    """
    Complete training pipeline with advanced features.
    
    Args:
        data_dir: Directory containing the dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        save_dir: Directory to save model weights
        architecture: Model architecture ('densenet121', 'efficientnet_b4', 'resnet50')
        num_classes: Number of classes (2=binary, 3=multi-class)
        use_attention: Add CBAM attention to the model
        use_class_weights: Use weighted loss for class imbalance
        use_ensemble: Train an ensemble of multiple architectures
        ensemble_models: List of architectures for ensemble
        use_cross_validation: Enable k-fold cross-validation
        cv_folds: Number of CV folds
        scheduler_type: 'plateau' or 'cosine'
        model_version: Version tag for model tracking
        resume_path: Path to checkpoint to resume training from
    """
    print("="*70)
    print("PNEUMONIA DETECTION MODEL TRAINING")
    print(f"  Architecture: {architecture} | Classes: {num_classes}")
    print(f"  Attention: {use_attention} | Ensemble: {use_ensemble}")
    print(f"  Class Weights: {use_class_weights} | CV: {use_cross_validation}")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Cross-validation mode
    if use_cross_validation:
        print(f"\n--- K-Fold Cross-Validation (k={cv_folds}) ---")
        try:
            from utils.cross_validation import cross_validate
            cv_results = cross_validate(
                data_dir=data_dir,
                build_model_fn=lambda: build_model(num_classes, True, architecture, use_attention),
                k_folds=cv_folds,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                save_dir=save_dir
            )
            return None, cv_results
        except ImportError:
            print("Cross-validation module not found, proceeding with standard training")
    
    # Load data
    print("\nLoading datasets...")
    dataloaders = create_dataloaders(data_dir, batch_size=batch_size)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Build model (single or ensemble)
    print("\nBuilding model...")
    if use_ensemble:
        try:
            from utils.ensemble import build_ensemble
            if ensemble_models is None:
                ensemble_models = ['densenet121', 'efficientnet_b4', 'resnet50']
            model = build_ensemble(ensemble_models, num_classes)
            print(f"Ensemble model with {len(ensemble_models)} architectures")
        except ImportError:
            print("Ensemble module not found, using single model")
            model = build_model(num_classes, True, architecture, use_attention)
    else:
        model = build_model(num_classes, True, architecture, use_attention)
    
    model = model.to(device)
    
    # Loss function with class weights
    if use_class_weights:
        try:
            class_weights = compute_class_weights(data_dir, device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
            print("Using weighted CrossEntropyLoss with label_smoothing=0.1")
        except Exception as e:
            print(f"Could not compute class weights: {e}")
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-5)
    
    # Resume from checkpoint
    start_epoch = 1
    best_val_acc = 0.0
    best_epoch = 0
    resumed_history = None
    
    if resume_path and os.path.exists(resume_path):
        print(f"\n--- Resuming from checkpoint: {resume_path} ---")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Reset optimizer with fresh state for fine-tuning (avoids stale momentum)
        trainable_params_list = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params_list, lr=learning_rate, weight_decay=1e-5)
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['val_acc']
        best_epoch = checkpoint['epoch']
        print(f"  Resumed from epoch {checkpoint['epoch']}")
        print(f"  Best val acc so far: {best_val_acc:.4f}")
        print(f"  Fine-tuning LR: {learning_rate}")
        print(f"  Continuing from epoch {start_epoch} to {epochs}")
        
        # Load previous history if available
        history_path = os.path.join(save_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                resumed_history = json.load(f)
            print(f"  Loaded previous training history ({len(resumed_history.get('train_loss',[]))} epochs)")
    
    # Scheduler (configured for remaining epochs)
    remaining_epochs = epochs - start_epoch + 1
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=max(remaining_epochs, 1), eta_min=1e-6)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training history
    if resumed_history:
        history = resumed_history
        if 'learning_rates' not in history:
            history['learning_rates'] = []
    else:
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
        }
    
    patience_counter = 0
    early_stop_patience = 7
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch}/{epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_model_path = os.path.join(save_dir, 'model_weights.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'architecture': architecture,
                'num_classes': num_classes,
                'use_attention': use_attention,
                'use_ensemble': use_ensemble,
                'model_version': model_version or f'v{epoch}',
            }, best_model_path)
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n  Early stopping triggered (no improvement for {early_stop_patience} epochs)")
            break
        
        print("-"*70)
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"Model saved to: {os.path.join(save_dir, 'model_weights.pth')}")
    
    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to: {history_path}")
    
    # Plot training curves
    plot_path = os.path.join(save_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Register model version in database
    try:
        from database import get_database
        db = get_database()
        version = model_version or f'v1.0_{architecture}'
        db.add_model_version(
            version=version,
            architecture=architecture,
            accuracy=best_val_acc,
            training_config={
                'epochs': epochs, 'batch_size': batch_size,
                'learning_rate': learning_rate, 'use_attention': use_attention,
                'use_class_weights': use_class_weights,
            },
            weights_path=os.path.join(save_dir, 'model_weights.pth')
        )
        db.set_active_model(version)
        print(f"Model version '{version}' registered in database")
    except Exception as e:
        print(f"Note: Could not register model in database: {e}")
    
    # Export to ONNX if requested
    try:
        from config import get_config
        config = get_config()
        if config.get('onnx', {}).get('auto_export', False):
            from utils.model_export import export_to_onnx
            export_to_onnx(model, output_path=config['onnx']['export_path'])
    except Exception:
        pass
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train pneumonia detection model')
    parser.add_argument('--data_dir', type=str, default='data/chest_xray',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save model weights')
    parser.add_argument('--architecture', type=str, default='densenet121',
                       choices=['densenet121', 'efficientnet_b4', 'resnet50'],
                       help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes (2 or 3)')
    parser.add_argument('--attention', action='store_true',
                       help='Enable CBAM attention mechanism')
    parser.add_argument('--class_weights', action='store_true', default=True,
                       help='Use class weights for imbalanced data')
    parser.add_argument('--ensemble', action='store_true',
                       help='Train ensemble of multiple architectures')
    parser.add_argument('--cross_val', action='store_true',
                       help='Use k-fold cross-validation')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine'],
                       help='Learning rate scheduler')
    parser.add_argument('--version', type=str, default=None,
                       help='Model version tag')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    train_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        architecture=args.architecture,
        num_classes=args.num_classes,
        use_attention=args.attention,
        use_class_weights=args.class_weights,
        use_ensemble=args.ensemble,
        use_cross_validation=args.cross_val,
        cv_folds=args.cv_folds,
        scheduler_type=args.scheduler,
        model_version=args.version,
        resume_path=args.resume,
    )
