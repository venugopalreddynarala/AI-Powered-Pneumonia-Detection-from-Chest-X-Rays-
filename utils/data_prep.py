"""
Data preparation utilities for chest X-ray pneumonia detection.
Handles dataset download, preprocessing, augmentation, and DataLoader creation.
"""

import os
import zipfile
import shutil
from pathlib import Path
from typing import Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class XRayDataset(Dataset):
    """Custom PyTorch Dataset for chest X-ray images."""
    
    def __init__(self, root_dir: str, transform=None, mode: str = 'train'):
        """
        Args:
            root_dir: Root directory containing the dataset
            transform: Torchvision transforms to apply
            mode: One of 'train', 'val', 'test'
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        
        # Build file paths and labels
        self.samples = []
        self.labels = []
        
        # Dataset structure: chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}
        mode_dir = self.root_dir / mode.upper() if mode == 'test' else self.root_dir / mode
        
        if not mode_dir.exists():
            # Try alternate naming
            mode_dir = self.root_dir / mode
        
        # Normal images (label 0)
        normal_dir = mode_dir / 'NORMAL'
        if normal_dir.exists():
            for img_file in normal_dir.glob('*.jpeg'):
                self.samples.append(str(img_file))
                self.labels.append(0)
            for img_file in normal_dir.glob('*.jpg'):
                self.samples.append(str(img_file))
                self.labels.append(0)
        
        # Pneumonia images (label 1)
        pneumonia_dir = mode_dir / 'PNEUMONIA'
        if pneumonia_dir.exists():
            for img_file in pneumonia_dir.glob('*.jpeg'):
                self.samples.append(str(img_file))
                self.labels.append(1)
            for img_file in pneumonia_dir.glob('*.jpg'):
                self.samples.append(str(img_file))
                self.labels.append(1)
        
        print(f"Loaded {len(self.samples)} images for {mode} mode")
        print(f"  Normal: {self.labels.count(0)}, Pneumonia: {self.labels.count(1)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def download_kaggle_dataset(dataset_name: str = "paultimothymooney/chest-xray-pneumonia", 
                           dest_dir: str = "data") -> bool:
    """
    Download dataset from Kaggle using Kaggle API.
    
    Args:
        dataset_name: Kaggle dataset identifier
        dest_dir: Destination directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import kaggle
        
        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {dataset_name} from Kaggle...")
        kaggle.api.dataset_download_files(
            dataset_name, 
            path=str(dest_path), 
            unzip=True
        )
        print("Download complete!")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\n" + "="*70)
        print("DATASET DOWNLOAD FAILED - KAGGLE API SETUP REQUIRED")
        print("="*70)
        print("\n🔧 SOLUTION: Choose one of these options:")
        print("\n1. EASIEST - Run the interactive setup script:")
        print("   python setup_kaggle.py")
        print("\n2. Manual Kaggle API Setup:")
        print("   a. Go to: https://www.kaggle.com/account")
        print("   b. Scroll to 'API' section")
        print("   c. Click 'Create New API Token'")
        print("   d. Save kaggle.json to:")
        print(f"      {Path.home() / '.kaggle' / 'kaggle.json'}")
        print("   e. Re-run this script")
        print("\n3. Manual Dataset Download:")
        print("   a. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("   b. Click the 'Download' button (requires Kaggle login)")
        print("   c. Extract the downloaded ZIP file")
        print("   d. Place the 'chest_xray' folder here:")
        print(f"      {Path(dest_dir).absolute() / 'chest_xray'}")
        print("   e. Folder structure should be:")
        print("      data/chest_xray/train/NORMAL/")
        print("      data/chest_xray/train/PNEUMONIA/")
        print("      data/chest_xray/test/NORMAL/")
        print("      data/chest_xray/test/PNEUMONIA/")
        print("      data/chest_xray/val/NORMAL/")
        print("      data/chest_xray/val/PNEUMONIA/")
        print("="*70)
        return False


def get_transforms(mode: str = 'train') -> transforms.Compose:
    """
    Get image transforms for different modes.
    
    Args:
        mode: One of 'train', 'val', 'test'
        
    Returns:
        Composed transforms
    """
    # ImageNet statistics for pretrained models
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def create_dataloaders(data_dir: str = "data/chest_xray", 
                      batch_size: int = 32,
                      num_workers: int = 4) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        data_dir: Root directory of the dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        Dictionary with train, val, and test DataLoaders
    """
    data_path = Path(data_dir)
    
    # Check if dataset exists
    if not data_path.exists():
        print(f"Dataset not found at {data_dir}")
        print("Attempting to download from Kaggle...")
        download_kaggle_dataset(dest_dir="data")
    
    # Create datasets
    train_dataset = XRayDataset(
        root_dir=data_path,
        transform=get_transforms('train'),
        mode='train'
    )
    
    val_dataset = XRayDataset(
        root_dir=data_path,
        transform=get_transforms('val'),
        mode='val'
    )
    
    test_dataset = XRayDataset(
        root_dir=data_path,
        transform=get_transforms('test'),
        mode='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def preprocess_single_image(image_path: str) -> torch.Tensor:
    """
    Preprocess a single image for inference.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image tensor with batch dimension
    """
    transform = get_transforms('test')
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension


if __name__ == "__main__":
    # Test data preparation
    print("Testing data preparation...")
    
    # Try to create dataloaders
    try:
        loaders = create_dataloaders(batch_size=16)
        print("\n✓ DataLoaders created successfully!")
        
        # Test a batch
        images, labels = next(iter(loaders['train']))
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label distribution: {labels.sum().item()}/{len(labels)} pneumonia cases")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
