import yaml
from pathlib import Path
from typing import Tuple, Dict
import torch
from torch.utils.data import DataLoader, random_split

from .dataset_loader import FireDetectionDataset, SmokeClassificationDataset
from .transforms import get_transforms


class DataModule:
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        dataset_name: str = "smoke",  # 'fire' or 'smoke'
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224
    ):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.setup()
    
    def setup(self):
        #Setup datasets and dataloaders.
        
        # Get transforms
        train_transform = get_transforms(
            image_size=self.image_size,
            is_training=True,
            use_albumentations=True
        )
        val_transform = get_transforms(
            image_size=self.image_size,
            is_training=False,
            use_albumentations=True
        )
        
        if self.dataset_name == "fire":
            dataset_config = self.config['datasets']['fire']
            root = dataset_config['root']
            
            self.train_dataset = FireDetectionDataset(
                root=root,
                split="train",
                transform=train_transform,
                mode="binary"
            )
            self.val_dataset = FireDetectionDataset(
                root=root,
                split="val",
                transform=val_transform,
                mode="binary"
            )
            self.test_dataset = FireDetectionDataset(
                root=root,
                split="test",
                transform=val_transform,
                mode="binary"
            )
            self.num_classes = 2  # Binary: fire/no-fire
            self.class_names = ["no_fire", "fire"]
            
        elif self.dataset_name == "smoke":
            dataset_config = self.config['datasets']['smoke']
            root = dataset_config['root']
            
            self.train_dataset = SmokeClassificationDataset(
                root=root,
                split="train",
                transform=train_transform
            )
            self.val_dataset = SmokeClassificationDataset(
                root=root,
                split="val",
                transform=val_transform
            )
            self.test_dataset = SmokeClassificationDataset(
                root=root,
                split="test",
                transform=val_transform
            )
            self.num_classes = dataset_config['num_classes']
            self.class_names = dataset_config['classes']
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        print(f"\n{'='*60}")
        print(f"Dataset: {self.dataset_name.upper()}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"{'='*60}\n")
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, val, test dataloaders."""
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_dataset_info(self) -> Dict:
        """Get dataset information."""
        return {
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'train_size': len(self.train_dataset),
            'val_size': len(self.val_dataset),
            'test_size': len(self.test_dataset)
        }

