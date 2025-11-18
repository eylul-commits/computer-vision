import os
from pathlib import Path
from typing import Tuple, List, Optional, Callable
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class FireDetectionDataset(Dataset):
    #Fire dataset with YOLO format annotations.
    #Converts detection format to classification by checking if fire/smoke is present.
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        mode: str = "binary"  # 'binary' (fire/no-fire) or 'multiclass' (smoke/fire/none)
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.mode = mode
        
        self.images_dir = self.root / split / "images"
        self.labels_dir = self.root / split / "labels"
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        
        # Class mapping: 0=smoke, 1=fire (from YOLO labels)
        self.class_names = ["smoke", "fire"]
        
        print(f"Loaded {len(self.image_files)} images from {split} split")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Load corresponding label file
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        
        # Parse YOLO labels to determine class
        label = self._parse_yolo_label(label_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _parse_yolo_label(self, label_path: Path) -> int:
        #Parse YOLO label file and determine classification label.
        #YOLO format: <class_id> <x_center> <y_center> <width> <height>
        if not label_path.exists():
            return 0  # No fire/smoke
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) == 0:
                return 0  # No fire/smoke
            
            # Get classes present in the image
            classes_present = set()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    classes_present.add(class_id)
            
            if self.mode == "binary":
                # Binary: 0=no fire/smoke, 1=fire/smoke present
                return 1 if len(classes_present) > 0 else 0
            else:
                # Multiclass: 0=none, 1=smoke, 2=fire, 3=both
                if 1 in classes_present and 0 in classes_present:
                    return 3  # Both
                elif 1 in classes_present:
                    return 2  # Fire only
                elif 0 in classes_present:
                    return 1  # Smoke only
                else:
                    return 0  # None
        except Exception as e:
            print(f"Error parsing {label_path}: {e}")
            return 0


class SmokeClassificationDataset(Dataset):
    #Smoke classification dataset with folder structure.
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        self.data_dir = self.root / split
        self.class_names = ["cloud", "other", "smoke"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Collect all image files
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} images from {split} split")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _get_class_distribution(self):
        dist = {cls: 0 for cls in self.class_names}
        for _, label in self.samples:
            dist[self.class_names[label]] += 1
        return dist
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


