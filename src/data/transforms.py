import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image


def get_train_transforms(image_size: int = 224, use_albumentations: bool = False):
    #Get training transforms with augmentation.
    
    if use_albumentations:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
            ], p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_val_transforms(image_size: int = 224, use_albumentations: bool = False):
    #Get validation/test transforms without augmentation.
    
    if use_albumentations:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class AlbumentationsTransform:
    #Wrapper for Albumentations transforms to work with PIL images.
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, img):
        # Convert PIL to numpy
        img = np.array(img)
        # Apply transform
        augmented = self.transform(image=img)
        # Return tensor
        return augmented['image']


def get_transforms(image_size: int = 224, is_training: bool = True, use_albumentations: bool = True):
    """
    Get appropriate transforms based on training/validation mode.
    
    Args:
        image_size: Target image size
        is_training: Whether to use training augmentations
        use_albumentations: Use albumentations library (more advanced) or torchvision
    """
    if is_training:
        transform = get_train_transforms(image_size, use_albumentations)
    else:
        transform = get_val_transforms(image_size, use_albumentations)
    
    if use_albumentations:
        return AlbumentationsTransform(transform)
    return transform

