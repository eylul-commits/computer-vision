"""Model architectures for image classification."""
from .baseline_cnn import SimpleCNN, ImprovedCNN, LightweightCNN, get_baseline_model
from .transfer_learning import (
    TransferLearningModel,
    ResNetTransfer,
    EfficientNetTransfer,
    ConvNeXtTransfer,
    MobileNetTransfer,
    get_transfer_model
)
from .transformers import (
    VisionTransformerTimm,
    DeiTModel,
    SwinTransformer,
    EVA02Model,
    get_transformer_model
)
from .sota_2025 import DinoV3Model, get_sota_2025_model, list_available_sota_models

__all__ = [
    # Baseline CNNs
    'SimpleCNN',
    'ImprovedCNN',
    'LightweightCNN',
    'get_baseline_model',
    
    # Transfer Learning
    'TransferLearningModel',
    'ResNetTransfer',
    'EfficientNetTransfer',
    'ConvNeXtTransfer',
    'MobileNetTransfer',
    'get_transfer_model',
    
    # Transformers
    'VisionTransformerTimm',
    'DeiTModel',
    'SwinTransformer',
    'EVA02Model',
    'get_transformer_model',
    
    # SOTA 2025 (restricted to DINOv3)
    'DinoV3Model',
    'get_sota_2025_model',
    'list_available_sota_models',
]

