import torch
import torch.nn as nn
import timm
from transformers import ViTModel, ViTConfig, AutoModelForImageClassification


class VisionTransformerTimm(nn.Module):
    #Vision Transformer using timm library.
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1
    ):
        super(VisionTransformerTimm, self).__init__()
        
        self.model_name = model_name
        
        # Load pretrained ViT from timm
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            # Freeze all except classifier head
            for name, param in self.model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)


class DeiTModel(nn.Module):
    #DeiT (Data-efficient Image Transformer) model.
    
    def __init__(
        self,
        model_name: str = "deit3_base_patch16_224",
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1
    ):
        super(DeiTModel, self).__init__()
        
        # Load DeiT from timm
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)


class SwinTransformer(nn.Module):
    #Swin Transformer - Hierarchical vision transformer.
    
    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1
    ):
        super(SwinTransformer, self).__init__()
        
        # Load Swin Transformer from timm
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)


class EVA02Model(nn.Module):
    #EVA-02 - Improved version of EVA (Exploring the Limits of Masked Visual Representation Learning).
    
    def __init__(
        self,
        model_name: str = "eva02_base_patch14_224",
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1
    ):
        super(EVA02Model, self).__init__()
        
        # Load EVA-02 from timm
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)

#Factory function to get transformer models.
def get_transformer_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.1
) -> nn.Module:

    # Standard ViT
    if model_name.startswith('vit_') and 'dinov2' not in model_name:
        return VisionTransformerTimm(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )
    
    # DeiT models
    elif model_name.startswith('deit'):
        return DeiTModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )
    
    # Swin Transformer
    elif model_name.startswith('swin'):
        return SwinTransformer(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )
    
    # EVA-02
    elif model_name.startswith('eva02'):
        return EVA02Model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )
    
    # Generic timm transformer
    else:
        return VisionTransformerTimm(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )

