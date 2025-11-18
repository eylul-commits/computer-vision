import torch
import torch.nn as nn
import timm
from torchvision import models


class TransferLearningModel(nn.Module):
    #Generic transfer learning model wrapper.
    
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5
    ):
        super(TransferLearningModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained model using timm (supports many architectures)
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0  # Remove classifier
            )
            in_features = self.backbone.num_features
        except Exception as e:
            print(f"Error loading {model_name} from timm: {e}")
            raise
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Handle different output shapes
        if len(features.shape) == 4:
            # If output is still spatial (B, C, H, W)
            features = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(features, 1), 1)
        
        out = self.classifier(features) if hasattr(self, 'classifier') else features
        return out


class ResNetTransfer(nn.Module):
    #ResNet-based transfer learning model.
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(ResNetTransfer, self).__init__()
        
        # Load pretrained ResNet
        if model_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown ResNet variant: {model_name}")
        
        # Get number of features from last layer
        in_features = self.backbone.fc.in_features
        
        # Replace classifier
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)


class EfficientNetTransfer(nn.Module):
    #EfficientNet-based transfer learning model.
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3
    ):
        super(EfficientNetTransfer, self).__init__()
        
        # Load pretrained EfficientNet from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)


class ConvNeXtTransfer(nn.Module):
    #ConvNeXt-based transfer learning model (Modern CNN architecture).
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "convnext_tiny",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3
    ):
        super(ConvNeXtTransfer, self).__init__()
        
        # Load pretrained ConvNeXt
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)


class MobileNetTransfer(nn.Module):
    #MobileNet-based transfer learning model (lightweight).
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "mobilenetv3_large_100",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.2
    ):
        super(MobileNetTransfer, self).__init__()
        
        # Load pretrained MobileNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)

#Factory function to get transfer learning models.
def get_transfer_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.3
) -> nn.Module:

    # ResNet models
    if model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
        return ResNetTransfer(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    
    # EfficientNet models
    elif 'efficientnet' in model_name:
        return EfficientNetTransfer(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )
    
    # ConvNeXt models
    elif 'convnext' in model_name:
        return ConvNeXtTransfer(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )
    
    # MobileNet models
    elif 'mobilenet' in model_name:
        return MobileNetTransfer(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )
    
    # Generic timm model
    else:
        return TransferLearningModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )

