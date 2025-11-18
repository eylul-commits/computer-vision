import torch
import torch.nn as nn
import timm

class DinoV3Model(nn.Module):
    """
    DINO-style wrapper using a timm backbone.

    Notes:
    - `backbone_name` should point to a DINOv2/3-compatible checkpoint available in your setup.
      By default we use the `timm/vit_small_patch14_dinov2.lvd142m` backbone; change this string
      if you use a different DINO model id.
    """

    def __init__(
        self,
        backbone_name: str = "timm/vit_small_patch14_dinov2.lvd142m",
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        print(f"Loading DINOv3-style backbone from timm: {backbone_name}")

        # Use timm model as the DINOv3 backbone. Replace `backbone_name` with your DINOv3 id.
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )

        # Optionally freeze everything except the classifier head
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "head" not in name and "fc" not in name and "classifier" not in name:
                    param.requires_grad = False

        print(f"DINOv3-style model parameters: "
              f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_sota_2025_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.1,
) -> nn.Module:

    key = model_name.lower()

    if key == "dinov2":
        return DinoV3Model(
            backbone_name="timm/vit_small_patch14_dinov2.lvd142m",
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
        )

    raise ValueError(
        f"Unknown SOTA 2025 model '{model_name}'. "
        f"Available: {list_available_sota_models()}"
    )


def list_available_sota_models():
    """
    Return a list of available SOTA 2025 model identifiers.

    This is used by the higher-level API (e.g., CLI help or documentation)
    to surface which `model_name` values are valid for `--model-type sota`.
    """
    return ["dinov2"]
