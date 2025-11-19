"""
Main training script for all models.
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json

from src.data import DataModule
from src.models import (
    get_baseline_model,
    get_transfer_model,
    get_transformer_model,
    get_sota_2025_model
)
from src.utils.trainer import Trainer
from src.utils.visualization import create_experiment_report


def parse_args():
    parser = argparse.ArgumentParser(description='Train image classification models')
    
    # Model configuration
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['baseline', 'transfer', 'transformer', 'sota'],
                       help='Type of model architecture')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Specific model name')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['fire', 'smoke'],
                       help='Dataset to use')
    
    # Training configuration
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Model-specific
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze backbone weights')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # System configuration
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Input image size')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for models')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    
    return parser.parse_args()


def get_model(args, num_classes):
    """Get model based on arguments."""
    
    print(f"\n{'='*70}")
    print(f"Loading model: {args.model_type} - {args.model_name}")
    print(f"{'='*70}\n")
    
    if args.model_type == 'baseline':
        model = get_baseline_model(
            model_name=args.model_name,
            num_classes=num_classes,
            dropout=args.dropout
        )
    
    elif args.model_type == 'transfer':
        model = get_transfer_model(
            model_name=args.model_name,
            num_classes=num_classes,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
            dropout=args.dropout
        )
    
    elif args.model_type == 'transformer':
        model = get_transformer_model(
            model_name=args.model_name,
            num_classes=num_classes,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
            dropout=args.dropout
        )
    
    elif args.model_type == 'sota':
        model = get_sota_2025_model(
            model_name=args.model_name,
            num_classes=num_classes,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
            dropout=args.dropout
        )
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    print(f"Frozen parameters: {(total_params - trainable_params)/1e6:.2f}M")
    
    return model


def main():
    args = parse_args()

    # Auto-adjust image size for certain SOTA backbones that require a fixed resolution
    # DINOv2 ViT-small (timm/vit_small_patch14_dinov2.lvd142m) expects 518x518 inputs.
    if args.model_type == 'sota' and args.model_name.lower() == 'dinov2':
        # If user didn't explicitly change the default 224, bump to 518 to match the backbone.
        if args.image_size == 224:
            print("[INFO] Overriding image_size to 518 for dinov2 backbone "
                  "(timm/vit_small_patch14_dinov2.lvd142m).")
            args.image_size = 518
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Load data
    data_module = DataModule(
        config_path=args.config,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    train_loader, val_loader, test_loader = data_module.get_loaders()
    dataset_info = data_module.get_dataset_info()
    
    # Get model
    model = get_model(args, dataset_info['num_classes'])
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if args.freeze_backbone:
        # Only optimize unfrozen parameters
        params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        params = model.parameters()
    
    optimizer = optim.Adam(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Generate experiment name
    if args.experiment_name is None:
        experiment_name = f"{args.dataset}_{args.model_type}_{args.model_name}"
    else:
        experiment_name = args.experiment_name
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=dataset_info['num_classes'],
        class_names=dataset_info['class_names'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=args.mixed_precision and device.type == 'cuda',
        save_dir=args.output_dir,
        model_name=experiment_name
    )
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_best=True
    )
    
    # Load best model and test
    best_model_path = Path(args.output_dir) / f"{experiment_name}_best.pth"
    if best_model_path.exists():
        print(f"\nLoading best model from {best_model_path}")
        trainer.load_checkpoint(str(best_model_path))
    
    # Test
    test_results = trainer.test()
    
    # Create experiment report
    figures_dir = Path('figures') / experiment_name
    create_experiment_report(
        model_name=experiment_name,
        dataset_name=args.dataset,
        metrics=test_results['metrics'],
        history=history,
        class_names=dataset_info['class_names'],
        save_dir=str(figures_dir)
    )
    
    # Save experiment config
    experiment_config = {
        'model_type': args.model_type,
        'model_name': args.model_name,
        'dataset': args.dataset,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'pretrained': args.pretrained,
        'freeze_backbone': args.freeze_backbone,
        'dropout': args.dropout,
        'image_size': args.image_size,
        'device': str(device),
        'mixed_precision': args.mixed_precision,
        'best_val_acc': trainer.best_val_acc,
        'test_metrics': test_results['metrics']
    }
    
    config_path = Path(args.output_dir) / f"{experiment_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Experiment completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Figures saved to: {figures_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

