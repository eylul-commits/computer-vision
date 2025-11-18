import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = False,
    figsize: tuple = (10, 8)
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the figure
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_roc_curves(
    roc_data: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """
    Plot ROC curves.
    
    Args:
        roc_data: Dictionary containing ROC curve data
        save_path: Path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    if 'binary' in roc_data:
        # Binary classification
        data = roc_data['binary']
        plt.plot(
            data['fpr'],
            data['tpr'],
            label=f"ROC curve (AUC = {data['auc']:.3f})",
            linewidth=2
        )
    else:
        # Multiclass - plot each class
        for class_name, data in roc_data.items():
            plt.plot(
                data['fpr'],
                data['tpr'],
                label=f"{class_name} (AUC = {data['auc']:.3f})",
                linewidth=2
            )
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.close()


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """
    Plot training history (loss and metrics).
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Plot learning rate
    if 'learning_rate' in history:
        axes[2].plot(history['learning_rate'], linewidth=2, color='green')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Learning Rate', fontsize=12)
        axes[2].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[2].grid(alpha=0.3)
        axes[2].set_yscale('log')
    else:
        # Plot F1 score if learning rate not available
        if 'train_f1' in history:
            axes[2].plot(history['train_f1'], label='Train F1', linewidth=2)
        if 'val_f1' in history:
            axes[2].plot(history['val_f1'], label='Val F1', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('F1 Score', fontsize=12)
        axes[2].set_title('F1 Score', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.close()


def plot_per_class_metrics(
    metrics: Dict,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    Plot per-class metrics (precision, recall, F1).
    
    Args:
        metrics: Dictionary containing per-class metrics
        class_names: List of class names
        save_path: Path to save the figure
        figsize: Figure size
    """
    x = np.arange(len(class_names))
    width = 0.25
    
    precision = [metrics['precision'][cls] for cls in class_names]
    recall = [metrics['recall'][cls] for cls in class_names]
    f1 = [metrics['f1'][cls] for cls in class_names]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics saved to {save_path}")
    
    plt.close()


def plot_sample_predictions(
    images: torch.Tensor,
    true_labels: List[int],
    pred_labels: List[int],
    class_names: List[str],
    probabilities: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    num_samples: int = 16,
    figsize: tuple = (16, 16)
):
    """
    Plot sample predictions with images.
    
    Args:
        images: Tensor of images
        true_labels: List of true labels
        pred_labels: List of predicted labels
        class_names: List of class names
        probabilities: Prediction probabilities
        save_path: Path to save the figure
        num_samples: Number of samples to plot
        figsize: Figure size
    """
    num_samples = min(num_samples, len(images))
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    # Denormalize images (assuming ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for idx in range(num_samples):
        img = images[idx].cpu()
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        true_label = class_names[true_labels[idx]]
        pred_label = class_names[pred_labels[idx]]
        
        # Color: green if correct, red if incorrect
        color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        title = f"True: {true_label}\nPred: {pred_label}"
        if probabilities is not None:
            conf = probabilities[idx, pred_labels[idx]].item()
            title += f"\nConf: {conf:.2f}"
        
        axes[idx].set_title(title, fontsize=10, color=color, fontweight='bold')
    
    # Hide extra subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Sample predictions saved to {save_path}")
    
    plt.close()


def compare_models(
    results: Dict[str, Dict],
    metric: str = 'accuracy',
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    Compare multiple models on a specific metric.
    
    Args:
        results: Dictionary mapping model names to their results
        metric: Metric to compare
        save_path: Path to save the figure
        figsize: Figure size
    """
    model_names = list(results.keys())
    values = [results[model][metric] for model in model_names]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(range(len(model_names)), values, alpha=0.8)
    
    # Color bars based on performance
    colors = plt.cm.RdYlGn(np.array(values) / max(values))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f'Model Comparison - {metric.capitalize()}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    
    plt.close()


def create_experiment_report(
    model_name: str,
    dataset_name: str,
    metrics: Dict,
    history: Dict,
    class_names: List[str],
    save_dir: str
):
    """
    Create a comprehensive experiment report with all visualizations.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        metrics: Evaluation metrics
        history: Training history
        class_names: List of class names
        save_dir: Directory to save the report
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating experiment report for {model_name} on {dataset_name}...")
    
    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        cm, class_names,
        save_path=str(save_dir / f'{model_name}_confusion_matrix.png')
    )
    plot_confusion_matrix(
        cm, class_names,
        save_path=str(save_dir / f'{model_name}_confusion_matrix_normalized.png'),
        normalize=True
    )
    
    # Plot training history
    if history:
        plot_training_history(
            history,
            save_path=str(save_dir / f'{model_name}_training_history.png')
        )
    
    # Plot per-class metrics
    if 'per_class' in metrics:
        plot_per_class_metrics(
            metrics['per_class'],
            class_names,
            save_path=str(save_dir / f'{model_name}_per_class_metrics.png')
        )
    
    print(f"Experiment report saved to {save_dir}")

