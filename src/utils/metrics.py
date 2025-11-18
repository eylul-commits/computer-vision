import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc
)
from typing import Dict, List, Tuple
import json


class MetricsCalculator:
    """Calculate and store evaluation metrics."""
    
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and labels."""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, preds: torch.Tensor, labels: torch.Tensor, probs: torch.Tensor = None):
        """
        Update metrics with new batch of predictions.
        
        Args:
            preds: Predicted class indices
            labels: Ground truth labels
            probs: Class probabilities (optional)
        """
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())
        
        if probs is not None:
            self.all_probs.extend(probs.cpu().numpy().tolist())
    
    def compute(self) -> Dict:
        """
        Compute all metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle binary vs multiclass
        average = 'binary' if self.num_classes == 2 else 'weighted'
        
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0, labels=range(self.num_classes)
        )
        recall_per_class = recall_score(
            y_true, y_pred, average=None, zero_division=0, labels=range(self.num_classes)
        )
        f1_per_class = f1_score(
            y_true, y_pred, average=None, zero_division=0, labels=range(self.num_classes)
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            labels=range(self.num_classes),
            zero_division=0
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'per_class': {
                'precision': {self.class_names[i]: float(precision_per_class[i]) 
                             for i in range(self.num_classes)},
                'recall': {self.class_names[i]: float(recall_per_class[i]) 
                          for i in range(self.num_classes)},
                'f1': {self.class_names[i]: float(f1_per_class[i]) 
                      for i in range(self.num_classes)},
            }
        }
        
        # ROC-AUC (if probabilities are available)
        if len(self.all_probs) > 0:
            y_probs = np.array(self.all_probs)
            
            if self.num_classes == 2:
                # Binary classification
                try:
                    roc_auc = roc_auc_score(y_true, y_probs[:, 1])
                    metrics['roc_auc'] = float(roc_auc)
                except:
                    metrics['roc_auc'] = None
            else:
                # Multiclass
                try:
                    roc_auc = roc_auc_score(
                        y_true, y_probs,
                        multi_class='ovr',
                        average='weighted',
                        labels=range(self.num_classes)
                    )
                    metrics['roc_auc'] = float(roc_auc)
                    
                    # Per-class ROC-AUC
                    roc_auc_per_class = {}
                    for i in range(self.num_classes):
                        try:
                            y_true_binary = (y_true == i).astype(int)
                            roc_auc_class = roc_auc_score(y_true_binary, y_probs[:, i])
                            roc_auc_per_class[self.class_names[i]] = float(roc_auc_class)
                        except:
                            roc_auc_per_class[self.class_names[i]] = None
                    
                    metrics['per_class']['roc_auc'] = roc_auc_per_class
                except:
                    metrics['roc_auc'] = None
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)
        return confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
    
    def get_roc_data(self) -> Dict:
        """Get ROC curve data for plotting."""
        if len(self.all_probs) == 0:
            return None
        
        y_true = np.array(self.all_labels)
        y_probs = np.array(self.all_probs)
        
        roc_data = {}
        
        if self.num_classes == 2:
            # Binary classification
            fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            roc_data['binary'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': float(roc_auc)
            }
        else:
            # Multiclass - one-vs-rest
            for i in range(self.num_classes):
                y_true_binary = (y_true == i).astype(int)
                try:
                    fpr, tpr, thresholds = roc_curve(y_true_binary, y_probs[:, i])
                    roc_auc = auc(fpr, tpr)
                    roc_data[self.class_names[i]] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'thresholds': thresholds.tolist(),
                        'auc': float(roc_auc)
                    }
                except:
                    continue
        
        return roc_data
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON file."""
        metrics = self.compute()
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")
    
    def print_metrics(self):
        """Print metrics in a formatted way."""
        metrics = self.compute()
        
        print("\n" + "=" * 70)
        print("EVALUATION METRICS")
        print("=" * 70)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        
        if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print("\n" + "-" * 70)
        print("PER-CLASS METRICS")
        print("-" * 70)
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 70)
        
        for class_name in self.class_names:
            prec = metrics['per_class']['precision'][class_name]
            rec = metrics['per_class']['recall'][class_name]
            f1 = metrics['per_class']['f1'][class_name]
            print(f"{class_name:<15} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
        
        print("\n" + "-" * 70)
        print("CLASSIFICATION REPORT")
        print("-" * 70)
        print(metrics['classification_report'])
        print("=" * 70 + "\n")


def calculate_top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        logits: Model output logits
        labels: Ground truth labels
        k: Top-k value
    
    Returns:
        Top-k accuracy
    """
    with torch.no_grad():
        _, top_k_preds = logits.topk(k, dim=1)
        correct = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds))
        top_k_acc = correct.any(dim=1).float().mean().item()
    
    return top_k_acc

