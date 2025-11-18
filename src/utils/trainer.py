import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import time

from .metrics import MetricsCalculator


class Trainer:
    """
    Trainer class for training and evaluating models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_classes: int,
        class_names: list,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        mixed_precision: bool = True,
        save_dir: str = 'models',
        model_name: str = 'model'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.class_names = class_names
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mixed_precision = mixed_precision
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        print(f"\nTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  Mixed Precision: {mixed_precision}")
        print(f"  Model: {model_name}")
        print(f"  Save Directory: {save_dir}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> Tuple[float, float, Dict]:
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        
        metrics_calc = MetricsCalculator(self.num_classes, self.class_names)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # Update metrics
                metrics_calc.update(predicted, labels, probs)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1)
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        metrics = metrics_calc.compute()
        epoch_acc = metrics['accuracy']
        
        return epoch_loss, epoch_acc, metrics
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10,
        save_best: bool = True
    ) -> Dict:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save the best model
        
        Returns:
            Training history
        """
        print(f"\n{'='*70}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  Val F1:     {val_metrics['f1']:.4f} | LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                patience_counter = 0
                
                if save_best:
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"  ✓ New best model saved! (Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                print(f"  Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        print(f"{'='*70}\n")
        
        # Save final model and history
        self.save_checkpoint(epoch, is_best=False)
        self.save_history()
        
        return self.history
    
    def test(self) -> Dict:
        """Test the model on test set."""
        print(f"\n{'='*70}")
        print("Evaluating on test set...")
        print(f"{'='*70}\n")
        
        self.model.eval()
        metrics_calc = MetricsCalculator(self.num_classes, self.class_names)
        
        test_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                test_loss += loss.item()
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # Update metrics
                metrics_calc.update(predicted, labels, probs)
        
        test_loss /= len(self.test_loader)
        metrics = metrics_calc.compute()
        
        print(f"\nTest Loss: {test_loss:.4f}")
        metrics_calc.print_metrics()
        
        # Save test metrics
        test_results = {
            'test_loss': test_loss,
            'metrics': metrics
        }
        
        save_path = self.save_dir / f'{self.model_name}_test_results.json'
        with open(save_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"Test results saved to {save_path}")
        
        return test_results
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if is_best:
            save_path = self.save_dir / f'{self.model_name}_best.pth'
        else:
            save_path = self.save_dir / f'{self.model_name}_final.pth'
        
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def save_history(self):
        """Save training history."""
        save_path = self.save_dir / f'{self.model_name}_history.json'
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {save_path}")

