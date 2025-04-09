from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.trainer.base import BaseTrainer
from src.tools.metrics import Metrics, OverallMetrics


class Trainer(BaseTrainer, Metrics):
    """Trainer class"""

    def __init__(self, model, classes, args) -> None:
        BaseTrainer.__init__(self, model, args)
        Metrics.__init__(self, classes)
        self.args = args
        self.model = model
        self.classes = classes
        self.model = self.move_to_device(model)
        self.loss_fn = self.get_loss_function()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

    def _prepare_labels(self, labels: np.ndarray) -> None:
        return labels.float().unsqueeze(1) if self.args.loss == "BCE" else labels.long()

    def forward_pass(self, images: Tensor) -> Tensor:
        return self.model(images)

    def train(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        print("\n" + "="*70)
        print(f"{'TRAINING - EPOCH ' + str(epoch) + '/' + str(self.args.epochs):^70}")
        print("="*70)

        with tqdm(total=len(train_loader), desc=f"Train Epoch {epoch}/{self.args.epochs}") as pbar:
            for batch_idx, (images, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                # Prepare inputs and labels
                original_labels = labels.clone()
                labels = self._prepare_labels(labels)
                images = self.move_to_device(images)
                labels = self.move_to_device(labels)

                # Forward pass and loss calculation
                preds = self.forward_pass(images)
                loss = self.loss_fn(preds, labels)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Calculate batch accuracy for progress display
                predictions = self.get_predictions(preds)
                batch_correct = (predictions == labels).sum().item()
                correct += batch_correct
                batch_size = labels.size(0)
                total += batch_size
                
                # Update metrics
                total_loss += float(loss.item())
                batch_acc = 100. * batch_correct / batch_size
                running_acc = 100. * correct / total
                
                # Apply gradient clipping if enabled
                if self.args.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                # Update progress bar with metrics
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'batch_acc': f'{batch_acc:.2f}%',
                    'run_acc': f'{running_acc:.2f}%'
                })
                pbar.update()

        # Step learning rate scheduler if enabled
        if self.args.lr_scheduler:
            self.scheduler.step()
            
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100. * correct / total
        
        print(f"Training Summary: Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%")
        return avg_loss

    def eval(
        self, val_loader: DataLoader, epoch: int, output_dir: float
    ) -> Tuple[OverallMetrics, float]:

        self.model.eval()
        self.init_class_metrics()

        total_loss = 0
        all_labels, all_preds = [], []
        
        print("\n" + "="*70)
        print(f"{'VALIDATION - EPOCH ' + str(epoch) + '/' + str(self.args.epochs):^70}")
        print("="*70)

        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Eval Epoch {epoch}/{self.args.epochs}") as pbar:
                for batch_idx, (images, labels) in enumerate(val_loader):
                    # Prepare inputs and labels
                    original_labels = labels.clone()
                    labels = self._prepare_labels(labels)
                    images = self.move_to_device(images)
                    labels = self.move_to_device(labels)
                    
                    # Forward pass and predictions
                    preds = self.forward_pass(images)
                    loss = self.loss_fn(preds, labels)
                    predictions = self.get_predictions(preds)

                    # Convert to lists for metric calculation
                    batch_labels = labels.cpu().tolist()
                    batch_preds = predictions.cpu().tolist()
                    
                    # Update running metrics
                    all_labels.extend(batch_labels)
                    all_preds.extend(batch_preds)
                    self.update_metrics(batch_labels, batch_preds)
                    
                    # Update loss
                    total_loss += float(loss.item())
                    
                    # Calculate batch accuracy for progress display
                    batch_correct = (predictions == labels).sum().item()
                    batch_size = labels.size(0)
                    batch_acc = 100. * batch_correct / batch_size
                    
                    # Update progress bar with current batch metrics
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'batch_acc': f'{batch_acc:.2f}%'
                    })
                    pbar.update()

        # Calculate and display overall accuracy
        self.calculate_accuracy()

        # Compute detailed metrics at the end of training
        if epoch == self.args.epochs:
            self.compute_metrics(all_labels, all_preds)
            self.plot_confusion_metrices(all_labels, all_preds, output_dir)

        avg_loss = total_loss / len(val_loader)
        return self.overall_metrics, avg_loss
