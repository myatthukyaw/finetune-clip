from typing import Tuple

import clip
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.trainer.base import BaseTrainer
from src.tools.metrics import Metrics, OverallMetrics


def generate_descriptions(classes: list) -> dict:
    descriptions = {}
    for cls in classes:
        descriptions[cls] = f"This is a photo containing a {cls}."
    return descriptions


class ClipTrainer(BaseTrainer, Metrics):
    """Trainer class"""

    def __init__(self, model, classes, descriptions, args) -> None:
        BaseTrainer.__init__(self, model, args)
        Metrics.__init__(self, classes)
        self.args = args
        self.model = model
        self.classes = classes
        self.descriptions = descriptions
        self.text_tokens = self.generate_text_tokens()
        self.model = self.move_to_device(model)
        self.loss_img = self.get_loss_function()
        self.loss_txt = self.get_loss_function()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

    def generate_text_tokens(self) -> Tensor:
        text_tokens = torch.cat([clip.tokenize(c) for c in self.descriptions.values()])
        return self.move_to_device(text_tokens)

    def argmax(self, iterable) -> int:
        return max(enumerate(iterable), key=lambda x: x[1])[0]

    def forward_pass(self, images: Tensor, texts: Tensor) -> Tuple[Tensor, Tensor]:
        return self.model(images, texts)

    def train(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        print("\n" + "="*70)
        print(f"{'CLIP TRAINING - EPOCH ' + str(epoch) + '/' + str(self.args.epochs):^70}")
        print("="*70)

        with tqdm(total=len(train_loader), desc=f"Train Epoch {epoch}/{self.args.epochs}") as pbar:
            for batch_idx, (images, texts, _, _) in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                # Move data to device
                images = self.move_to_device(images)
                texts = self.move_to_device(texts)

                # Forward pass
                logits_per_img, logits_per_text = self.forward_pass(images, texts)
                
                # Calculate ground truth indices
                ground_truth = torch.arange(
                    len(images), dtype=torch.long, device=self.args.device
                )
                
                # Calculate loss (combined image-text and text-image losses)
                loss = (
                    self.loss_img(logits_per_img, ground_truth)
                    + self.loss_txt(logits_per_text, ground_truth)
                ) / 2

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy for progress display
                img_correct = (logits_per_img.argmax(dim=-1) == ground_truth).sum().item()
                txt_correct = (logits_per_text.argmax(dim=-1) == ground_truth).sum().item()
                batch_correct = (img_correct + txt_correct) / 2
                correct += batch_correct
                batch_size = len(images)
                total += batch_size
                
                # Update total loss and calculate running metrics
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
        print(f"{'CLIP VALIDATION - EPOCH ' + str(epoch) + '/' + str(self.args.epochs):^70}")
        print("="*70)

        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Eval Epoch {epoch}/{self.args.epochs}") as pbar:
                for batch_idx, (images, _, labels, _) in enumerate(val_loader):
                    # Move data to device
                    images = self.move_to_device(images)
                    batch_labels = labels.tolist()

                    # Calculate similarity scores between image and text
                    logits_per_img, logits_per_text = self.forward_pass(
                        images, self.text_tokens
                    )
                    
                    # Calculate loss
                    ground_truth = self.move_to_device(labels)
                    loss = (
                        self.loss_img(logits_per_img, ground_truth)
                        + self.loss_txt(logits_per_text.T, ground_truth)
                    ) / 2
                    
                    # Calculate probabilities and get predictions
                    probs = logits_per_img.softmax(dim=-1).cpu().numpy()
                    indices = probs.argmax(axis=-1)
                    batch_preds = indices.tolist()
                    
                    # Update overall metrics
                    all_labels.extend(batch_labels)
                    all_preds.extend(batch_preds)
                    self.update_metrics(batch_labels, batch_preds)
                    
                    # Calculate batch accuracy for progress display
                    batch_correct = sum(1 for p, l in zip(batch_preds, batch_labels) if p == l)
                    batch_size = len(batch_labels)
                    batch_acc = 100. * batch_correct / batch_size if batch_size > 0 else 0

                    # Update total loss
                    total_loss += float(loss.item())
                    
                    # Update progress bar with metrics
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
