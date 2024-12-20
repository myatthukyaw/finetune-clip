from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer
from .metrics import Metrics, OverallMetrics


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

        with tqdm(total=len(train_loader)) as pbar:
            for images, labels in train_loader:

                self.optimizer.zero_grad()
                labels = self._prepare_labels(labels)
                images = self.move_to_device(images)
                labels = self.move_to_device(labels)

                preds = self.forward_pass(images)
                loss = self.loss_fn(preds, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if self.args.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                pbar.set_description(
                    f"Epoch {epoch}/{self.args.epochs}, Loss: {loss.item():.4f}"
                )
                pbar.update()

        if self.args.lr_scheduler:
            self.scheduler.step()

        return total_loss / len(train_loader)

    def eval(
        self, val_loader: DataLoader, epoch: int, output_dir: float
    ) -> Tuple[OverallMetrics, float]:

        self.model.eval()
        self.init_class_metrics()

        total_loss = 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            with tqdm(total=len(val_loader), desc="Evaluating") as pbar:
                for images, labels in val_loader:
                    labels = self._prepare_labels(labels, self.args)
                    images = self.move_to_device(images)
                    labels = self.move_to_device(labels)
                    preds = self.forward_pass(images)

                    loss = self.loss_fn(preds, labels)
                    total_loss += loss.item()
                    predictions = self.get_predictions(preds)

                    labels, predictions = labels.tolist(), predictions.tolist()
                    all_labels.extend(labels)
                    all_preds.extend(predictions)

                    self.update_metrics(labels, predictions)

                    pbar.set_postfix(loss=loss.item())
                    pbar.update()

        self.calculate_accuracy()

        if epoch == self.args.epochs:
            self.compute_metrics(all_labels, all_preds)
            self.plot_confusion_metrices(all_labels, all_preds, output_dir)

        return self.overall_metrics, total_loss / len(val_loader)
