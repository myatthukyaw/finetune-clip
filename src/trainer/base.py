from abc import ABC, abstractmethod
from typing import Union
import os

import torch
from torch import Tensor, nn, optim
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss


class BaseTrainer(ABC):
    """Trainer class"""

    def __init__(self, model, args) -> None:
        self.args = args
        self.model = model

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def eval(self) -> None:
        pass

    @abstractmethod
    def forward_pass(self, images: list) -> None:
        pass

    def move_to_device(self, target) -> None:
        return target.to(self.args.device)

    def save_model(self, saved_name: str) -> None:
        os.makedirs(os.path.dirname(saved_name), exist_ok=True)
        torch.save(self.model.state_dict(), saved_name)
        print(f"Model saved to {saved_name}")

    def get_loss_function(self) -> Union[CrossEntropyLoss, BCEWithLogitsLoss, MSELoss]:
        if self.args.loss == "CE":
            return CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        elif self.args.loss == "BCE":
            return BCEWithLogitsLoss()
        elif self.args.loss == "MSE":
            return MSELoss()
        else:
            raise ValueError(f"Loss {self.args.loss} not implemented.")

    def get_optimizer(self) -> optim.Optimizer:
        if self.args.optimizer == "SGD":
            return optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                momentum=self.args.momentum,
            )
        elif self.args.optimizer == "Adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError(f"Optimizer {self.args.optimizer} not implemented.")

    def get_scheduler(
        self,
    ) -> Union[lr_scheduler.CosineAnnealingLR, lr_scheduler.SequentialLR, None]:
        if self.args.lr_scheduler:
            if self.args.warmup:
                warmup_scheduler = lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.1, total_iters=5
                )
                cosine_scheduler = lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.args.epochs - 5
                )
                return lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[5],
                )
            else:
                return lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.args.epochs
                )
        return None

    def get_predictions(self, outputs):
        """Extract predictions from model outputs based on model type."""
        if self.args.loss == "BCE":
            # For binary classification
            return (outputs > 0).int()
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            # For tuple outputs (like in some models)
            return outputs[0].argmax(dim=1)
        else:
            # For standard classification
            return outputs.argmax(dim=1)
