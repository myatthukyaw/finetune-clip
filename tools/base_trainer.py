from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.optim import lr_scheduler


class BaseTrainer(ABC):
    """Trainer class"""

    def __init__(self, model, args):
        self.args = args
        self.model = model

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def forward_pass(self, images):
        pass

    def move_to_device(self, target):
        return target.to(self.args.device)

    def save_model(self, saved_name):
        torch.save(self.model.state_dict(), saved_name)

    def get_loss_function(self):
        if self.args.loss == "CE":
            return nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        if self.args.loss == "BCE":
            return nn.BCEWithLogitsLoss()

    def get_optimizer(self):
        if self.args.optimizer == "SGD":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                momentum=self.args.momentum,
            )
        elif self.args.optimizer == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def get_scheduler(self):
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

    def get_predictions(self, preds):
        return (
            (torch.sigmoid(preds) > 0.5).float()
            if self.args.loss == "BCE"
            else torch.argmax(preds, 1)
        )