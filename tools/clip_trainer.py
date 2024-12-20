from typing import Tuple

import clip
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer
from .metrics import Metrics, OverallMetrics


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

        with tqdm(total=len(train_loader)) as pbar:
            for images, texts, _, _ in train_loader:

                self.optimizer.zero_grad()
                images = self.move_to_device(images)
                texts = self.move_to_device(texts)

                logits_per_img, logits_per_text = self.forward_pass(images, texts)

                ground_truth = torch.arange(
                    len(images), dtype=torch.long, device=self.args.device
                )
                loss = (
                    self.loss_img(logits_per_img, ground_truth)
                    + self.loss_txt(logits_per_text, ground_truth)
                ) / 2

                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.item())

                if self.args.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                pbar.set_description(
                    f"Epoch {epoch}/{self.args.epochs}, Loss: {loss.item():.4f}"
                )
                pbar.update()

        return total_loss

    def eval(
        self, val_loader: DataLoader, epoch: int, output_dir: float
    ) -> Tuple[OverallMetrics, float]:

        self.model.eval()
        self.init_class_metrics()

        total_loss = 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            with tqdm(total=len(val_loader), desc="Evaluating") as pbar:
                for images, _, labels, _ in val_loader:
                    images = self.move_to_device(images)

                    # Calculate similarity scores between image and text
                    logits_per_img, logits_per_text = self.forward_pass(
                        images, self.text_tokens
                    )
                    ground_truth = self.move_to_device(labels)
                    loss = (
                        self.loss_img(logits_per_img, ground_truth)
                        + self.loss_txt(logits_per_text.T, ground_truth)
                    ) / 2
                    probs = logits_per_img.softmax(dim=-1).cpu().numpy()

                    # Get the indices of the max probability for each image
                    indices = probs.argmax(axis=-1)
                    predictions = indices.tolist()
                    labels = labels.tolist()

                    all_labels.extend(labels)
                    all_preds.extend(predictions)

                    self.update_metrics(labels, predictions)

                    total_loss += float(loss.item())
                    pbar.set_postfix(loss=loss.item())
                    pbar.update()

        self.calculate_accuracy()

        if epoch == self.args.epochs:
            self.compute_metrics(all_labels, all_preds)
            self.plot_confusion_metrices(all_labels, all_preds, output_dir)

        return self.overall_metrics, total_loss / len(val_loader)
