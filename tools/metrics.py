import os
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             precision_recall_fscore_support)


@dataclass
class ClassMetrics:
    class_name: str
    label: float
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    correct: int = 0
    total: int = 0


@dataclass
class OverallMetrics:
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    precision_weighted: float = 0.0
    recall_weighted: float = 0.0
    f1_weighted: float = 0.0
    accuracy: float = 0.0
    correct: int = 0
    total: int = 0


class Metrics:
    def __init__(self, class_names: List[str]) -> None:
        self.class_names = class_names

    def init_class_metrics(self):
        """Initiate overall and class-wise metrics."""
        self.overall_metrics = OverallMetrics()
        self.class_metrics = [
            ClassMetrics(
                class_name=cls,
                label=i,
                total=0,
                correct=0,
            )
            for i, cls in enumerate(self.classes)
        ]
    
    def update_metrics(self, labels : list, preds : list) -> None:
        """Update metrics for each step."""
        for gt_label, pred in zip(labels, preds):
            if self.args.loss == "BCE":
                gt_label, pred = gt_label[0], pred[0]
            for class_metrix in self.class_metrics:
                if class_metrix.label == gt_label:
                    class_metrix.total += 1
                    if gt_label == pred:
                        class_metrix.correct += 1
                        self.overall_metrics.correct += 1
            self.overall_metrics.total += 1

    def compute_metrics(self, labels: List[int], preds: List[int]) -> None:
        """Compute class-wise and overall metrics."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average=None, labels=list(range(len(self.class_names)))
        )
        (
            self.overall_metrics.precision_macro,
            self.overall_metrics.recall_macro,
            self.overall_metrics.f1_macro,
            _,
        ) = precision_recall_fscore_support(
            labels, preds, average="macro", labels=list(range(len(self.class_names)))
        )
        (
            self.overall_metrics.precision_weighted,
            self.overall_metrics.recall_weighted,
            self.overall_metrics.f1_weighted,
            _,
        ) = precision_recall_fscore_support(
            labels,
            preds,
            average="weighted",
            labels=list(range(len(self.class_names))),
        )

        for i, cls_metrix in enumerate(self.class_metrics):
            cls_metrix.precision = precision[i]
            cls_metrix.recall = recall[i]
            cls_metrix.f1 = f1[i]

        self._print_metrics()

    def calculate_accuracy(self) -> None:
        """Calculate total accuracy and accuracy for each class."""
        self.overall_metrics.accuracy = round(100 * self.overall_metrics.correct / self.overall_metrics.total,4)
        for cls_met in self.class_metrics:
            cls_met.accuracy = round(100 * cls_met.correct / cls_met.total, 4) if cls_met.total > 0 else 0
            print(f"Eval accuracy on {cls_met.class_name} : {cls_met.accuracy:.4f}")
        print(f"\nTotal eval accuracy (Top-1): {self.overall_metrics.accuracy:.4f}%\n")

    def plot_confusion_metrices(self, labels: list, preds: list, output_dir: str) -> None:
        """Plot and save confusion metrices."""
        self._plot_cm(labels, preds, output_dir, normalized=False)
        self._plot_cm(labels, preds, output_dir, normalized=True)

    def _plot_cm(self, labels: list, preds: list, output_dir: str, normalized=False) -> None:
        """Plot and save confusion metrix."""
        suffix = "_normalized" if normalized else None
        save_path = os.path.join(output_dir, f"confusion_matrix{suffix}.png")
        cm = confusion_matrix(
            labels,
            preds,
            labels=list(range(len(self.class_names))),
            normalize="true" if normalized else None,
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.class_names
        )
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(save_path, bbox_inches="tight")

    def write_to_tensorboard(self, writer, train_loss, eval_loss, epoch):
        writer.add_scalar("Train loss", train_loss, epoch)
        writer.add_scalar("Eval loss", eval_loss, epoch)
        writer.add_scalar("Eval accuracy", self.metrics.accuracy, epoch)
        writer.add_scalar("Precision Macro", self.overall_metrics.precision_macro, epoch)
        writer.add_scalar("Precision Macro", self.overall_metrics.recall_macro, epoch)
        writer.add_scalar("Precision Macro", self.overall_metrics.f1_macro, epoch)
        writer.add_scalar("Precision Weighted", self.overall_metrics.precision_weighted, epoch)
        writer.add_scalar("Precision Weighted", self.overall_metrics.recall_weighted, epoch)
        writer.add_scalar("Precision Weighted", self.overall_metrics.f1_weighted, epoch)

    def _print_metrics(self) -> None:
        """Print class-wise metrics and overall metrics."""

        print("\nClass-wise Metrics:")
        print(
            f"{'Class':<25}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}{'Accuracy':>12}"
        )
        print("-" * 70)
        for metric in self.class_metrics:
            print(
                f"{metric.class_name:<25}"
                f"{metric.accuracy:>12.2f}{metric.precision * 100:>12.2f}"
                f"{metric.recall * 100:>12.2f}{metric.f1 * 100:>12.2f}"
            )

        print("\nOverall Metrics:")
        print(f"{'Metric':<15}{'Macro':>12}{'Weighted':>12}")
        print("-" * 39)
        print(
            f"{'Precision':<15}{self.overall_metrics.precision_macro * 100:>12.2f}"
            f"{self.overall_metrics.precision_weighted * 100:>12.2f}"
        )
        print(
            f"{'Recall':<15}{self.overall_metrics.recall_macro * 100:>12.2f}"
            f"{self.overall_metrics.recall_weighted * 100:>12.2f}"
        )
        print(
            f"{'F1-Score':<15}{self.overall_metrics.f1_macro * 100:>12.2f}"
            f"{self.overall_metrics.f1_weighted * 100:>12.2f}"
        )

        print(f"\nTotal eval accuracy (Top-1 accuracy): {self.overall_metrics.accuracy:.4f}%\n")