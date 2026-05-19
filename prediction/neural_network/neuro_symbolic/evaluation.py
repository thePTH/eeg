from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
)

from prediction.neural_network.helpers import MacroToMicroSegmenter
from prediction.neural_network.neural_backbone.logits_aggregator import (
    MicroLogitsToMacroProbabilityAggregator,
)


@dataclass(frozen=True)
class ClassEvaluationMetrics:
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "support": self.support,
        }


@dataclass(frozen=True)
class NeuralNetworkEvaluationResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray

    threshold: float
    macro_aggregation_method: str

    accuracy: float
    balanced_accuracy: float

    macro_precision: float
    macro_recall: float
    macro_f1: float

    weighted_precision: float
    weighted_recall: float
    weighted_f1: float

    roc_auc: float | None
    average_precision: float | None

    matthews_corrcoef: float
    cohen_kappa: float

    confusion_matrix: np.ndarray
    class_metrics: tuple[ClassEvaluationMetrics, ...]

    def predictions_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "y_true": self.y_true,
                "y_pred": self.y_pred,
                "y_proba_ad": self.y_proba,
            }
        )

    def class_metrics_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [metric.to_dict() for metric in self.class_metrics]
        )

    def confusion_matrix_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.confusion_matrix,
            index=["true_healthy", "true_alzheimer"],
            columns=["pred_healthy", "pred_alzheimer"],
        )

    def summary_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "threshold": self.threshold,
                    "macro_aggregation_method": self.macro_aggregation_method,
                    "accuracy": self.accuracy,
                    "balanced_accuracy": self.balanced_accuracy,
                    "macro_precision": self.macro_precision,
                    "macro_recall": self.macro_recall,
                    "macro_f1": self.macro_f1,
                    "weighted_precision": self.weighted_precision,
                    "weighted_recall": self.weighted_recall,
                    "weighted_f1": self.weighted_f1,
                    "roc_auc": self.roc_auc,
                    "average_precision": self.average_precision,
                    "matthews_corrcoef": self.matthews_corrcoef,
                    "cohen_kappa": self.cohen_kappa,
                    "n_samples": len(self.y_true),
                }
            ]
        )

    def print_report(self) -> None:
        print("\n" + "=" * 90)
        print("EEG NEURAL NETWORK EVALUATION")
        print("=" * 90)

        print("\n[0] Evaluation setup")
        print("-" * 90)
        print(f"Threshold:                 {self.threshold:.4f}")
        print(f"Aggregation method:        {self.macro_aggregation_method}")

        print("\n[1] Global metrics")
        print("-" * 90)
        print(f"Accuracy:                  {self.accuracy:.4f}")
        print(f"Balanced accuracy:         {self.balanced_accuracy:.4f}")
        print(f"Macro precision:           {self.macro_precision:.4f}")
        print(f"Macro recall:              {self.macro_recall:.4f}")
        print(f"Macro F1-score:            {self.macro_f1:.4f}")
        print(f"Weighted precision:        {self.weighted_precision:.4f}")
        print(f"Weighted recall:           {self.weighted_recall:.4f}")
        print(f"Weighted F1-score:         {self.weighted_f1:.4f}")

        if self.roc_auc is not None:
            print(f"ROC AUC:                   {self.roc_auc:.4f}")

        if self.average_precision is not None:
            print(f"Average precision:         {self.average_precision:.4f}")

        print(f"Matthews corrcoef:         {self.matthews_corrcoef:.4f}")
        print(f"Cohen kappa:               {self.cohen_kappa:.4f}")

        print("\n[2] Confusion matrix")
        print("-" * 90)
        print(self.confusion_matrix_dataframe().to_string())

        print("\n[3] Metrics per class")
        print("-" * 90)
        print(self.class_metrics_dataframe().to_string(index=False))

        print("\n" + "=" * 90)


class NeuralNetworkEvaluationEngine:
    @staticmethod
    @torch.no_grad()
    def evaluate(
        model: nn.Module,
        dataloader,
        threshold: float = 0.5,
        n_micro_segments: int = 60,
        macro_aggregation_method: str = "mean_logit",
        class_names: tuple[str, str] = ("Healthy", "Alzheimer"),
    ) -> NeuralNetworkEvaluationResult:

        device = next(model.parameters()).device
        model.eval()

        y_true_list: list[torch.Tensor] = []
        y_proba_list: list[torch.Tensor] = []

        for macro_x_raw, macro_x_feat, y_true in dataloader:
            macro_x_raw = macro_x_raw.to(device)

            micro_x_raws = MacroToMicroSegmenter.split(
                macro_x_raw,
                n_micro_segments=n_micro_segments,
            )

            micro_logits = torch.stack(
                [
                    model(micro_x_raw).squeeze(-1)
                    for micro_x_raw in micro_x_raws
                ]
            )

            

            macro_ad_proba = MicroLogitsToMacroProbabilityAggregator.compute(
                micro_logits=micro_logits,
                method=macro_aggregation_method,
            )

            if macro_ad_proba.ndim != 1:
                raise ValueError(
                    "Expected macro_ad_proba with shape [B]. "
                    f"Got {macro_ad_proba.shape}."
                )

            y_true_list.append(y_true.detach().cpu())
            y_proba_list.append(macro_ad_proba.detach().cpu())

        y_true_arr = torch.cat(y_true_list).numpy().astype(int)
        y_proba_arr = torch.cat(y_proba_list).numpy()

        y_pred_arr = (y_proba_arr >= threshold).astype(int)

        labels = [0, 1]

        accuracy = accuracy_score(y_true_arr, y_pred_arr)
        balanced_accuracy = balanced_accuracy_score(y_true_arr, y_pred_arr)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            zero_division=0,
        )

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            average="weighted",
            zero_division=0,
        )

        class_metrics = tuple(
            ClassEvaluationMetrics(
                class_name=class_names[i],
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1_score=float(f1[i]),
                support=int(support[i]),
            )
            for i in labels
        )

        cm = confusion_matrix(
            y_true_arr,
            y_pred_arr,
            labels=labels,
        )

        mcc = matthews_corrcoef(y_true_arr, y_pred_arr)
        kappa = cohen_kappa_score(y_true_arr, y_pred_arr)

        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_proba_arr))
        except ValueError:
            roc_auc = None

        try:
            average_precision = float(
                average_precision_score(y_true_arr, y_proba_arr)
            )
        except ValueError:
            average_precision = None

        return NeuralNetworkEvaluationResult(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            y_proba=y_proba_arr,
            threshold=float(threshold),
            macro_aggregation_method=macro_aggregation_method,
            accuracy=float(accuracy),
            balanced_accuracy=float(balanced_accuracy),
            macro_precision=float(macro_precision),
            macro_recall=float(macro_recall),
            macro_f1=float(macro_f1),
            weighted_precision=float(weighted_precision),
            weighted_recall=float(weighted_recall),
            weighted_f1=float(weighted_f1),
            roc_auc=roc_auc,
            average_precision=average_precision,
            matthews_corrcoef=float(mcc),
            cohen_kappa=float(kappa),
            confusion_matrix=cm,
            class_metrics=class_metrics,
        )