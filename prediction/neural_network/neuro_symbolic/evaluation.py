from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm.auto import tqdm

from prediction.neural_network.neural_backbone.logits_aggregator import (
    MicroLogitsToMacroProbabilityAggregator,
)


@dataclass(frozen=True)
class ClassEvaluationMetrics:
    """Per-class evaluation metrics."""

    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int

    def to_dict(self) -> dict[str, Any]:
        """Return metrics as a dictionary."""
        return {
            "class_name": self.class_name,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "support": self.support,
        }


@dataclass(frozen=True)
class NeuralNetworkEvaluationResult:
    """Complete evaluation result for a binary neural-network classifier."""

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
        """Return sample-level predictions as a DataFrame."""
        return pd.DataFrame(
            {
                "y_true": self.y_true,
                "y_pred": self.y_pred,
                "y_proba_ad": self.y_proba,
            }
        )

    def class_metrics_dataframe(self) -> pd.DataFrame:
        """Return per-class metrics as a DataFrame."""
        return pd.DataFrame(
            [metric.to_dict() for metric in self.class_metrics]
        )

    def confusion_matrix_dataframe(self) -> pd.DataFrame:
        """Return the confusion matrix as a labeled DataFrame."""
        return pd.DataFrame(
            self.confusion_matrix,
            index=["true_healthy", "true_alzheimer"],
            columns=["pred_healthy", "pred_alzheimer"],
        )

    def summary_dataframe(self) -> pd.DataFrame:
        """Return global evaluation metrics as a one-row DataFrame."""
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
        """Print a readable evaluation report."""
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
        print(f"Threshold:                 {self.threshold:.4f}")

        print("\n[2] Confusion matrix")
        print("-" * 90)
        print(self.confusion_matrix_dataframe().to_string())

        print("\n[3] Metrics per class")
        print("-" * 90)
        print(self.class_metrics_dataframe().to_string(index=False))


@dataclass
class NeuralNetworkEvaluationEngineParameters:
    """Configuration parameters for neural-network evaluation."""

    macro_aggregation_method: str = "mean_logit"
    class_names: tuple[str, str] = ("Healthy", "Alzheimer")


class NeuralNetworkEvaluationEngine:
    """Evaluation engine for binary neuro-symbolic neural-network models."""

    def __init__(self, params: NeuralNetworkEvaluationEngineParameters):
        self.params = params

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader,
        threshold: float = 0.5,
        show_progress: bool = False,
    ) -> NeuralNetworkEvaluationResult:
        """Evaluate a neural network over a dataloader."""
        device = next(model.parameters()).device
        model.eval()

        y_true_list: list[torch.Tensor] = []
        y_proba_list: list[torch.Tensor] = []

        iterator = (
            tqdm(
                dataloader,
                desc=f"Evaluation threshold={threshold:.4f}",
                leave=False,
            )
            if show_progress
            else dataloader
        )

        for micro_x_raws, macro_x_feat, y_true in iterator:
            micro_x_raws = micro_x_raws.to(device)
            y_true = y_true.to(device).float()

            if micro_x_raws.ndim != 4:
                raise ValueError(
                    "Expected micro_x_raws with shape "
                    "[batch, n_micro_segments, channels, samples]. "
                    f"Got {micro_x_raws.shape}."
                )

            micro_x_raws = micro_x_raws.permute(
                1,
                0,
                2,
                3,
            ).contiguous()

            micro_logits = torch.stack(
                [
                    model(micro_x_raw).squeeze(-1)
                    for micro_x_raw in micro_x_raws
                ],
                dim=0,
            )

            macro_ad_proba = MicroLogitsToMacroProbabilityAggregator.compute(
                micro_logits=micro_logits,
                method=self.params.macro_aggregation_method,
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

        macro_precision, macro_recall, macro_f1, _ = (
            precision_recall_fscore_support(
                y_true_arr,
                y_pred_arr,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )

        weighted_precision, weighted_recall, weighted_f1, _ = (
            precision_recall_fscore_support(
                y_true_arr,
                y_pred_arr,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        )

        class_metrics = tuple(
            ClassEvaluationMetrics(
                class_name=self.params.class_names[i],
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
            macro_aggregation_method=self.params.macro_aggregation_method,
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


class NeuralNetworkBestThresholdFactory:
    """Factory used to find the best decision threshold for a neural model."""

    @staticmethod
    def find(
        model: nn.Module,
        dataloader,
        decision_metrics: str,
        evaluation_params: NeuralNetworkEvaluationEngineParameters,
        thresholds: np.ndarray | None = None,
    ) -> NeuralNetworkEvaluationResult:
        """
        Find the best threshold for a chosen metric.

        The method assumes that the threshold-to-score curve is approximately
        unimodal and uses a discrete ternary search instead of evaluating all
        thresholds.
        """
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 99)

        thresholds = np.asarray(thresholds, dtype=float)

        if thresholds.ndim != 1 or len(thresholds) == 0:
            raise ValueError("thresholds must be a non-empty 1D array.")

        engine = NeuralNetworkEvaluationEngine(evaluation_params)

        cache: dict[int, NeuralNetworkEvaluationResult] = {}

        def evaluate_index(index: int) -> tuple[float, NeuralNetworkEvaluationResult]:
            """Evaluate one threshold index and cache the result."""
            if index not in cache:
                result = engine.evaluate(
                    model=model,
                    dataloader=dataloader,
                    threshold=float(thresholds[index]),
                )

                if not hasattr(result, decision_metrics):
                    raise ValueError(
                        f"Unknown decision metric: {decision_metrics}. "
                        f"Available metrics include: "
                        f"accuracy, balanced_accuracy, macro_precision, "
                        f"macro_recall, macro_f1, weighted_precision, "
                        f"weighted_recall, weighted_f1, "
                        f"matthews_corrcoef, cohen_kappa."
                    )

                cache[index] = result

            result = cache[index]
            score = getattr(result, decision_metrics)

            if score is None:
                return -np.inf, result

            return float(score), result

        left = 0
        right = len(thresholds) - 1

        progress_bar = tqdm(
            total=max(1, int(np.ceil(np.log(len(thresholds)) / np.log(1.5)))),
            desc=f"Smart threshold search ({decision_metrics})",
        )

        while right - left > 3:
            third = (right - left) // 3

            mid1 = left + third
            mid2 = right - third

            score1, _ = evaluate_index(mid1)
            score2, _ = evaluate_index(mid2)

            progress_bar.set_postfix(
                left=f"{thresholds[left]:.4f}",
                right=f"{thresholds[right]:.4f}",
                mid1=f"{thresholds[mid1]:.4f}",
                mid2=f"{thresholds[mid2]:.4f}",
                score1=f"{score1:.4f}",
                score2=f"{score2:.4f}",
            )
            progress_bar.update(1)

            if score1 < score2:
                left = mid1
            else:
                right = mid2

        progress_bar.close()

        best_score = -np.inf
        best_result: NeuralNetworkEvaluationResult | None = None

        for index in range(left, right + 1):
            score, result = evaluate_index(index)

            if score > best_score:
                best_score = score
                best_result = result

        if best_result is None:
            raise RuntimeError(
                f"Could not find a valid threshold using metric '{decision_metrics}'."
            )

        return best_result