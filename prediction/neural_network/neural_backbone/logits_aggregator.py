from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MacroProbabilityAggregationParams:
    method: str = "mean_logit"
    threshold: float = 0.5
    temperature: float = 1.0
    eps: float = 1e-6


@dataclass(frozen=True)
class MicroSupervisedLossAggregationParams:
    method: str = "mean_micro_bce"
    macro_probability_method: str = "mean_logit"
    reduction: str = "mean"
    eps: float = 1e-6


class MicroLogitsToMacroProbabilityAggregator:
    """
    Agrège des micro_logits issus de segments EEG micro en une
    probabilité macro.

    Formats supportés
    -----------------
    [n_micro_segments]
    [batch_size, n_micro_segments]
    [n_micro_segments, batch_size]

    Output
    ------
    Tensor de shape [batch_size]
    """

    @staticmethod
    def compute(
        micro_logits: torch.Tensor,
        params: MacroProbabilityAggregationParams | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        params = params or MacroProbabilityAggregationParams()

        micro_logits = (
            MicroLogitsToMacroProbabilityAggregator._ensure_batch_first(
                micro_logits,
                batch_size=batch_size,
            )
        )

        micro_probas_ad = torch.sigmoid(micro_logits)

        if params.method == "mean_logit":
            return torch.sigmoid(micro_logits.mean(dim=1))

        if params.method == "mean_probability":
            return micro_probas_ad.mean(dim=1)

        if params.method == "median_probability":
            return micro_probas_ad.median(dim=1).values

        if params.method == "max_probability":
            return micro_probas_ad.max(dim=1).values

        if params.method == "min_probability":
            return micro_probas_ad.min(dim=1).values

        if params.method == "noisy_or":
            micro_probas_ad = torch.clamp(
                micro_probas_ad,
                params.eps,
                1.0 - params.eps,
            )

            return 1.0 - torch.prod(
                1.0 - micro_probas_ad,
                dim=1,
            )

        if params.method == "majority_vote":
            return (
                (micro_probas_ad >= params.threshold)
                .float()
                .mean(dim=1)
            )

        if params.method == "soft_majority_vote":
            soft_votes_ad = torch.sigmoid(
                (micro_probas_ad - params.threshold)
                / params.temperature
            )

            return soft_votes_ad.mean(dim=1)

        if params.method == "confidence_weighted_probability":
            weights = torch.abs(micro_probas_ad - 0.5)

            weights = weights / (
                weights.sum(dim=1, keepdim=True)
                + params.eps
            )

            return (weights * micro_probas_ad).sum(dim=1)

        if params.method == "attention_probability":
            weights = torch.softmax(
                torch.abs(micro_logits)
                / params.temperature,
                dim=1,
            )

            return (weights * micro_probas_ad).sum(dim=1)

        raise ValueError(
            f"Unknown macro probability aggregation method: "
            f"{params.method}"
        )

    @staticmethod
    def _ensure_batch_first(
        micro_logits: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        Garantit un format :
            [batch_size, n_micro_segments]
        """

        if micro_logits.ndim == 1:
            return micro_logits.unsqueeze(0)

        if micro_logits.ndim != 2:
            raise ValueError(
                "micro_logits must have shape "
                "[n_micro_segments], "
                "[batch_size, n_micro_segments], "
                "or [n_micro_segments, batch_size]. "
                f"Got {micro_logits.shape}."
            )

        if batch_size is None:
            return micro_logits

        # Déjà batch-first
        if micro_logits.shape[0] == batch_size:
            return micro_logits

        # Format [n_micro_segments, batch_size]
        if micro_logits.shape[1] == batch_size:
            return micro_logits.transpose(0, 1)

        raise ValueError(
            f"Cannot infer batch dimension from "
            f"micro_logits.shape={micro_logits.shape} "
            f"and batch_size={batch_size}."
        )


class MicroLogitsSupervisedLossAggregator:
    """
    Calcule une loss supervisée à partir de micro_logits
    et d'un label macro.

    Le label macro est partagé par tous les micro-segments.
    """

    @staticmethod
    def compute(
        micro_logits: torch.Tensor,
        macro_y_true: torch.Tensor,
        params: MicroSupervisedLossAggregationParams | None = None,
    ) -> torch.Tensor:
        params = params or MicroSupervisedLossAggregationParams()

        if not torch.is_tensor(macro_y_true):
            macro_y_true = torch.tensor(macro_y_true)

        macro_y_true = macro_y_true.float().to(
            micro_logits.device
        )

        if macro_y_true.ndim == 0:
            macro_y_true = macro_y_true.unsqueeze(0)

        if macro_y_true.ndim != 1:
            raise ValueError(
                "macro_y_true must have shape [] "
                "or [batch_size]. "
                f"Got {macro_y_true.shape}."
            )

        batch_size = macro_y_true.shape[0]

        micro_logits = (
            MicroLogitsToMacroProbabilityAggregator._ensure_batch_first(
                micro_logits,
                batch_size=batch_size,
            )
        )

        macro_y_true_expanded = (
            macro_y_true.unsqueeze(1)
            .expand_as(micro_logits)
        )

        # ==========================================================
        # Mean micro BCE
        # ==========================================================

        if params.method == "mean_micro_bce":
            micro_bce = F.binary_cross_entropy_with_logits(
                micro_logits,
                macro_y_true_expanded,
                reduction="none",
            )

            return micro_bce.mean()

        # ==========================================================
        # Sum micro BCE
        # ==========================================================

        if params.method == "sum_micro_bce":
            micro_bce = F.binary_cross_entropy_with_logits(
                micro_logits,
                macro_y_true_expanded,
                reduction="none",
            )

            return micro_bce.sum()

        # ==========================================================
        # Max micro BCE
        # ==========================================================

        if params.method == "max_micro_bce":
            micro_bce = F.binary_cross_entropy_with_logits(
                micro_logits,
                macro_y_true_expanded,
                reduction="none",
            )

            return micro_bce.max(dim=1).values.mean()

        # ==========================================================
        # Min micro BCE
        # ==========================================================

        if params.method == "min_micro_bce":
            micro_bce = F.binary_cross_entropy_with_logits(
                micro_logits,
                macro_y_true_expanded,
                reduction="none",
            )

            return micro_bce.min(dim=1).values.mean()

        # ==========================================================
        # Macro BCE
        # ==========================================================

        if params.method == "macro_bce":
            macro_proba_ad = (
                MicroLogitsToMacroProbabilityAggregator.compute(
                    micro_logits=micro_logits,
                    params=MacroProbabilityAggregationParams(
                        method=params.macro_probability_method,
                        eps=params.eps,
                    ),
                    batch_size=batch_size,
                )
            )

            macro_proba_ad = torch.clamp(
                macro_proba_ad,
                params.eps,
                1.0 - params.eps,
            )

            return F.binary_cross_entropy(
                macro_proba_ad,
                macro_y_true,
                reduction=params.reduction,
            )

        # ==========================================================
        # Macro BCE with logits
        # ==========================================================

        if params.method == "macro_bce_with_logits":
            if (
                params.macro_probability_method
                == "mean_logit"
            ):
                macro_logit = micro_logits.mean(dim=1)

            else:
                macro_proba_ad = (
                    MicroLogitsToMacroProbabilityAggregator.compute(
                        micro_logits=micro_logits,
                        params=MacroProbabilityAggregationParams(
                            method=params.macro_probability_method,
                            eps=params.eps,
                        ),
                        batch_size=batch_size,
                    )
                )

                macro_logit = (
                    MicroLogitsSupervisedLossAggregator._safe_logit(
                        macro_proba_ad,
                        eps=params.eps,
                    )
                )

            return F.binary_cross_entropy_with_logits(
                macro_logit,
                macro_y_true,
                reduction=params.reduction,
            )

        # ==========================================================
        # Confidence weighted micro BCE
        # ==========================================================

        if params.method == "confidence_weighted_micro_bce":
            micro_bce = F.binary_cross_entropy_with_logits(
                micro_logits,
                macro_y_true_expanded,
                reduction="none",
            )

            micro_probas_ad = torch.sigmoid(micro_logits)

            weights = torch.abs(micro_probas_ad - 0.5)

            weights = weights / (
                weights.sum(dim=1, keepdim=True)
                + params.eps
            )

            return (
                (weights * micro_bce)
                .sum(dim=1)
                .mean()
            )

        # ==========================================================
        # Focal micro BCE
        # ==========================================================

        if params.method == "focal_micro_bce":
            gamma = 2.0

            micro_bce = F.binary_cross_entropy_with_logits(
                micro_logits,
                macro_y_true_expanded,
                reduction="none",
            )

            micro_probas_ad = torch.sigmoid(micro_logits)

            p_t = (
                macro_y_true_expanded
                * micro_probas_ad
                + (1.0 - macro_y_true_expanded)
                * (1.0 - micro_probas_ad)
            )

            focal_weight = (1.0 - p_t) ** gamma

            return (
                focal_weight * micro_bce
            ).mean()

        raise ValueError(
            f"Unknown micro supervised loss method: "
            f"{params.method}"
        )

    @staticmethod
    def _safe_logit(
        proba: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        proba = torch.clamp(
            proba,
            eps,
            1.0 - eps,
        )

        return torch.logit(proba)