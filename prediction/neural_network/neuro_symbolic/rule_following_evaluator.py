from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rules.differentiable_rule import (
    DifferentiableDecisionRule,
    DifferentiableRuleCandidateFactory,
    TruthDegreeEngine,
)

from prediction.neural_network.neural_backbone.logits_aggregator import (
    MicroLogitsToMacroProbabilityAggregator,
)


@dataclass
class RuleEvaluationResults:
    rule_agreement: float
    rule_compliance: float

    logical_agreement: float
    logical_alignment: float
    logical_mse: float

    active_count: float
    truth_mass: float
    n_rules: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class RuleEvaluator:

    def __init__(
        self,
        threshold: float = 0.5,
        rule_activation_threshold: float = 0.8,
        macro_aggregation_method: str = "mean_probability",
        eps: float = 1e-8,
    ) -> None:
        self.threshold = threshold
        self.rule_activation_threshold = rule_activation_threshold
        self.macro_aggregation_method = macro_aggregation_method
        self.eps = eps

    def evaluation(
        self,
        model: nn.Module,
        rules: list[DifferentiableDecisionRule],
        dataloader: DataLoader,
    ) -> RuleEvaluationResults:

        if len(rules) == 0:
            raise ValueError("`rules` cannot be empty.")

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = model.to(device)
        model.eval()

        agreement_numerator = 0.0
        agreement_denominator = 0.0

        compliance_numerator = 0.0
        compliance_denominator = 0.0

        logical_agreement_numerator = 0.0
        logical_agreement_denominator = 0.0

        logical_alignment_numerator = 0.0
        logical_mse_numerator = 0.0
        logical_denominator = 0.0

        with torch.no_grad():
            for micro_x_raws, macro_x_feat, _ in tqdm(
                dataloader,
                desc="Rule evaluation",
                leave=False,
            ):
                micro_x_raws = micro_x_raws.to(device)
                macro_x_feat = macro_x_feat.to(device)

                macro_ad_proba = self._predict_macro_ad_probability(
                    model=model,
                    micro_x_raws=micro_x_raws,
                )

                for rule in rules:
                    rule_weight = float(rule.score)

                    truth_degree = self._compute_truth_degree(
                        rule=rule,
                        macro_x_feat=macro_x_feat,
                    )

                    rule_class_proba = self._get_rule_class_probability(
                        macro_ad_proba=macro_ad_proba,
                        predicted_class=rule.predicted_class,
                    )

                    rule_active = (
                        truth_degree >= self.rule_activation_threshold
                    ).float()

                    rule_followed = (
                        rule_class_proba >= self.threshold
                    ).float()

                    # Anciennes métriques :
                    # uniquement quand la règle est fortement vraie.
                    agreement_numerator += rule_weight * float(
                        (rule_active * rule_followed).sum().item()
                    )

                    agreement_denominator += rule_weight * float(
                        rule_active.sum().item()
                    )

                    compliance_numerator += rule_weight * float(
                        (truth_degree * rule_class_proba).sum().item()
                    )

                    compliance_denominator += rule_weight * float(
                        truth_degree.sum().item()
                    )

                    # Nouvelle interprétation :
                    # truth_degree proche de 1 -> preuve pour la classe de la règle
                    # truth_degree proche de 0 -> preuve pour la classe opposée.
                    logical_ad_proba = self._get_logical_ad_probability(
                        truth_degree=truth_degree,
                        predicted_class=rule.predicted_class,
                    )

                    nn_ad_pred = (
                        macro_ad_proba >= self.threshold
                    ).float()

                    logical_ad_pred = (
                        logical_ad_proba >= self.threshold
                    ).float()

                    logical_agreement_numerator += rule_weight * float(
                        (nn_ad_pred == logical_ad_pred).float().sum().item()
                    )

                    logical_agreement_denominator += rule_weight * float(
                        logical_ad_pred.numel()
                    )

                    absolute_error = torch.abs(
                        macro_ad_proba - logical_ad_proba
                    )

                    squared_error = (
                        macro_ad_proba - logical_ad_proba
                    ) ** 2

                    logical_alignment_numerator += rule_weight * float(
                        (1.0 - absolute_error).sum().item()
                    )

                    logical_mse_numerator += rule_weight * float(
                        squared_error.sum().item()
                    )

                    logical_denominator += rule_weight * float(
                        logical_ad_proba.numel()
                    )

        rule_agreement = agreement_numerator / max(
            agreement_denominator,
            self.eps,
        )

        rule_compliance = compliance_numerator / max(
            compliance_denominator,
            self.eps,
        )

        logical_agreement = (
            logical_agreement_numerator
            / max(logical_agreement_denominator, self.eps)
        )

        logical_alignment = (
            logical_alignment_numerator
            / max(logical_denominator, self.eps)
        )

        logical_mse = (
            logical_mse_numerator
            / max(logical_denominator, self.eps)
        )

        return RuleEvaluationResults(
            rule_agreement=float(rule_agreement),
            rule_compliance=float(rule_compliance),
            logical_agreement=float(logical_agreement),
            logical_alignment=float(logical_alignment),
            logical_mse=float(logical_mse),
            active_count=float(agreement_denominator),
            truth_mass=float(compliance_denominator),
            n_rules=len(rules),
        )

    def _predict_macro_ad_probability(
        self,
        model: nn.Module,
        micro_x_raws: torch.Tensor,
    ) -> torch.Tensor:

        micro_logits = self._forward_micro_segments(
            model=model,
            micro_x_raws=micro_x_raws,
        )

        return MicroLogitsToMacroProbabilityAggregator.compute(
            micro_logits,
            method=self.macro_aggregation_method,
        )

    @staticmethod
    def _forward_micro_segments(
        model: nn.Module,
        micro_x_raws: torch.Tensor,
    ) -> torch.Tensor:

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

        return micro_logits

    @staticmethod
    def _compute_truth_degree(
        rule: DifferentiableDecisionRule,
        macro_x_feat: torch.Tensor,
    ) -> torch.Tensor:

        candidate = DifferentiableRuleCandidateFactory.from_tensor(
            macro_x_feat
        )

        return TruthDegreeEngine.compute(
            rule=rule,
            candidate=candidate,
        )

    @staticmethod
    def _get_rule_class_probability(
        macro_ad_proba: torch.Tensor,
        predicted_class: str,
    ) -> torch.Tensor:

        if predicted_class == "Alzheimer":
            return macro_ad_proba

        if predicted_class == "Healthy":
            return 1.0 - macro_ad_proba

        raise ValueError(
            f"Unsupported predicted_class: {predicted_class}"
        )

    @staticmethod
    def _get_logical_ad_probability(
        truth_degree: torch.Tensor,
        predicted_class: str,
    ) -> torch.Tensor:
        """
        Convertit une règle en probabilité logique d'Alzheimer.

        Si règle => Alzheimer :
            truth_degree = 1 signifie Alzheimer
            truth_degree = 0 signifie Healthy
            donc P_logic(AD) = truth_degree

        Si règle => Healthy :
            truth_degree = 1 signifie Healthy
            truth_degree = 0 signifie Alzheimer
            donc P_logic(AD) = 1 - truth_degree
        """

        if predicted_class == "Alzheimer":
            return truth_degree

        if predicted_class == "Healthy":
            return 1.0 - truth_degree

        raise ValueError(
            f"Unsupported predicted_class: {predicted_class}"
        )