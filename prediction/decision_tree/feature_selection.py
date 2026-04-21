from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from features.dataset import FeaturesDataset, FeaturesDatasetSelector, SelectedFeaturesDataset
from features.name import FeatureNameHelper
from prediction.decision_tree.trainer import (
    DecisionTreeParameters,
    DecisionTreeTrainer,
    DecisionTreeClassifier
)


@dataclass(frozen=True)
class FeatureCandidate:
    """
    Représente une feature logique candidate à la sélection.

    Exemple
    -------
    - "variance"      -> plusieurs colonnes EEG
    - "cn_alpha"      -> plusieurs colonnes de connectivité
    - "subject_age"   -> une ou plusieurs colonnes metadata sujet
    """
    name: str
    columns: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("`name` must be a non-empty string.")

        if not isinstance(self.columns, list) or len(self.columns) == 0:
            raise ValueError(f"`columns` for candidate '{self.name}' must be a non-empty list.")

        if not all(isinstance(col, str) and col.strip() for col in self.columns):
            raise ValueError(
                f"All columns for candidate '{self.name}' must be non-empty strings."
            )


@dataclass(frozen=True)
class CandidateEvaluation:
    """
    Résultat d'évaluation d'un candidat ajouté à l'état courant.
    """
    candidate: FeatureCandidate
    score_mean: float
    score_std: float
    objective_value: float
    columns_after_addition: list[str]


@dataclass(frozen=True)
class FeatureSelectionResult:
    """
    Résultat final de la sélection forward.
    """
    selected_feature_names: list[str]
    selected_columns: list[str]
    best_score: float
    best_score_std: float
    best_objective_value: float
    final_datasets: tuple[SelectedFeaturesDataset, SelectedFeaturesDataset]
    final_model: DecisionTreeClassifier
    history: list[dict]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


@dataclass
class SelectionState:
    """
    État mutable de la sélection au cours de l'algorithme.
    """
    selected_candidates: list[FeatureCandidate] = field(default_factory=list)
    selected_columns: list[str] = field(default_factory=list)
    best_score: float = -np.inf
    best_score_std: float = np.nan
    best_objective_value: float = -np.inf
    history: list[dict] = field(default_factory=list)

    @property
    def selected_feature_names(self) -> list[str]:
        return [candidate.name for candidate in self.selected_candidates]

    @property
    def n_selected_features(self) -> int:
        return len(self.selected_candidates)

    @property
    def n_selected_columns(self) -> int:
        return len(self.selected_columns)

    def add_evaluation(self, evaluation: CandidateEvaluation, step: int) -> None:
        previous_best_score = self.best_score
        previous_best_score_std = self.best_score_std
        previous_best_objective_value = self.best_objective_value

        objective_improvement = (
            np.nan if np.isneginf(previous_best_objective_value)
            else evaluation.objective_value - previous_best_objective_value
        )

        score_improvement = (
            np.nan if np.isneginf(previous_best_score)
            else evaluation.score_mean - previous_best_score
        )

        self.selected_candidates.append(evaluation.candidate)
        self.selected_columns = evaluation.columns_after_addition
        self.best_score = evaluation.score_mean
        self.best_score_std = evaluation.score_std
        self.best_objective_value = evaluation.objective_value

        self.history.append(
            {
                "step": step,
                "action": "add",
                "feature": evaluation.candidate.name,
                "candidate_score_mean": evaluation.score_mean,
                "candidate_score_std": evaluation.score_std,
                "candidate_objective_value": evaluation.objective_value,
                "previous_best_score": previous_best_score,
                "previous_best_score_std": previous_best_score_std,
                "previous_best_objective_value": previous_best_objective_value,
                "score_improvement": score_improvement,
                "objective_improvement": objective_improvement,
                "n_columns_added": len(evaluation.candidate.columns),
                "n_total_columns": len(evaluation.columns_after_addition),
                "selected_features_so_far": self.selected_feature_names.copy(),
            }
        )


class FeatureCandidateBuilder:
    """
    Construit les blocs de colonnes associés à chaque feature logique.
    """

    def __init__(self, helper: FeatureNameHelper):
        self.helper = helper

    def build_one(self, feature_name: str) -> FeatureCandidate:
        if not isinstance(feature_name, str):
            raise TypeError("`feature_name` must be a string.")

        feature_name = feature_name.strip()
        if not feature_name:
            raise ValueError("`feature_name` must be a non-empty string.")

        if feature_name.startswith("cn_"):
            band_name = feature_name[len("cn_"):]
            columns = self.helper.build(cn=band_name)
        elif feature_name.startswith("subject_"):
            columns = self.helper.build(subject=feature_name)
        else:
            columns = self.helper.build(eeg=feature_name)

        columns = self._unique_preserve_order(columns)

        if not columns:
            raise ValueError(
                f"No columns were resolved for feature candidate '{feature_name}'."
            )

        return FeatureCandidate(name=feature_name, columns=columns)

    def build_many(self, feature_names: Iterable[str]) -> list[FeatureCandidate]:
        if feature_names is None:
            raise ValueError("`feature_names` cannot be None.")

        candidates = [self.build_one(name) for name in feature_names]
        self._validate_unique_names(candidates)
        return candidates

    @staticmethod
    def _unique_preserve_order(values: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(values))

    @staticmethod
    def _validate_unique_names(candidates: list[FeatureCandidate]) -> None:
        names = [candidate.name for candidate in candidates]
        duplicates = [name for name in set(names) if names.count(name) > 1]
        if duplicates:
            raise ValueError(f"Duplicate candidate names detected: {duplicates}")


class FeatureSetEvaluator:
    """
    Responsable unique de l'évaluation d'un ensemble de colonnes.
    """

    def __init__(self, trainer: DecisionTreeTrainer):
        self.trainer = trainer

    def evaluate_columns(
        self,
        dataset: FeaturesDataset,
        params: DecisionTreeParameters,
        columns: list[str],
    ) -> tuple[float, float]:
        candidate_dataset = FeaturesDatasetSelector.select(
            dataset=dataset,
            selection=columns,
        )
        score_mean, score_std = self.trainer.evaluate(
            dataset=candidate_dataset,
            params=params,
        )
        return score_mean, score_std


class PenalizedObjective:
    """
    Critère de sélection pénalisé par la variance.
    objective = mean - lambda_std * std
    """

    def __init__(self, lambda_std: float = 0.0):
        if not isinstance(lambda_std, (int, float)):
            raise TypeError("`lambda_std` must be numeric.")
        if lambda_std < 0:
            raise ValueError("`lambda_std` must be >= 0.")
        self.lambda_std = float(lambda_std)

    def compute(self, score_mean: float, score_std: float) -> float:
        return float(score_mean - self.lambda_std * score_std)


class FeatureSelectionProgressDisplay:
    """
    Affichage console sur une seule ligne, réécrite à chaque étape.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._last_message_length = 0

    def update(
        self,
        step: int,
        max_steps: int,
        n_remaining_candidates: int,
        best_score: float,
        best_score_std: float,
        best_objective_value: float,
    ) -> None:
        if not self.enabled:
            return

        best_score_str = "-inf" if np.isneginf(best_score) else f"{best_score:.6f}"
        best_std_str = "nan" if np.isnan(best_score_std) else f"{best_score_std:.6f}"
        best_obj_str = "-inf" if np.isneginf(best_objective_value) else f"{best_objective_value:.6f}"

        message = (
            f"\rFeature Selection - remaining : {n_remaining_candidates} "
            f"[mean: {best_score_str}"
            f" | std: {best_std_str}"
            f" | objective: {best_obj_str}]"
        )

        padding = max(0, self._last_message_length - len(message))
        print(message + (" " * padding), end="", flush=True)
        self._last_message_length = len(message)

    def finish(
        self,
        n_selected_features: int,
        best_score: float,
        best_score_std: float,
        best_objective_value: float,
    ) -> None:
        if not self.enabled:
            return

        best_score_str = "-inf" if np.isneginf(best_score) else f"{best_score:.6f}"
        best_std_str = "nan" if np.isnan(best_score_std) else f"{best_score_std:.6f}"
        best_obj_str = "-inf" if np.isneginf(best_objective_value) else f"{best_objective_value:.6f}"

        message = (
            f"\rFeature Selection —"
            f" — selected: {n_selected_features} "
            f"[mean: {best_score_str}"
            f" — std: {best_std_str}"
            f" — objective: {best_obj_str}]"
        )

        padding = max(0, self._last_message_length - len(message))
        print(message + (" " * padding))
        self._last_message_length = 0


class FinalModelBuilder:
    """
    Construit le dataset final réduit et entraîne le modèle final associé.
    """

    def __init__(self, trainer: DecisionTreeTrainer):
        self.trainer = trainer

    def build(
        self,
        dataset: FeaturesDataset,
        params: DecisionTreeParameters,
        selected_columns: list[str],
    ) -> tuple[SelectedFeaturesDataset, SelectedFeaturesDataset, DecisionTreeClassifier]:
        final_dataset = FeaturesDatasetSelector.select(
            dataset=dataset,
            selection=selected_columns,
        )

        train_dataset, test_dataset = final_dataset.selector.group_train_test_split()


        params_dico = params.to_dict()
        params_dico["random_state"] = self.trainer.random_state
        final_model = DecisionTreeClassifier(**params_dico)
        final_model.fit(X = train_dataset.X, y=train_dataset.y)
        

        return train_dataset, test_dataset, final_model


class ForwardSelectionEngine:
    """
    Moteur de sélection forward pure :
    à chaque étape, on ajoute le meilleur bloc restant uniquement
    s'il améliore strictement le critère pénalisé courant.
    """

    def __init__(
        self,
        evaluator: FeatureSetEvaluator,
        objective: PenalizedObjective,
        show_progress: bool = True,
    ):
        self.evaluator = evaluator
        self.objective = objective
        self.progress_display = FeatureSelectionProgressDisplay(enabled=show_progress)

    def run(
        self,
        dataset: FeaturesDataset,
        params: DecisionTreeParameters,
        candidates: list[FeatureCandidate],
        max_features: int,
    ) -> SelectionState:
        self._validate_max_features(max_features)

        if not candidates:
            raise ValueError("`candidates` cannot be empty.")

        state = SelectionState()
        remaining_candidates = candidates.copy()
        n_steps_max = min(max_features, len(candidates))

        for step in range(1, n_steps_max + 1):
            if not remaining_candidates:
                break

            self.progress_display.update(
                step=step,
                max_steps=n_steps_max,
                n_remaining_candidates=len(remaining_candidates),
                best_score=state.best_score,
                best_score_std=state.best_score_std,
                best_objective_value=state.best_objective_value,
            )

            best_evaluation = self._find_best_next_candidate(
                dataset=dataset,
                params=params,
                state=state,
                remaining_candidates=remaining_candidates,
            )

            if not self._should_accept_candidate(
                current_best_objective_value=state.best_objective_value,
                candidate_objective_value=best_evaluation.objective_value,
            ):
                break

            state.add_evaluation(best_evaluation, step=step)
            remaining_candidates = [
                candidate
                for candidate in remaining_candidates
                if candidate.name != best_evaluation.candidate.name
            ]

        self.progress_display.finish(
            n_selected_features=state.n_selected_features,
            best_score=state.best_score,
            best_score_std=state.best_score_std,
            best_objective_value=state.best_objective_value,
        )

        return state

    def _find_best_next_candidate(
        self,
        dataset: FeaturesDataset,
        params: DecisionTreeParameters,
        state: SelectionState,
        remaining_candidates: list[FeatureCandidate],
    ) -> CandidateEvaluation:
        best_evaluation: CandidateEvaluation | None = None

        for candidate in remaining_candidates:
            evaluation = self._evaluate_candidate_addition(
                dataset=dataset,
                params=params,
                state=state,
                candidate=candidate,
            )

            if (
                best_evaluation is None
                or evaluation.objective_value > best_evaluation.objective_value
            ):
                best_evaluation = evaluation

        if best_evaluation is None:
            raise RuntimeError("Internal error: no candidate evaluation was produced.")

        return best_evaluation

    def _evaluate_candidate_addition(
        self,
        dataset: FeaturesDataset,
        params: DecisionTreeParameters,
        state: SelectionState,
        candidate: FeatureCandidate,
    ) -> CandidateEvaluation:
        candidate_columns = self._merge_columns(
            state.selected_columns,
            candidate.columns,
        )

        score_mean, score_std = self.evaluator.evaluate_columns(
            dataset=dataset,
            params=params,
            columns=candidate_columns,
        )

        objective_value = self.objective.compute(
            score_mean=score_mean,
            score_std=score_std,
        )

        return CandidateEvaluation(
            candidate=candidate,
            score_mean=score_mean,
            score_std=score_std,
            objective_value=objective_value,
            columns_after_addition=candidate_columns,
        )

    @staticmethod
    def _should_accept_candidate(
        current_best_objective_value: float,
        candidate_objective_value: float,
    ) -> bool:
        if np.isneginf(current_best_objective_value):
            return True
        return candidate_objective_value > current_best_objective_value

    @staticmethod
    def _merge_columns(
        current_columns: Iterable[str],
        new_columns: Iterable[str],
    ) -> list[str]:
        return list(dict.fromkeys([*current_columns, *new_columns]))

    @staticmethod
    def _validate_max_features(max_features: int) -> None:
        if not isinstance(max_features, int):
            raise TypeError("`max_features` must be an integer.")
        if max_features <= 0:
            raise ValueError("`max_features` must be > 0.")


class FeatureSelector:
    """
    Façade haut niveau pour la sélection de features.
    """

    def __init__(
        self,
        trainer: DecisionTreeTrainer,
        helper: FeatureNameHelper,
        lambda_std:int,
    ):
        self.trainer = trainer
        self.helper = helper
        self.lambda_std = lambda_std

        self.candidate_builder = FeatureCandidateBuilder(helper)
        self.evaluator = FeatureSetEvaluator(trainer)
        self.final_model_builder = FinalModelBuilder(trainer)
        self._last_result: FeatureSelectionResult | None = None

    def select(
        self,
        dataset: FeaturesDataset,
        params: DecisionTreeParameters,
        max_features: int = 20,
        show_progress: bool = True,
    ) -> FeatureSelectionResult:
        candidates = self._build_candidates_from_dataset(dataset)

        engine = ForwardSelectionEngine(
            evaluator=self.evaluator,
            objective=PenalizedObjective(lambda_std=self.lambda_std),
            show_progress=show_progress,
        )

        state = engine.run(
            dataset=dataset,
            params=params,
            candidates=candidates,
            max_features=max_features,
        )

        train_dataset, test_dataset, final_model = self.final_model_builder.build(
            dataset=dataset,
            params=params,
            selected_columns=state.selected_columns,
        )

        result = FeatureSelectionResult(
            selected_feature_names=state.selected_feature_names,
            selected_columns=state.selected_columns,
            best_score=state.best_score,
            best_score_std=state.best_score_std,
            best_objective_value=state.best_objective_value,
            final_datasets=(train_dataset, test_dataset),
            final_model=final_model,
            history=state.history,
        )

        self._last_result = result
        return result

    def forward_selection(
        self,
        dataset: FeaturesDataset,
        params: DecisionTreeParameters,
        max_features: int = 20,
        show_progress: bool = True,
    ) -> tuple[list[str], list[str], float, float, float, Any]:
        result = self.select(
            dataset=dataset,
            params=params,
            max_features=max_features,
            lambda_std=self.lambda_std,
            show_progress=show_progress,
        )
        return (
            result.selected_feature_names,
            result.selected_columns,
            result.best_score,
            result.best_score_std,
            result.best_objective_value,
            result.final_model,
        )

    def get_results_dataframe(self) -> pd.DataFrame:
        if self._last_result is None:
            return pd.DataFrame()
        return self._last_result.to_dataframe()

    def get_last_result(self) -> FeatureSelectionResult | None:
        return self._last_result

    def _build_candidates_from_dataset(self, dataset: FeaturesDataset) -> list[FeatureCandidate]:
        feature_names = dataset.feature_names

        if feature_names is None:
            raise ValueError("`dataset.feature_names` cannot be None.")

        if len(feature_names) == 0:
            raise ValueError("`dataset.feature_names` is empty, nothing to select.")

        return self.candidate_builder.build_many(feature_names)