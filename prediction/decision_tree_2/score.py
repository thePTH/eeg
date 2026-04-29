from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, cross_val_score

from features.dataset import SelectedFeaturesDataset
from .base import DecisionTree, TrainedDecisionTree


@dataclass(frozen=True, slots=True)
class DecisionTreeScoreResult:
    mean: float
    std: float
    scoring: str

    def adjusted_score(self, lambda_std:float):
        return self.mean - lambda_std*self.std


from dataclasses import dataclass
from typing import Any
import pandas as pd


@dataclass(frozen=True, slots=True)
class PerClassMetricResult:
    mean_by_class: dict[Any, float]
    std_by_class: dict[Any, float]
    metric_name: str

    def to_dataframe(self) -> pd.DataFrame:

        df = pd.DataFrame({
            f"{self.metric_name}_mean": self.mean_by_class,
            f"{self.metric_name}_std": self.std_by_class,

        })

        df.index.name = "label"
        return df

    def to_series(self) -> pd.Series:
        """
        Convertit les moyennes par classe en Series pandas.
        """
        return pd.Series(
            self.mean_by_class,
            name=self.metric_name,
        )


import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConfusionMatrixResult:
    labels: list[Any]
    mean: np.ndarray
    std: np.ndarray
    normalize: str | None

    def to_dataframe(self, include_std: bool = False) -> pd.DataFrame:
        
        if include_std:
            formatted = np.empty_like(self.mean, dtype=object)

            for i in range(self.mean.shape[0]):
                for j in range(self.mean.shape[1]):
                    formatted[i, j] = f"{self.mean[i, j]:.3f} ± {self.std[i, j]:.3f}"

            return pd.DataFrame(
                formatted,
                index=pd.Index(self.labels, name="true"),
                columns=pd.Index(self.labels, name="pred"),
            )

        return pd.DataFrame(
            self.mean,
            index=pd.Index(self.labels, name="true"),
            columns=pd.Index(self.labels, name="pred"),
        )


@dataclass(frozen=True, slots=True)
class DecisionTreeDetailedScoreResult:
    accuracy: DecisionTreeScoreResult
    balanced_accuracy: DecisionTreeScoreResult
    recall_by_class: PerClassMetricResult
    precision_by_class: PerClassMetricResult
    f1_by_class: PerClassMetricResult
    confusion_matrix: ConfusionMatrixResult

    def report(self) -> None:
        """
        Affiche un rapport clair et lisible des performances du modèle.
        Conçu pour console / notebook.
        """

        print("\n" + "=" * 50)
        print("🌳 DECISION TREE – FINAL TEST REPORT")
        print("=" * 50)

        print("\n📊 Global metrics")
        print("-" * 50)
        print(f"Accuracy          : {self.accuracy.mean:.4f}")
        print(f"Balanced accuracy : {self.balanced_accuracy.mean:.4f}")

        print("\n📊 Recall by class")
        print("-" * 50)
        print(self.recall_by_class.to_series().to_frame("recall"))

        print("\n📊 Precision by class")
        print("-" * 50)
        print(self.precision_by_class.to_series().to_frame("precision"))

        print("\n📊 F1-score by class")
        print("-" * 50)
        print(self.f1_by_class.to_series().to_frame("f1"))

        print("\n📊 Confusion matrix")
        print("-" * 50)
        print(self.confusion_matrix.to_dataframe())

        print("\n" + "=" * 50 + "\n")


class DecisionTreeScoreEngine:
    def __init__(self, n_splits: int = 5, scoring: str = "balanced_accuracy"):
        if n_splits < 2:
            raise ValueError("`n_splits` must be at least 2.")

        if not isinstance(scoring, str) or not scoring.strip():
            raise ValueError("`scoring` must be a non-empty string.")

        self.n_splits = n_splits
        self.scoring = scoring

    def score(
        self,
        decision_tree: DecisionTree,
        dataset: SelectedFeaturesDataset,
    ) -> DecisionTreeScoreResult:
        X = dataset.X
        y = dataset.y
        groups = dataset.groups

        cv = GroupKFold(n_splits=self.n_splits)

        scores = cross_val_score(
            estimator=decision_tree.classifier,
            X=X,
            y=y,
            groups=groups,
            cv=cv,
            scoring=self.scoring,
            n_jobs=-1,
    )

        return DecisionTreeScoreResult(
            mean=scores.mean(),
            std=scores.std(),
            scoring=self.scoring,
        )

    def recall_by_class(
        self,
        decision_tree: DecisionTree,
        dataset: SelectedFeaturesDataset,
    ) -> PerClassMetricResult:
        return self._score_metric_by_class(
            decision_tree=decision_tree,
            dataset=dataset,
            metric_name="recall",
        )

    def precision_by_class(
        self,
        decision_tree: DecisionTree,
        dataset: SelectedFeaturesDataset,
    ) -> PerClassMetricResult:
        return self._score_metric_by_class(
            decision_tree=decision_tree,
            dataset=dataset,
            metric_name="precision",
        )

    def f1_by_class(
        self,
        decision_tree: DecisionTree,
        dataset: SelectedFeaturesDataset,
    ) -> PerClassMetricResult:
        return self._score_metric_by_class(
            decision_tree=decision_tree,
            dataset=dataset,
            metric_name="f1",
        )

    def confusion_matrix(
        self,
        decision_tree: DecisionTree,
        dataset: SelectedFeaturesDataset,
        normalize: str | None = None,
    ) -> ConfusionMatrixResult:
        X = dataset.X
        y = dataset.y
        groups = dataset.groups

        labels = list(np.unique(y))
        cv = GroupKFold(n_splits=self.n_splits)

        matrices: list[np.ndarray] = []

        for train_idx, test_idx in cv.split(X, y, groups):
            train_dataset_fold = dataset.select_rows(train_idx)
            test_dataset_fold = dataset.select_rows(test_idx)

            trained_tree = decision_tree.train(train_dataset_fold)
            y_pred = trained_tree.classifier.predict(test_dataset_fold.X)

            cm = confusion_matrix(
                y_true=test_dataset_fold.y,
                y_pred=y_pred,
                labels=labels,
                normalize=normalize,
            )
            matrices.append(cm)

        matrices_array = np.asarray(matrices, dtype=float)

        return ConfusionMatrixResult(
            labels=labels,
            mean=np.mean(matrices_array, axis=0),
            std=np.std(matrices_array, axis=0),
            normalize=normalize,
        )

    def full_scores(
        self,
        decision_tree: DecisionTree,
        dataset: SelectedFeaturesDataset,
    ) -> DecisionTreeDetailedScoreResult:
        return DecisionTreeDetailedScoreResult(
            accuracy=self._global_score(
                decision_tree=decision_tree,
                dataset=dataset,
                scoring="accuracy",
            ),
            balanced_accuracy=self._global_score(
                decision_tree=decision_tree,
                dataset=dataset,
                scoring="balanced_accuracy",
            ),
            recall_by_class=self.recall_by_class(
                decision_tree=decision_tree,
                dataset=dataset,
            ),
            precision_by_class=self.precision_by_class(
                decision_tree=decision_tree,
                dataset=dataset,
            ),
            f1_by_class=self.f1_by_class(
                decision_tree=decision_tree,
                dataset=dataset,
            ),
            confusion_matrix=self.confusion_matrix(
                decision_tree=decision_tree,
                dataset=dataset,
                normalize="true",
            ),
        )

    def classification_report(
        self,
        decision_tree: DecisionTree,
        dataset: SelectedFeaturesDataset,
    ) -> pd.DataFrame:
        y_true_all, y_pred_all = self._cross_validated_predictions(
            decision_tree=decision_tree,
            dataset=dataset,
        )

        report = classification_report(
            y_true=y_true_all,
            y_pred=y_pred_all,
            output_dict=True,
            zero_division=0,
        )

        return pd.DataFrame(report).T

    def _global_score(
        self,
        decision_tree: DecisionTree,
        dataset: SelectedFeaturesDataset,
        scoring: str,
    ) -> DecisionTreeScoreResult:
        X = dataset.X
        y = dataset.y
        groups = dataset.groups

        cv = GroupKFold(n_splits=self.n_splits)

        scores = cross_val_score(
            estimator=decision_tree.classifier,
            X=X,
            y=y,
            groups=groups,
            cv=cv,
            scoring=scoring,
        )

        return DecisionTreeScoreResult(
            mean=float(np.mean(scores)),
            std=float(np.std(scores)),
            scoring=scoring,
        )

    def _score_metric_by_class(
        self,
        decision_tree: DecisionTree,
        dataset: SelectedFeaturesDataset,
        metric_name: str,
    ) -> PerClassMetricResult:
        X = dataset.X
        y = dataset.y
        groups = dataset.groups

        cv = GroupKFold(n_splits=self.n_splits)
        labels = list(np.unique(y))

        scores_by_class: dict[Any, list[float]] = {label: [] for label in labels}

        for train_idx, test_idx in cv.split(X, y, groups):
            train_dataset_fold = dataset.select_rows(train_idx)
            test_dataset_fold = dataset.select_rows(test_idx)

            trained_tree = decision_tree.train(train_dataset_fold)
            y_pred = trained_tree.classifier.predict(test_dataset_fold.X)

            if metric_name == "recall":
                values = recall_score(
                    y_true=test_dataset_fold.y,
                    y_pred=y_pred,
                    labels=labels,
                    average=None,
                    zero_division=0,
                )
            elif metric_name == "precision":
                values = precision_score(
                    y_true=test_dataset_fold.y,
                    y_pred=y_pred,
                    labels=labels,
                    average=None,
                    zero_division=0,
                )
            elif metric_name == "f1":
                values = f1_score(
                    y_true=test_dataset_fold.y,
                    y_pred=y_pred,
                    labels=labels,
                    average=None,
                    zero_division=0,
                )
            else:
                raise ValueError(
                    "`metric_name` must be one of: 'recall', 'precision', 'f1'."
                )

            for label, value in zip(labels, values):
                scores_by_class[label].append(float(value))

        return PerClassMetricResult(
            mean_by_class={
                label: float(np.mean(values))
                for label, values in scores_by_class.items()
            },
            std_by_class={
                label: float(np.std(values))
                for label, values in scores_by_class.items()
            },
            metric_name=metric_name,
        )

    def _cross_validated_predictions(
        self,
        decision_tree: DecisionTree,
        dataset: SelectedFeaturesDataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = dataset.X
        y = dataset.y
        groups = dataset.groups

        cv = GroupKFold(n_splits=self.n_splits)

        y_true_all: list[Any] = []
        y_pred_all: list[Any] = []

        for train_idx, test_idx in cv.split(X, y, groups):
            train_dataset_fold = dataset.select_rows(train_idx)
            test_dataset_fold = dataset.select_rows(test_idx)

            trained_tree = decision_tree.train(train_dataset_fold)
            y_pred = trained_tree.classifier.predict(test_dataset_fold.X)

            y_true_all.extend(test_dataset_fold.y.to_list())
            y_pred_all.extend(list(y_pred))

        return np.asarray(y_true_all), np.asarray(y_pred_all)
    

    def evaluate_trained_tree(
    self,
    trained_decision_tree: TrainedDecisionTree,
    test_dataset: SelectedFeaturesDataset,
    normalize: str | None = "true",
) -> DecisionTreeDetailedScoreResult:
        """
        Évalue un arbre déjà entraîné sur un dataset de test indépendant.

        Contrairement à `full_scores`, cette méthode ne fait pas de cross-validation.
        Elle suppose que `trained_decision_tree` est déjà entraîné.
        """
        if trained_decision_tree is None:
            raise ValueError("`trained_decision_tree` cannot be None.")

        if test_dataset is None:
            raise ValueError("`test_dataset` cannot be None.")

        if not hasattr(trained_decision_tree, "classifier"):
            raise TypeError("`trained_decision_tree` must have a `classifier` attribute.")

        classifier = trained_decision_tree.classifier

        if not hasattr(classifier, "predict"):
            raise TypeError("`trained_decision_tree.classifier` must implement `predict`.")

        X_test = test_dataset.X
        y_true = test_dataset.y

        if X_test is None or y_true is None:
            raise ValueError("`test_dataset.X` and `test_dataset.y` cannot be None.")

        if len(X_test) != len(y_true):
            raise ValueError(
                f"Inconsistent test dataset sizes: len(X_test)={len(X_test)} "
                f"but len(y_true)={len(y_true)}."
            )

        if len(y_true) == 0:
            raise ValueError("`test_dataset` must contain at least one sample.")

        y_pred = classifier.predict(X_test)

        labels = list(np.unique(np.concatenate([
            np.asarray(y_true),
            np.asarray(y_pred),
        ])))

        accuracy_value = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

        balanced_recall_values = recall_score(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )

        balanced_accuracy_value = float(np.mean(balanced_recall_values))

        recall_values = recall_score(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )

        precision_values = precision_score(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )

        f1_values = f1_score(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )

        cm = confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            normalize=normalize,
        )

        zero_std_cm = np.zeros_like(cm, dtype=float)

        return DecisionTreeDetailedScoreResult(
            accuracy=DecisionTreeScoreResult(
                mean=accuracy_value,
                std=0.0,
                scoring="accuracy",
            ),
            balanced_accuracy=DecisionTreeScoreResult(
                mean=balanced_accuracy_value,
                std=0.0,
                scoring="balanced_accuracy",
            ),
            recall_by_class=PerClassMetricResult(
                mean_by_class={
                    label: float(value)
                    for label, value in zip(labels, recall_values)
                },
                std_by_class={
                    label: 0.0
                    for label in labels
                },
                metric_name="recall",
            ),
            precision_by_class=PerClassMetricResult(
                mean_by_class={
                    label: float(value)
                    for label, value in zip(labels, precision_values)
                },
                std_by_class={
                    label: 0.0
                    for label in labels
                },
                metric_name="precision",
            ),
            f1_by_class=PerClassMetricResult(
                mean_by_class={
                    label: float(value)
                    for label, value in zip(labels, f1_values)
                },
                std_by_class={
                    label: 0.0
                    for label in labels
                },
                metric_name="f1",
            ),
            confusion_matrix=ConfusionMatrixResult(
                labels=labels,
                mean=np.asarray(cm, dtype=float),
                std=zero_std_cm,
                normalize=normalize,
            ),
        )