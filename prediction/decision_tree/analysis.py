from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .base import TrainedDecisionTree


# =============================================================================
# Domain objects
# =============================================================================

@dataclass(frozen=True, slots=True)
class ClassProbabilityDistribution:
    """Class-probability distribution stored at a decision-tree node."""

    probabilities: dict[Any, float]

    @property
    def predicted_class(self) -> Any:
        """Return the class with the highest probability."""
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def predicted_probability(self) -> float:
        """Return the probability of the predicted class."""
        return self.probabilities[self.predicted_class]

    def probability_of(self, class_name: Any) -> float:
        """Return the probability of a given class."""
        return self.probabilities.get(class_name, 0.0)

    def to_dict(self) -> dict[Any, float]:
        """Return the distribution as a dictionary."""
        return dict(self.probabilities)


@dataclass(frozen=True, slots=True)
class LogicalCondition:
    """Human-readable logical condition extracted from a decision-tree split."""

    feature_name: str
    operator: str
    threshold: float

    def __str__(self) -> str:
        """Return the condition as a readable string."""
        return f"{self.feature_name} {self.operator} {self.threshold:.6g}"


@dataclass(frozen=True, slots=True)
class LogicalRule:
    """Logical rule represented as a conjunction of conditions."""

    conditions: tuple[LogicalCondition, ...]

    @property
    def is_root(self) -> bool:
        """Return whether the rule corresponds to the tree root."""
        return len(self.conditions) == 0

    def __str__(self) -> str:
        """Return the rule as a readable string."""
        if self.is_root:
            return "ROOT"

        return " AND ".join(str(condition) for condition in self.conditions)

    def extend(self, condition: LogicalCondition) -> LogicalRule:
        """Return a new rule extended with one additional condition."""
        return LogicalRule(self.conditions + (condition,))


@dataclass(frozen=True, slots=True)
class NodeProbabilityAnalysis:
    """Probability analysis associated with one decision-tree node."""

    class_counts: dict[Any, float]
    distribution: ClassProbabilityDistribution

    @property
    def predicted_class(self) -> Any:
        """Return the predicted class at the node."""
        return self.distribution.predicted_class

    @property
    def confidence(self) -> float:
        """Return the predicted-class probability."""
        return self.distribution.predicted_probability

    @property
    def total_count(self) -> float:
        """Return the total weighted class count at the node."""
        return float(sum(self.class_counts.values()))

    def to_dict(self) -> dict[str, Any]:
        """Return the node probability analysis as a dictionary."""
        return {
            "class_probas": self.distribution.to_dict(),
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "total_count": self.total_count,
        }


@dataclass(frozen=True, slots=True)
class TreeNodeResult:
    """Analysis result for one decision-tree node."""

    node_id: int
    depth: int
    is_leaf: bool
    samples: int
    impurity_name: str
    impurity_value: float
    rule: LogicalRule
    probability_analysis: NodeProbabilityAnalysis
    split_feature: str | None
    split_threshold: float | None

    @property
    def predicted_class(self) -> Any:
        """Return the predicted class at the node."""
        return self.probability_analysis.predicted_class

    @property
    def confidence(self) -> float:
        """Return the predicted-class confidence at the node."""
        return self.probability_analysis.confidence

    def to_dict(self) -> dict[str, Any]:
        """Return the node result as a dictionary."""
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "is_leaf": self.is_leaf,
            "samples": self.samples,
            "impurity_value": self.impurity_value,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "class_probas": self.probability_analysis.distribution.to_dict(),
            "split_feature": self.split_feature,
            "split_threshold": self.split_threshold,
            "rule": str(self.rule),
        }


@dataclass(frozen=True, slots=True)
class SplitDiscriminationScore:
    """Impurity-based discrimination score for an internal split."""

    impurity_name: str
    parent_impurity: float
    left_impurity: float
    right_impurity: float
    weighted_children_impurity: float
    impurity_gain: float
    weighted_impurity_gain: float

    def to_dict(self) -> dict[str, float | str]:
        """Return the split score as a dictionary."""
        return {
            "parent_impurity": self.parent_impurity,
            "left_impurity": self.left_impurity,
            "right_impurity": self.right_impurity,
            "weighted_children_impurity": self.weighted_children_impurity,
            "impurity_gain": self.impurity_gain,
            "weighted_impurity_gain": self.weighted_impurity_gain,
        }


@dataclass(frozen=True, slots=True)
class InternalSplitResult:
    """Analysis result for one internal decision-tree split."""

    node_id: int
    depth: int
    samples: int
    feature_name: str
    threshold: float
    rule: LogicalRule
    score: SplitDiscriminationScore

    def to_dict(self) -> dict[str, Any]:
        """Return the internal split result as a dictionary."""
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "samples": self.samples,
            "feature": self.feature_name,
            "threshold": self.threshold,
            "rule": str(self.rule),
            **self.score.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LeafRuleStrengthScore:
    """Strength score associated with a leaf rule."""

    confidence: float
    support: int
    strength: float

    @staticmethod
    def from_confidence_and_support(
        confidence: float,
        support: int,
    ) -> LeafRuleStrengthScore:
        """Build a leaf-rule strength score from confidence and support."""
        return LeafRuleStrengthScore(
            confidence=confidence,
            support=support,
            strength=float(confidence * np.log1p(support)),
        )


@dataclass(frozen=True, slots=True)
class LeafRuleResult:
    """Analysis result for one decision-tree leaf rule."""

    node_id: int
    predicted_class: Any
    rule: LogicalRule
    probability_analysis: NodeProbabilityAnalysis
    impurity_name: str
    impurity_value: float
    score: LeafRuleStrengthScore

    def to_dict(self) -> dict[str, Any]:
        """Return the leaf rule result as a dictionary."""
        return {
            "node_id": self.node_id,
            "predicted_class": self.predicted_class,
            "support": self.score.support,
            "confidence": self.score.confidence,
            "strength": self.score.strength,
            "impurity_value": self.impurity_value,
            "class_probas": self.probability_analysis.distribution.to_dict(),
            "rule": str(self.rule),
        }


@dataclass(frozen=True, slots=True)
class EEGFeatureDescriptor:
    """Structured descriptor inferred from an EEG feature name."""

    feature_name: str
    kind: str
    channel_1: str | None
    channel_2: str | None
    edge: str | None

    def to_dict(self) -> dict[str, str | None]:
        """Return the EEG feature descriptor as a dictionary."""
        return {
            "feature": self.feature_name,
            "kind": self.kind,
            "channel_1": self.channel_1,
            "channel_2": self.channel_2,
            "edge": self.edge,
        }


# =============================================================================
# Result containers
# =============================================================================

@dataclass(frozen=True, slots=True)
class NodeAnalysisResult:
    """Container for decision-tree node analysis results."""

    nodes: tuple[TreeNodeResult, ...]

    def to_dataframe(self) -> pd.DataFrame:
        """Return node analysis results as a DataFrame."""
        return pd.DataFrame([node.to_dict() for node in self.nodes])


@dataclass(frozen=True, slots=True)
class InternalSplitAnalysisResult:
    """Container for internal split analysis results."""

    splits: tuple[InternalSplitResult, ...]

    def to_dataframe(self) -> pd.DataFrame:
        """Return internal split analysis results as a sorted DataFrame."""
        df = pd.DataFrame([split.to_dict() for split in self.splits])

        if df.empty:
            return df

        return df.sort_values(
            "weighted_impurity_gain",
            ascending=False,
        ).reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class LeafRuleAnalysisResult:
    """Container for leaf rule analysis results."""

    rules: tuple[LeafRuleResult, ...]

    def to_dataframe(self) -> pd.DataFrame:
        """Return leaf rule analysis results as a sorted DataFrame."""
        df = pd.DataFrame([rule.to_dict() for rule in self.rules])

        if df.empty:
            return df

        return df.sort_values(
            ["strength", "confidence", "support"],
            ascending=[False, False, False],
        ).reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class FeatureImportanceAnalysisResult:
    """Container for feature importance analysis results."""

    rows: tuple[dict[str, Any], ...]

    def to_dataframe(self) -> pd.DataFrame:
        """Return feature importance results as a sorted DataFrame."""
        df = pd.DataFrame(list(self.rows))

        if df.empty:
            return df

        return df.sort_values(
            ["sklearn_importance", "weighted_impurity_gain", "n_splits"],
            ascending=[False, False, False],
        ).reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class EEGFeatureAnalysisResult:
    """Container for EEG-structure analysis results."""

    rows: tuple[dict[str, Any], ...]

    def to_dataframe(self) -> pd.DataFrame:
        """Return EEG feature analysis results as a sorted DataFrame."""
        df = pd.DataFrame(list(self.rows))

        if df.empty:
            return df

        return df.sort_values(
            ["weighted_impurity_gain", "sklearn_importance", "n_splits"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    def channel_summary(self) -> pd.DataFrame:
        """Return an aggregated summary by EEG channel."""
        df = self.to_dataframe()

        if df.empty:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            for channel in [row["channel_1"], row["channel_2"]]:
                if channel is None:
                    continue

                rows.append(
                    {
                        "channel": channel,
                        "feature": row["feature"],
                        "kind": row["kind"],
                        "n_splits": row["n_splits"],
                        "sklearn_importance": row["sklearn_importance"],
                        "weighted_impurity_gain": row["weighted_impurity_gain"],
                    }
                )

        if not rows:
            return pd.DataFrame()

        return (
            pd.DataFrame(rows)
            .groupby("channel")
            .agg(
                n_features=("feature", "nunique"),
                n_splits=("n_splits", "sum"),
                sklearn_importance=("sklearn_importance", "sum"),
                weighted_impurity_gain=("weighted_impurity_gain", "sum"),
            )
            .sort_values("weighted_impurity_gain", ascending=False)
            .reset_index()
        )

    def edge_summary(self) -> pd.DataFrame:
        """Return an aggregated summary by connectivity edge."""
        df = self.to_dataframe()
        df = df[df["kind"] == "edge"].copy()

        if df.empty:
            return pd.DataFrame()

        return (
            df.groupby("edge")
            .agg(
                n_features=("feature", "nunique"),
                n_splits=("n_splits", "sum"),
                sklearn_importance=("sklearn_importance", "sum"),
                weighted_impurity_gain=("weighted_impurity_gain", "sum"),
            )
            .sort_values("weighted_impurity_gain", ascending=False)
            .reset_index()
        )


# =============================================================================
# Metadata and parsers
# =============================================================================

class TrainedDecisionTreeMetadataExtractor:
    """Extractor for metadata attached to a trained decision tree."""

    def __init__(self, trained_decision_tree: TrainedDecisionTree) -> None:
        self.trained_decision_tree = trained_decision_tree

    @property
    def classifier(self) -> DecisionTreeClassifier:
        """Return the trained classifier."""
        return self.trained_decision_tree.classifier

    @property
    def dataset(self) -> Any:
        """Return the dataset used to train the tree."""
        return self.trained_decision_tree.dataset

    @property
    def criterion(self) -> str:
        """Return the impurity criterion used by the tree."""
        return str(self.classifier.get_params().get("criterion", "impurity"))

    @property
    def class_names(self) -> list[Any]:
        """Return the class names known by the classifier or dataset."""
        if hasattr(self.dataset, "class_names"):
            return list(self.dataset.class_names)

        if hasattr(self.dataset, "classes"):
            return list(self.dataset.classes)

        return list(self.classifier.classes_)

    @property
    def feature_names(self) -> list[str]:
        """Return the feature names used by the trained tree."""
        return self.dataset.all_feature_names


class EEGFeatureParser:
    """Parser used to infer EEG structure from feature names."""

    DEFAULT_EEG_CHANNELS = {
        "Fp1", "Fp2",
        "F3", "F4", "F7", "F8", "Fz",
        "C3", "C4", "Cz",
        "P3", "P4", "Pz",
        "T3", "T4", "T5", "T6",
        "O1", "O2",
    }

    def __init__(self, eeg_channels: Iterable[str] | None = None) -> None:
        self.eeg_channels = set(eeg_channels or self.DEFAULT_EEG_CHANNELS)

    def parse(self, feature_name: str) -> EEGFeatureDescriptor:
        """Parse a feature name into an EEG feature descriptor."""
        tokens = feature_name.split("_")
        channels = [token for token in tokens if token in self.eeg_channels]

        if len(channels) >= 2:
            channel_1, channel_2 = channels[-2], channels[-1]

            return EEGFeatureDescriptor(
                feature_name=feature_name,
                kind="edge",
                channel_1=channel_1,
                channel_2=channel_2,
                edge=f"{channel_1}-{channel_2}",
            )

        if len(channels) == 1:
            channel = channels[0]

            return EEGFeatureDescriptor(
                feature_name=feature_name,
                kind="channel",
                channel_1=channel,
                channel_2=None,
                edge=None,
            )

        return EEGFeatureDescriptor(
            feature_name=feature_name,
            kind="unknown",
            channel_1=None,
            channel_2=None,
            edge=None,
        )


# =============================================================================
# Sub-engines
# =============================================================================

class NodeProbabilityAnalysisEngine:
    """Engine used to analyze class probabilities at decision-tree nodes."""

    def __init__(self, class_names: list[Any]) -> None:
        self.class_names = class_names

    def analyze_node(
        self,
        classifier: DecisionTreeClassifier,
        node_id: int,
    ) -> NodeProbabilityAnalysis:
        """Analyze the class distribution stored at one tree node."""
        tree_ = classifier.tree_

        raw_values = tree_.value[node_id][0].astype(float)
        total = float(raw_values.sum())

        if total == 0.0:
            probabilities = np.zeros_like(raw_values)
        else:
            probabilities = raw_values / total

        class_counts = {
            self.class_names[i]: float(raw_values[i])
            for i in range(len(self.class_names))
        }

        probability_dict = {
            self.class_names[i]: float(probabilities[i])
            for i in range(len(self.class_names))
        }

        return NodeProbabilityAnalysis(
            class_counts=class_counts,
            distribution=ClassProbabilityDistribution(probability_dict),
        )


class TreeTraversalAnalysisEngine:
    """Engine used to traverse a decision tree and analyze each node."""

    def __init__(
        self,
        classifier: DecisionTreeClassifier,
        feature_names: list[str],
        class_names: list[Any],
        impurity_name: str,
    ) -> None:
        self.classifier = classifier
        self.feature_names = feature_names
        self.class_names = class_names
        self.impurity_name = impurity_name
        self.probability_engine = NodeProbabilityAnalysisEngine(class_names)

    def analyze(self) -> NodeAnalysisResult:
        """Traverse the full tree and return node analysis results."""
        nodes: list[TreeNodeResult] = []

        self._walk(
            node_id=0,
            depth=0,
            rule=LogicalRule(()),
            nodes=nodes,
        )

        return NodeAnalysisResult(tuple(nodes))

    def _walk(
        self,
        node_id: int,
        depth: int,
        rule: LogicalRule,
        nodes: list[TreeNodeResult],
    ) -> None:
        """Recursively traverse the tree and collect node results."""
        tree_ = self.classifier.tree_

        left_child = tree_.children_left[node_id]
        right_child = tree_.children_right[node_id]
        is_leaf = left_child == right_child

        if is_leaf:
            split_feature = None
            split_threshold = None
        else:
            feature_index = int(tree_.feature[node_id])
            split_feature = self.feature_names[feature_index]
            split_threshold = float(tree_.threshold[node_id])

        probability_analysis = self.probability_engine.analyze_node(
            classifier=self.classifier,
            node_id=node_id,
        )

        nodes.append(
            TreeNodeResult(
                node_id=node_id,
                depth=depth,
                is_leaf=is_leaf,
                samples=int(tree_.n_node_samples[node_id]),
                impurity_name=self.impurity_name,
                impurity_value=float(tree_.impurity[node_id]),
                rule=rule,
                probability_analysis=probability_analysis,
                split_feature=split_feature,
                split_threshold=split_threshold,
            )
        )

        if is_leaf:
            return

        assert split_feature is not None
        assert split_threshold is not None

        self._walk(
            node_id=left_child,
            depth=depth + 1,
            rule=rule.extend(
                LogicalCondition(
                    feature_name=split_feature,
                    operator="<=",
                    threshold=split_threshold,
                )
            ),
            nodes=nodes,
        )

        self._walk(
            node_id=right_child,
            depth=depth + 1,
            rule=rule.extend(
                LogicalCondition(
                    feature_name=split_feature,
                    operator=">",
                    threshold=split_threshold,
                )
            ),
            nodes=nodes,
        )


class InternalSplitAnalysisEngine:
    """Engine used to score internal decision-tree splits."""

    def __init__(
        self,
        classifier: DecisionTreeClassifier,
        feature_names: list[str],
        impurity_name: str,
        node_result: NodeAnalysisResult,
    ) -> None:
        self.classifier = classifier
        self.feature_names = feature_names
        self.impurity_name = impurity_name
        self.node_result = node_result

    def analyze(self) -> InternalSplitAnalysisResult:
        """Analyze and score all internal tree splits."""
        tree_ = self.classifier.tree_

        splits: list[InternalSplitResult] = []

        for node in self.node_result.nodes:
            node_id = node.node_id

            left_child = tree_.children_left[node_id]
            right_child = tree_.children_right[node_id]

            if left_child == right_child:
                continue

            parent_weight = float(tree_.weighted_n_node_samples[node_id])
            left_weight = float(tree_.weighted_n_node_samples[left_child])
            right_weight = float(tree_.weighted_n_node_samples[right_child])

            parent_impurity = float(tree_.impurity[node_id])
            left_impurity = float(tree_.impurity[left_child])
            right_impurity = float(tree_.impurity[right_child])

            weighted_children_impurity = (
                left_weight / parent_weight * left_impurity
                + right_weight / parent_weight * right_impurity
            )

            impurity_gain = parent_impurity - weighted_children_impurity
            weighted_impurity_gain = impurity_gain * parent_weight

            feature_index = int(tree_.feature[node_id])
            feature_name = self.feature_names[feature_index]

            score = SplitDiscriminationScore(
                impurity_name=self.impurity_name,
                parent_impurity=parent_impurity,
                left_impurity=left_impurity,
                right_impurity=right_impurity,
                weighted_children_impurity=float(weighted_children_impurity),
                impurity_gain=float(impurity_gain),
                weighted_impurity_gain=float(weighted_impurity_gain),
            )

            splits.append(
                InternalSplitResult(
                    node_id=node_id,
                    depth=node.depth,
                    samples=int(tree_.n_node_samples[node_id]),
                    feature_name=feature_name,
                    threshold=float(tree_.threshold[node_id]),
                    rule=node.rule,
                    score=score,
                )
            )

        return InternalSplitAnalysisResult(tuple(splits))


class LeafRuleAnalysisEngine:
    """Engine used to extract and score leaf-level rules."""

    def __init__(self, node_result: NodeAnalysisResult) -> None:
        self.node_result = node_result

    def analyze(self, min_samples: int = 1) -> LeafRuleAnalysisResult:
        """Analyze leaf rules with at least ``min_samples`` samples."""
        rules: list[LeafRuleResult] = []

        for node in self.node_result.nodes:
            if not node.is_leaf:
                continue

            if node.samples < min_samples:
                continue

            score = LeafRuleStrengthScore.from_confidence_and_support(
                confidence=node.confidence,
                support=node.samples,
            )

            rules.append(
                LeafRuleResult(
                    node_id=node.node_id,
                    predicted_class=node.predicted_class,
                    rule=node.rule,
                    probability_analysis=node.probability_analysis,
                    impurity_name=node.impurity_name,
                    impurity_value=node.impurity_value,
                    score=score,
                )
            )

        return LeafRuleAnalysisResult(tuple(rules))


class FeatureImportanceAnalysisEngine:
    """Engine combining sklearn importances with split-level impurity gains."""

    def __init__(
        self,
        classifier: DecisionTreeClassifier,
        feature_names: list[str],
        split_result: InternalSplitAnalysisResult,
    ) -> None:
        self.classifier = classifier
        self.feature_names = feature_names
        self.split_result = split_result

    def analyze(self) -> FeatureImportanceAnalysisResult:
        """Analyze feature importance from sklearn importances and split usage."""
        split_df = self.split_result.to_dataframe()

        if split_df.empty:
            rows = [
                {
                    "feature": feature_name,
                    "sklearn_importance": float(importance),
                    "n_splits": 0,
                    "impurity_gain": 0.0,
                    "weighted_impurity_gain": 0.0,
                    "mean_impurity_gain": 0.0,
                }
                for feature_name, importance in zip(
                    self.feature_names,
                    self.classifier.feature_importances_,
                )
            ]

            return FeatureImportanceAnalysisResult(tuple(rows))

        usage_df = (
            split_df.groupby("feature")
            .agg(
                n_splits=("feature", "size"),
                impurity_gain=("impurity_gain", "sum"),
                weighted_impurity_gain=("weighted_impurity_gain", "sum"),
                mean_impurity_gain=("impurity_gain", "mean"),
            )
            .reset_index()
        )

        sklearn_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "sklearn_importance": self.classifier.feature_importances_,
            }
        )

        merged = sklearn_df.merge(
            usage_df,
            on="feature",
            how="left",
        ).fillna(
            {
                "n_splits": 0,
                "impurity_gain": 0.0,
                "weighted_impurity_gain": 0.0,
                "mean_impurity_gain": 0.0,
            }
        )

        return FeatureImportanceAnalysisResult(
            tuple(merged.to_dict(orient="records"))
        )


class EEGStructureAnalysisEngine:
    """Engine used to attach EEG structural metadata to feature importance rows."""

    def __init__(
        self,
        feature_result: FeatureImportanceAnalysisResult,
        parser: EEGFeatureParser | None = None,
    ) -> None:
        self.feature_result = feature_result
        self.parser = parser or EEGFeatureParser()

    def analyze(self) -> EEGFeatureAnalysisResult:
        """Analyze EEG channel and edge structure from feature names."""
        feature_df = self.feature_result.to_dataframe()

        rows: list[dict[str, Any]] = []

        for _, row in feature_df.iterrows():
            descriptor = self.parser.parse(str(row["feature"]))

            rows.append(
                {
                    **descriptor.to_dict(),
                    "n_splits": int(row["n_splits"]),
                    "sklearn_importance": float(row["sklearn_importance"]),
                    "impurity_gain": float(row["impurity_gain"]),
                    "weighted_impurity_gain": float(row["weighted_impurity_gain"]),
                    "mean_impurity_gain": float(row["mean_impurity_gain"]),
                }
            )

        return EEGFeatureAnalysisResult(tuple(rows))


# =============================================================================
# High-level orchestrator
# =============================================================================

class DecisionTreeAnalysisEngine:
    """High-level orchestrator for complete decision-tree interpretability analysis."""

    def __init__(
        self,
        trained_decision_tree: TrainedDecisionTree,
        *,
        eeg_channels: Iterable[str] | None = None,
    ) -> None:
        self.trained_decision_tree = trained_decision_tree

        metadata = TrainedDecisionTreeMetadataExtractor(trained_decision_tree)

        self.classifier = metadata.classifier
        self.dataset = metadata.dataset
        self.feature_names = metadata.feature_names
        self.class_names = metadata.class_names
        self.impurity_name = metadata.criterion

        self.eeg_parser = EEGFeatureParser(eeg_channels=eeg_channels)

        self._node_result: NodeAnalysisResult | None = None
        self._split_result: InternalSplitAnalysisResult | None = None
        self._feature_result: FeatureImportanceAnalysisResult | None = None
        self._eeg_result: EEGFeatureAnalysisResult | None = None

    def node_analysis(self) -> NodeAnalysisResult:
        """Return cached or newly computed node analysis."""
        if self._node_result is None:
            self._node_result = TreeTraversalAnalysisEngine(
                classifier=self.classifier,
                feature_names=self.feature_names,
                class_names=self.class_names,
                impurity_name=self.impurity_name,
            ).analyze()

        return self._node_result

    def internal_split_analysis(self) -> InternalSplitAnalysisResult:
        """Return cached or newly computed internal split analysis."""
        if self._split_result is None:
            self._split_result = InternalSplitAnalysisEngine(
                classifier=self.classifier,
                feature_names=self.feature_names,
                impurity_name=self.impurity_name,
                node_result=self.node_analysis(),
            ).analyze()

        return self._split_result

    def leaf_rule_analysis(self, min_samples: int = 1) -> LeafRuleAnalysisResult:
        """Return leaf rule analysis."""
        return LeafRuleAnalysisEngine(
            node_result=self.node_analysis(),
        ).analyze(min_samples=min_samples)

    def feature_importance_analysis(self) -> FeatureImportanceAnalysisResult:
        """Return cached or newly computed feature importance analysis."""
        if self._feature_result is None:
            self._feature_result = FeatureImportanceAnalysisEngine(
                classifier=self.classifier,
                feature_names=self.feature_names,
                split_result=self.internal_split_analysis(),
            ).analyze()

        return self._feature_result

    def eeg_structure_analysis(self) -> EEGFeatureAnalysisResult:
        """Return cached or newly computed EEG structure analysis."""
        if self._eeg_result is None:
            self._eeg_result = EEGStructureAnalysisEngine(
                feature_result=self.feature_importance_analysis(),
                parser=self.eeg_parser,
            ).analyze()

        return self._eeg_result

    def node_dataframe(self) -> pd.DataFrame:
        """Return node analysis as a DataFrame."""
        return self.node_analysis().to_dataframe()

    def strongest_leaf_rules(
        self,
        *,
        min_samples: int = 5,
        sort_by: str = "strength",
    ) -> pd.DataFrame:
        """Return strongest leaf rules sorted by the requested criterion."""
        df = self.leaf_rule_analysis(min_samples=min_samples).to_dataframe()

        if df.empty:
            return df

        if sort_by == "strength":
            return df.sort_values(
                ["strength", "confidence", "support"],
                ascending=[False, False, False],
            ).reset_index(drop=True)

        if sort_by == "confidence":
            return df.sort_values(
                ["confidence", "support"],
                ascending=[False, False],
            ).reset_index(drop=True)

        if sort_by == "samples":
            return df.sort_values(
                ["support", "confidence"],
                ascending=[False, False],
            ).reset_index(drop=True)

        raise ValueError("sort_by doit être 'strength', 'confidence' ou 'samples'.")

    def strongest_internal_splits(self) -> pd.DataFrame:
        """Return internal splits sorted by weighted impurity gain."""
        return self.internal_split_analysis().to_dataframe()

    def feature_importance_report(self) -> pd.DataFrame:
        """Return the feature importance report."""
        return self.feature_importance_analysis().to_dataframe()

    def channel_edge_report(self) -> pd.DataFrame:
        """Return the channel and edge report."""
        return self.eeg_structure_analysis().to_dataframe()

    def channel_summary(self) -> pd.DataFrame:
        """Return the aggregated channel summary."""
        return self.eeg_structure_analysis().channel_summary()

    def edge_summary(self) -> pd.DataFrame:
        """Return the aggregated edge summary."""
        return self.eeg_structure_analysis().edge_summary()

    def print_report(
        self,
        *,
        min_samples_leaf: int = 5,
        top_n: int = 10,
    ) -> None:
        """Print a human-readable decision-tree analysis report."""
        print("\n" + "=" * 100)
        print("DECISION TREE ANALYSIS REPORT")
        print("=" * 100)

        print(f"\nCriterion: {self.impurity_name}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Classes: {self.class_names}")

        print("\n[1] Règles terminales les plus fortes")
        print("-" * 100)
        leaf_df = self.strongest_leaf_rules(
            min_samples=min_samples_leaf,
            sort_by="strength",
        )
        print(
            leaf_df.head(top_n).to_string(index=False)
            if not leaf_df.empty
            else "Aucune règle."
        )

        print("\n[2] Splits internes les plus discriminants")
        print("-" * 100)
        split_df = self.strongest_internal_splits()
        print(
            split_df.head(top_n).to_string(index=False)
            if not split_df.empty
            else "Aucun split."
        )

        print("\n[3] Features les plus discriminantes")
        print("-" * 100)
        feature_df = self.feature_importance_report()
        print(
            feature_df.head(top_n).to_string(index=False)
            if not feature_df.empty
            else "Aucune feature."
        )

        print("\n[4] Channels les plus informatifs")
        print("-" * 100)
        channel_df = self.channel_summary()
        print(
            channel_df.head(top_n).to_string(index=False)
            if not channel_df.empty
            else "Aucun channel détecté."
        )

        print("\n[5] Edges les plus informatives")
        print("-" * 100)
        edge_df = self.edge_summary()
        print(
            edge_df.head(top_n).to_string(index=False)
            if not edge_df.empty
            else "Aucune edge détectée."
        )

        print("\n" + "=" * 100)