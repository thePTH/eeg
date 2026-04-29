from .base import DecisionTree
from features.dataset import (
    SelectedFeaturesDataset,
    SelectedFeature,
    SelectedFeaturesDatasetFactory,
    SelectedFeaturesConcatEngine,
)
from .score import DecisionTreeScoreEngine
import numpy as np


class FeatureForwardSelectionEngine:
    def __init__(
        self,
        score_engine: DecisionTreeScoreEngine,
        decision_tree: DecisionTree,
        *,
        verbose: bool = True,
    ):
        self.score_engine = score_engine
        self.decision_tree = decision_tree
        self.verbose = verbose

    def _display(self, message: str):
        if self.verbose:
            print(f"\r{message}", end="", flush=True)

    def _evaluate_one_new_feature(
        self,
        dataset: SelectedFeaturesDataset,
        selected_features: list[SelectedFeature],
        feature_to_evaluate: SelectedFeature,
    ):
        new_columns = (
            feature_to_evaluate.columns
            + SelectedFeaturesConcatEngine.concat_columns(selected_features)
        )

        new_dataset = SelectedFeaturesDatasetFactory.from_selected_columns(
            dataset,
            new_columns,
        )

        return self.score_engine.score(self.decision_tree, new_dataset)

    def _find_best_new_feature(
        self,
        remaining_features: list[SelectedFeature],
        selected_features: list[SelectedFeature],
        dataset: SelectedFeaturesDataset,
        lambda_std: float,
        iteration: int,
    ):
        best_feature = None
        best_score = None
        best_adjusted_score = -np.inf

        for feature in remaining_features:
            score = self._evaluate_one_new_feature(
                dataset,
                selected_features,
                feature,
            )

            adjusted_score = score.adjusted_score(lambda_std)

            

            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_feature = feature
                best_score = score
                

        return best_feature, best_score

    def run(
        self,
        dataset: SelectedFeaturesDataset,
        lambda_std: float = 0,
    ):
        selected_features = []
        remaining_features = dataset.selected_features.copy()

        current_score = -np.inf
        iteration = 1

        while remaining_features:
            best_feature, score = self._find_best_new_feature(
                remaining_features,
                selected_features,
                dataset,
                lambda_std,
                iteration,
            )

            adjusted_score = score.adjusted_score(lambda_std)

            if np.isneginf(current_score) or adjusted_score > current_score:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                current_score = adjusted_score
                
            else:
                break

            iteration += 1

        if self.verbose:
            print()

        return SelectedFeaturesDatasetFactory.from_selected_features_list(dataset, selected_features)




class DecisionTreeFeatureSelectionTrainer:
    def __init__(self, score_engine:DecisionTreeScoreEngine, lambda_std:float=0):
        self.score_engine = score_engine
        self.lambda_std = lambda_std

    def train(self, decision_tree:DecisionTree, dataset:SelectedFeaturesDataset):
        feature_selector = FeatureForwardSelectionEngine(self.score_engine, decision_tree)

        selected_dataset = feature_selector.run(dataset, self.lambda_std)
        train_dataset, test_dataset = selected_dataset.selector.group_train_test_split()
        trained_tree = decision_tree.train(train_dataset)

        return trained_tree, test_dataset