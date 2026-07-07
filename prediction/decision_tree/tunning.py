from functools import cached_property
from itertools import product

import numpy as np
from tqdm import tqdm

from features.dataset import SelectedFeaturesDataset

from .base import DecisionTree, DecisionTreeParameters
from .score import DecisionTreeScoreEngine


class HyperparameterGrid:
    """Hyperparameter grid for decision-tree optimization."""

    criterion: list[str] = ["gini", "entropy"]
    max_depth: list[int] = [5, 6, 7, 8, 10, 15, 20]
    min_samples_split: list[int] = [2, 5, 10, 20]
    min_samples_leaf: list[int] = [2, 5, 10, 20]

    def to_dict(self):
        """Return the grid as a dictionary."""
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "criterion": self.criterion,
        }

    @property
    def keys(self):
        """Return hyperparameter names."""
        return list(self.to_dict().keys())

    @property
    def values(self):
        """Return hyperparameter value lists."""
        return list(self.to_dict().values())

    def iter_combinations(self):
        """Iterate over all decision-tree parameter combinations."""
        for combo in product(*self.values):
            params_dico = dict(zip(self.keys, combo))

            yield DecisionTreeParameters(
                criterion=params_dico["criterion"],
                max_depth=params_dico["max_depth"],
                min_samples_leaf=params_dico["min_samples_leaf"],
                min_samples_split=params_dico["min_samples_split"],
            )

    @cached_property
    def size(self):
        """Return the total number of hyperparameter combinations."""
        size = 1

        for values in self.values:
            size *= len(values)

        return size

    def __repr__(self):
        """Return the grid representation."""
        return f"{self.to_dict()}"


class DecisionTreeOptimizer:
    """Optimizer performing grid search over decision-tree hyperparameters."""

    def __init__(
        self,
        dataset: SelectedFeaturesDataset,
        score_engine: DecisionTreeScoreEngine,
    ):
        self.dataset = dataset
        self.scorer = score_engine

    def optimize(self, grid: HyperparameterGrid, lambda_std: float = 0):
        """Run grid search and return the best decision tree with its score."""
        best_adjusted_score = -np.inf
        best_scoring = None
        best_params = None

        print(f"🔍 Hyperparameter search ({grid.size} combinations)")

        for params in tqdm(grid.iter_combinations(), total=grid.size):
            decision_tree = DecisionTree(parameters=params)
            scoring = self.scorer.score(decision_tree, self.dataset)
            adjusted_score = scoring.adjusted_score(lambda_std)

            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_params = params
                best_scoring = scoring

        print(f"\n✅ Best params found: {best_params}")
        print(f"Adjusted score = {best_adjusted_score:.4f}")

        return DecisionTree(best_params), best_scoring