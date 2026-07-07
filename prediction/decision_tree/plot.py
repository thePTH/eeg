from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import plot_tree

from .base import TrainedDecisionTree


class DecisionTreeVisualizationEngine:
    """Visualization engine for trained scikit-learn decision trees."""

    @staticmethod
    def plot(
        decision_tree: TrainedDecisionTree,
        *,
        max_depth: int | None = None,
        figsize: tuple[float, float] | None = None,
        fontsize: int = 10,
        filled: bool = True,
        rounded: bool = True,
        precision: int = 2,
        proportion: bool = False,
        impurity: bool = True,
        label: str = "all",
        title: str | None = None,
        show_params: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot a trained scikit-learn DecisionTreeClassifier.

        The method automatically retrieves feature names and class names from
        the associated training dataset, and can display the main model
        hyperparameters as a subtitle.
        """
        tree = decision_tree.classifier
        train_dataset = decision_tree.dataset

        feature_names = train_dataset.all_feature_names
        class_names = np.unique(train_dataset.wide_dataframe.subject_health)

        if figsize is None:
            displayed_depth = (
                tree.get_depth()
                if max_depth is None
                else min(tree.get_depth(), max_depth)
            )
            width = max(14.0, 2.6 * (displayed_depth + 2))
            height = max(7.0, 1.8 * (displayed_depth + 3))
            figsize = (width, height)

        params: dict[str, Any] = tree.get_params()

        displayed_params = {
            "criterion": params.get("criterion"),
            "splitter": params.get("splitter"),
            "max_depth": params.get("max_depth"),
            "min_samples_split": params.get("min_samples_split"),
            "min_samples_leaf": params.get("min_samples_leaf"),
            "max_features": params.get("max_features"),
            "ccp_alpha": params.get("ccp_alpha"),
            "random_state": params.get("random_state"),
        }

        params_text = " | ".join(
            f"{key}={value}"
            for key, value in displayed_params.items()
        )

        fig, ax = plt.subplots(figsize=figsize)

        plot_tree(
            decision_tree=tree,
            feature_names=feature_names,
            class_names=class_names,
            filled=filled,
            rounded=rounded,
            max_depth=max_depth,
            fontsize=fontsize,
            precision=precision,
            proportion=proportion,
            impurity=impurity,
            label=label,
            ax=ax,
        )

        if title is None:
            title = "Decision Tree Classifier"

        ax.set_title(
            title,
            fontsize=fontsize + 4,
            pad=28,
            fontweight="bold",
        )

        if show_params:
            fig.text(
                0.5,
                0.965,
                params_text,
                ha="center",
                va="top",
                fontsize=max(fontsize - 1, 8),
                color="dimgray",
            )

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.plot()