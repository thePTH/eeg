from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError
from prediction.decision_tree.predictor import TrainedDecisionTreePredictor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


        
class DecisionTreeVisualization:

    @staticmethod
    def plot(decision_tree:TrainedDecisionTreePredictor, figsize=(20, 10), filled=True, rounded=True, fontsize=10):
        model = decision_tree.classifier
        feature_names = decision_tree.train_dataset.selector.X().columns
        class_names=["Healthy", "Alzheimer"]
        max_depth = decision_tree.parameters.max_depth



        plt.figure(figsize=figsize)

        plot_tree(
            model,
            feature_names=feature_names,
            class_names=class_names,
            filled=filled,
            rounded=rounded,
            fontsize=fontsize,
            max_depth=max_depth,
        )

        plt.tight_layout()
        plt.show()
