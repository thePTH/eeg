from dataclasses import dataclass

from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier

from features.dataset import SelectedFeaturesDataset


@dataclass
class DecisionTreeParameters:
    """Configuration parameters for a scikit-learn decision tree classifier."""

    criterion: str = "entropy"
    max_depth: int = 4
    min_samples_split: int = 5
    min_samples_leaf: int = 10

    def to_dict(self):
        """Convert parameters to a dictionary compatible with DecisionTreeClassifier."""
        return {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": 42,
        }


class TrainedDecisionTree:
    """Wrapper around a trained decision tree and its training dataset."""

    def __init__(
        self,
        trained_classifier: DecisionTreeClassifier,
        train_dataset: SelectedFeaturesDataset,
    ):
        self._trained_classifier = trained_classifier
        self._train_dataset = train_dataset

    @property
    def classifier(self):
        """Return the trained decision tree classifier."""
        return self._trained_classifier

    @property
    def dataset(self):
        """Return the dataset used to train the classifier."""
        return self._train_dataset

    @property
    def parameters(self):
        """Return the classifier parameters as DecisionTreeParameters."""
        dico_params = self.classifier.get_params()

        return DecisionTreeParameters(
            criterion=dico_params["criterion"],
            max_depth=dico_params["max_depth"],
            min_samples_split=dico_params["min_sample_split"],
            min_samples_leaf=dico_params["min_samplt_leaf"],
        )

    def train(self, train_dataset):
        """Clone and retrain the current classifier on another dataset."""
        trained_classifier: DecisionTreeClassifier = clone(self.classifier)
        trained_classifier.fit(X=train_dataset.X, y=train_dataset.y)

        return TrainedDecisionTree(
            trained_classifier=trained_classifier,
            train_dataset=train_dataset,
        )


class DecisionTree:
    """Trainable decision tree wrapper."""

    def __init__(self, parameters: DecisionTreeParameters):
        self._parameters = parameters
        self._classifier = DecisionTreeClassifier(**parameters.to_dict())

    @property
    def parameters(self):
        """Return the decision tree parameters."""
        return self._parameters

    @property
    def classifier(self):
        """Return the underlying decision tree classifier."""
        return self._classifier

    def train(self, train_dataset: SelectedFeaturesDataset):
        """Train the decision tree on a selected-features dataset."""
        trained_classifier: DecisionTreeClassifier = clone(self.classifier)
        trained_classifier.fit(X=train_dataset.X, y=train_dataset.y)

        return TrainedDecisionTree(
            trained_classifier=trained_classifier,
            train_dataset=train_dataset,
        )