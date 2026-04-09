from prediction.decision_tree.hyperparameters import DecisionTreeParameters
from features.dataset import FeaturesDataset
from sklearn.tree import DecisionTreeClassifier


class TrainedDecisionTreePredictor:
    def __init__(self, trained_classifier:DecisionTreeClassifier, train_dataset:FeaturesDataset,
    parameters:DecisionTreeParameters):
        
        self._trained_classifier = trained_classifier
        self._train_dataset = train_dataset
        self._parameters = parameters

    @property
    def train_dataset(self):
        return self._train_dataset
    
    @property
    def classifier(self):
        return self._trained_classifier
    
    @property
    def parameters(self):
        return self._parameters
    
    def predict(self, X):
        return self.classifier.predict(X)