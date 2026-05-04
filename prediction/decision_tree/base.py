from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
import numpy as np
from typing import Tuple
from features.dataset import SelectedFeaturesDataset

from sklearn.base import clone



@dataclass
class DecisionTreeParameters:
    criterion:str="entropy"
    max_depth:int=4
    min_samples_split:int=5
    min_samples_leaf:int=10
    

    def to_dict(self):
        return {"criterion" : self.criterion, "max_depth":self.max_depth, "min_samples_split":self.min_samples_split, "min_samples_leaf":self.min_samples_leaf, "random_state":42}
   


class TrainedDecisionTree:
    def __init__(self, trained_classifier:DecisionTreeClassifier, train_dataset:SelectedFeaturesDataset):
        self._trained_classifier = trained_classifier
        self._train_dataset = train_dataset

    @property
    def classifier(self):
        return self._trained_classifier
    
    @property
    def dataset(self):
        return self._train_dataset
    
    @property
    def parameters(self):
        dico_params = self.classifier.get_params()
        return DecisionTreeParameters(criterion=dico_params["criterion"], max_depth=dico_params["max_depth"], min_samples_split=dico_params["min_sample_split"], min_samples_leaf=dico_params["min_samplt_leaf"])
    
    def train(self, train_dataset):
        trained_classifier : DecisionTreeClassifier = clone(self.classifier)
        trained_classifier.fit(X = train_dataset.X, y=train_dataset.y)
        return TrainedDecisionTree(trained_classifier=trained_classifier, train_dataset=train_dataset)

    




class DecisionTree:
    def __init__(self, parameters:DecisionTreeParameters):
        self._parameters = parameters
        self._classifier = DecisionTreeClassifier(**parameters.to_dict())

    @property
    def parameters(self):
        return self._parameters
    
    @property
    def classifier(self):
        return self._classifier
    
    def train(self, train_dataset:SelectedFeaturesDataset):
        trained_classifier : DecisionTreeClassifier = clone(self.classifier)
        trained_classifier.fit(X = train_dataset.X, y=train_dataset.y)
        return TrainedDecisionTree(trained_classifier=trained_classifier, train_dataset=train_dataset)



    



