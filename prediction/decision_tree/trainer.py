from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
import numpy as np
from typing import Tuple
from features.dataset import SelectedFeaturesDataset


@dataclass
class DecisionTreeParameters:
    criterion:str="entropy"
    max_depth:int=4
    min_samples_split:int=5
    min_samples_leaf:int=10
    

    def to_dict(self):
        return {"criterion" : self.criterion, "max_depth":self.max_depth, "min_samples_split":self.min_samples_split, "min_samples_leaf":self.min_samples_leaf}
    




@dataclass
class DecisionTreeTrainer:
    n_splits: int = 5
    random_state: int = 42

    def evaluate(
        self,

        dataset:SelectedFeaturesDataset,
        params:DecisionTreeParameters,

    )-> Tuple[float, float]:
        
        X = dataset.X
        y = dataset.y
        groups = dataset.groups
        
        params_dico = params.to_dict()
        params_dico["random_state"] = self.random_state

        model = DecisionTreeClassifier(**params_dico)
        cv = GroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        scores = cross_val_score(
            model,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring="accuracy",
        )

        return scores.mean(), scores.std()