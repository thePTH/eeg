from itertools import product
from prediction.decision_tree.trainer import DecisionTreeTrainer, SelectedFeaturesDataset, DecisionTreeParameters
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from functools import cached_property


class HyperparameterGrid:
    criterion:list[str]=["gini", "entropy"]
    max_depth:list[int]=[6]
    min_samples_split:list[int]=[4]
    min_samples_leaf:list[int]=[1]
    

    def to_dict(self):
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "criterion": self.criterion
        }
    
    @property
    def keys(self):
        return list(self.to_dict().keys())
        

    @property
    def values(self):
        return list(self.to_dict().values())
    
    @cached_property
    def all_combinations(self) ->list[DecisionTreeParameters]:
        values = self.values
        keys = self.keys
        cartesian_product = list(product(*values))
        all_params = []
        for combo in cartesian_product :
            params_dico = dict(zip(keys, combo))
            
            params = DecisionTreeParameters(criterion=params_dico["criterion"], max_depth=params_dico["max_depth"], min_samples_leaf=params_dico["min_samples_leaf"], min_samples_split=params_dico["min_samples_split"])

            all_params.append(params)

        return all_params
    
    @property
    def size(self):
        return len(self.all_combinations)
    

    def __repr__(self):
        return f"{self.to_dict()}"




   


class HyperparameterSearch:

    def __init__(self, dataset:SelectedFeaturesDataset,trainer:DecisionTreeTrainer, lambda_std: float = 0.5):
        """
        Parameters
        ----------
        trainer : DecisionTreeTrainer
            Objet responsable de la cross-validation.

        lambda_std : float
            Coefficient de pénalisation de l'écart-type :
            score_adjusted = mean - lambda_std * std
        """
        self.dataset = dataset
        self.trainer = trainer
        self.lambda_std = lambda_std

        # stockage complet des résultats
        self.results = []

    def search(self, grid:HyperparameterGrid):

        best_adjusted_score = -np.inf
        best_params = None

        

        print(f"🔍 Hyperparameter search ({grid.size} combinations)")

        for params in tqdm(grid.all_combinations):


            score_mean, score_std = self.trainer.evaluate(
                self.dataset,
                params,
            )

            adjusted_score = score_mean - self.lambda_std * score_std

            # stockage historique complet
            self.results.append(
                {
                    "params": params,
                    "mean": score_mean,
                    "std": score_std,
                    "adjusted": adjusted_score,
                }
            )

            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_params = params

        print(f"\n✅ Best params found: {best_params}")
        
        print(f"Adjusted score = {best_adjusted_score:.4f}")

        return best_params, best_adjusted_score

    def get_top_k(self, k=5):
        """
        Retourne les k meilleurs résultats triés par score ajusté.
        """
        return sorted(
            self.results,
            key=lambda x: x["adjusted"],
            reverse=True,
        )[:k]

    def get_results_dataframe(self):
        """
        Convertit l'historique en DataFrame pandas (pratique pour analyse).
        """
        import pandas as pd

        rows = []

        for entry in self.results:
            row = entry["params"].to_dict()
            row["mean"] = entry["mean"]
            row["std"] = entry["std"]
            row["adjusted"] = entry["adjusted"]
            rows.append(row)

        return pd.DataFrame(rows)