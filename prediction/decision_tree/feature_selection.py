from __future__ import annotations

import numpy as np
from tqdm import tqdm

from features.dataset import FeaturesDataset, FeaturesDatasetSelector
from features.name import FeatureNameHelper
from prediction.decision_tree.trainer import (
    DecisionTreeParameters,
    DecisionTreeTrainer,
)
import pandas as pd


class FeatureSelector:
    """
    Forward selection par blocs de features, cohérente avec le framework.

    Philosophie
    -----------
    - Les noms manipulés par cette classe sont des noms "simples" de familles
      de features :
          * EEG scalaire : "variance", "theta_beta_ratio", ...
          * connectivité : "cn_alpha", "cn_theta", ...
          * metadata     : "subject_age", "subject_mmse", ...
    - L'expansion en vraies colonnes du wide dataframe est déléguée à
      `FeatureNameHelper`.
    - L'évaluation ML est déléguée à `DecisionTreeTrainer`, qui attend un
      `SelectedFeaturesDataset`.
    """

    def __init__(self, trainer: DecisionTreeTrainer, helper: FeatureNameHelper):
        self.trainer = trainer
        self.helper = helper
        self.results = []

    def _expand_feature(self, feature_name: str) -> list[str]:
        """
        Convertit un nom de feature simple en bloc réel de colonnes via le helper.

        Règles
        ------
        - "variance"      -> helper.build(eeg="variance")
        - "cn_alpha"      -> helper.build(cn="alpha")
        - "subject_age"   -> helper.build(subject="subject_age")
        """
        if not isinstance(feature_name, str):
            raise TypeError("`feature_name` must be a string.")

        if feature_name.startswith("cn_"):
            band_name = feature_name[len("cn_"):]
            return self.helper.build(cn=band_name)

        if feature_name.startswith("subject_"):
            return self.helper.build(subject=feature_name)

        return self.helper.build(eeg=feature_name)

    def forward_selection(
        self,
        dataset: FeaturesDataset,
        params: DecisionTreeParameters,
        max_features: int = 20,
    ) -> tuple[list[str], list[str], float]:
        """
        Effectue une forward selection par blocs.

        Paramètres
        ----------
        dataset :
            Dataset complet de départ.
        feature_names :
            Liste de noms simples de familles de features.
            Exemples :
                ["theta_beta_ratio", "spectral_flux", "cn_alpha", "subject_age"]
        params :
            Hyperparamètres du decision tree déjà choisis.
        max_features :
            Nombre maximal de blocs à sélectionner.

        Retour
        ------
        selected_feature_names :
            Liste des noms simples sélectionnés.
        selected_columns :
            Liste aplatie des colonnes effectivement utilisées.
        best_score :
            Meilleur score moyen CV obtenu.
        """
        feature_names = dataset.feature_names

        if max_features <= 0:
            raise ValueError("`max_features` must be > 0.")

        self.results = []

        selected_feature_names: list[str] = []
        selected_columns: list[str] = []
        remaining = list(feature_names)

        best_score = -np.inf

        print(
            f"🔍 Forward feature selection "
            f"(max {min(max_features, len(feature_names))} blocks over {len(feature_names)} candidates)"
        )

        outer_total = min(max_features, len(feature_names))

        with tqdm(total=outer_total, desc="Selecting blocks") as pbar:
            while remaining and len(selected_feature_names) < max_features:
                scores: list[tuple[float, float, str, list[str]]] = []

                for feature_name in tqdm(
                    remaining,
                    desc=f"Iteration {len(selected_feature_names) + 1}",
                    leave=False,
                ):
                    block_columns = self._expand_feature(feature_name)

                    candidate_columns = selected_columns + block_columns
                    candidate_columns = list(dict.fromkeys(candidate_columns))

                    candidate_dataset = FeaturesDatasetSelector.select(
                        dataset=dataset,
                        selection=candidate_columns,
                    )

                    score_mean, score_std = self.trainer.evaluate(
                        dataset=candidate_dataset,
                        params=params,
                    )

                    scores.append((score_mean, score_std, feature_name, block_columns))

                scores.sort(key=lambda x: x[0], reverse=True)

                best_candidate_score, best_candidate_std, best_feature, best_block = scores[0]

                self.results.append(
                    {
                        "step": len(selected_feature_names) + 1,
                        "feature": best_feature,
                        "n_columns_added": len(best_block),
                        "n_total_columns": len(list(dict.fromkeys(selected_columns + best_block))),
                        "mean": best_candidate_score,
                        "std": best_candidate_std,
                    }
                )

                if best_candidate_score <= best_score:
                    print("Stopping: no further improvement.")
                    break

                selected_feature_names.append(best_feature)
                selected_columns.extend(best_block)
                selected_columns = list(dict.fromkeys(selected_columns))

                remaining.remove(best_feature)
                best_score = best_candidate_score

                print(
                    f"Added {best_feature} "
                    f"({len(best_block)} cols) "
                    f"-> score={best_candidate_score:.4f} ± {best_candidate_std:.4f}"
                )

                pbar.update(1)

        return selected_feature_names, selected_columns, best_score

    def get_results_dataframe(self):
        
        return pd.DataFrame(self.results)