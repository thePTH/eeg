from features.dataset import SelectedFeaturesDataset
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

class DecisionTreeVisualizationEngine :
    @staticmethod
    def plot(
        tree: DecisionTreeClassifier,
        train_dataset: SelectedFeaturesDataset,
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
        Affiche proprement et élégamment un `DecisionTreeClassifier` sklearn
        en utilisant automatiquement les noms de features et de classes contenus
        dans `train_dataset : SelectedFeaturesDataset`.

        Cette fonction :
        - vérifie que l'arbre est bien entraîné,
        - récupère les noms de variables depuis `train_dataset.X.columns`,
        - récupère les noms de classes depuis `train_dataset.y`,
        - récupère les hyperparamètres via `tree.get_params()`,
        - ajoute ces informations dans le rendu matplotlib.

        Parameters
        ----------
        tree:
            Arbre de décision sklearn déjà entraîné.
        train_dataset:
            Dataset d'entraînement de type `SelectedFeaturesDataset`.
            On suppose qu'il contient au minimum :
            - `X` : DataFrame pandas
            - `y` : série / array-like des labels
        max_depth:
            Profondeur maximale affichée dans le graphique.
            Ne modifie pas l'arbre, seulement l'affichage.
        figsize:
            Taille explicite de la figure. Si `None`, une taille adaptée est calculée.
        fontsize:
            Taille de police utilisée dans le dessin de l'arbre.
        filled:
            Si True, colore les noeuds selon la classe majoritaire.
        rounded:
            Si True, utilise des boîtes arrondies.
        precision:
            Nombre de décimales affichées.
        proportion:
            Si True, affiche les proportions au lieu des effectifs bruts.
        impurity:
            Si True, affiche la mesure d'impureté (gini ou entropy).
        label:
            Paramètre transmis à `sklearn.tree.plot_tree`.
        title:
            Titre principal de la figure.
        show_params:
            Si True, affiche un sous-titre avec les principaux hyperparamètres.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            La figure matplotlib et son axe.
        """
        

        # ------------------------------------------------------------------
        # 2. Récupération automatique des noms de features
        # ------------------------------------------------------------------
        feature_names = train_dataset.all_feature_names



        # ------------------------------------------------------------------
        # 3. Récupération automatique des noms de classes
        # ------------------------------------------------------------------
        # On essaie plusieurs stratégies pour rester robuste selon ton implémentation
        # de SelectedFeaturesDataset.
        class_names = np.unique(train_dataset.wide_dataframe.subject_health)

        
        # ------------------------------------------------------------------
        # 4. Taille automatique de la figure
        # ------------------------------------------------------------------
        if figsize is None:
            displayed_depth = tree.get_depth() if max_depth is None else min(tree.get_depth(), max_depth)
            width = max(14.0, 2.6 * (displayed_depth + 2))
            height = max(7.0, 1.8 * (displayed_depth + 3))
            figsize = (width, height)

        # ------------------------------------------------------------------
        # 5. Hyperparamètres du modèle via get_params()
        # ------------------------------------------------------------------
        params: dict[str, Any] = tree.get_params()

        # On sélectionne les paramètres les plus utiles visuellement.
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
            f"{key}={value}" for key, value in displayed_params.items()
        )

        # ------------------------------------------------------------------
        # 6. Plot
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 7. Titre et sous-titre élégants
        # ------------------------------------------------------------------
        if title is None:
            title = "Decision Tree Classifier"

        ax.set_title(title, fontsize=fontsize + 4, pad=28, fontweight="bold")

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