from __future__ import annotations
from typing import ClassVar


from abc import ABC, abstractmethod
from features.context import FeatureExtractionContext
from features.categories import FeatureCategory


class EEGExtractedFeature:
    def __init__(self, name:str, category:FeatureCategory, value:float):
        self._name =name
        self._category = category
        self._value = value

    @property
    def name(self):
        return self._name
    
    @property
    def category(self):
        return self._category
    
    @property
    def value(self):
        return self._value
    
    def __repr__(self):
        return str(self.value)
    


class EEGFeature(ABC):
    """
    Classe abstraite de base pour une feature EEG.
    """

    name: ClassVar[str]
    category: ClassVar[FeatureCategory]

    @staticmethod
    @abstractmethod
    def _extract(context: FeatureExtractionContext) -> float:
        """
        Calcule la feature à partir du contexte.
        """
        raise NotImplementedError
    
    @classmethod
    def compute(cls, context:FeatureExtractionContext) -> EEGExtractedFeature :
        return EEGExtractedFeature(name=cls.name, category=cls.category, value=cls._extract(context))
    





from collections import defaultdict
from typing import Type

from features.categories import FeatureCategory


class RegisteredFeatureProvider:
    """
    Registre central des classes de features EEG.

    Stocke automatiquement les classes décorées par @register_feature
    et permet de les récupérer facilement :
    - toutes les features
    - par catégorie
    - par nom
    """

    _features: list[type[EEGFeature]] = []
    _features_by_category: dict[FeatureCategory, list[type[EEGFeature]]] = defaultdict(list)
    _features_by_name: dict[str, type[EEGFeature]] = {}

    @classmethod
    def register(cls, feature_cls: type[EEGFeature]) -> type[EEGFeature]:
        """
        Enregistre une classe de feature dans le registre global.

        Paramètres
        ----------
        feature_cls : type[EEGFeature]
            Classe de feature à enregistrer.

        Retour
        ------
        type[EEGFeature]
            La classe elle-même, pour permettre l'usage en décorateur.
        """

        # Vérification minimale : on s'assure qu'on enregistre bien une sous-classe de EEGFeature
        if not issubclass(feature_cls, EEGFeature):
            raise TypeError(
                f"{feature_cls.__name__} doit hériter de EEGFeature pour être enregistrée."
            )

        # Vérification de la présence des attributs nécessaires
        if not hasattr(feature_cls, "name"):
            raise AttributeError(
                f"{feature_cls.__name__} doit définir un attribut de classe 'name'."
            )

        if not hasattr(feature_cls, "category"):
            raise AttributeError(
                f"{feature_cls.__name__} doit définir un attribut de classe 'category'."
            )

        feature_name = feature_cls.name
        feature_category = feature_cls.category

        # On évite les doublons de nom
        if feature_name in cls._features_by_name:
            existing_cls = cls._features_by_name[feature_name]
            raise ValueError(
                f"La feature '{feature_name}' est déjà enregistrée "
                f"par la classe {existing_cls.__name__}."
            )

        cls._features.append(feature_cls)
        cls._features_by_category[feature_category].append(feature_cls)
        cls._features_by_name[feature_name] = feature_cls

        return feature_cls

    @classmethod
    def get_all(cls) -> list[type[EEGFeature]]:
        """
        Retourne toutes les classes de features enregistrées.
        """
        return list(cls._features)

    @classmethod
    def get_by_category(cls, category: FeatureCategory) -> list[type[EEGFeature]]:
        """
        Retourne toutes les features enregistrées pour une catégorie donnée.
        """
        return list(cls._features_by_category.get(category, []))

    @classmethod
    def get_by_categories(
        cls,
        categories: list[FeatureCategory]
    ) -> list[type[EEGFeature]]:
        """
        Retourne toutes les features correspondant à plusieurs catégories.
        L'ordre suit l'ordre d'enregistrement.
        """
        selected_categories = set(categories)
        return [
            feature_cls
            for feature_cls in cls._features
            if feature_cls.category in selected_categories
        ]

    @classmethod
    def get_by_name(cls, name: str) -> type[EEGFeature]:
        """
        Retourne la classe de feature correspondant à un nom.
        """
        return cls._features_by_name[name]

    @classmethod
    def clear(cls) -> None:
        """
        Vide le registre.
        Utile en test unitaire.
        """
        cls._features.clear()
        cls._features_by_category.clear()
        cls._features_by_name.clear()


def register_feature(feature_cls: type[EEGFeature]) -> type[EEGFeature]:
    """
    Décorateur d'enregistrement automatique d'une feature.
    """
    return RegisteredFeatureProvider.register(feature_cls)