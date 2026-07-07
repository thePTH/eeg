from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import ClassVar

from features.categories import FeatureCategory
from features.context import FeatureExtractionContext


class EEGExtractedFeature:
    """Represent a computed EEG feature value."""

    def __init__(self, name: str, category: FeatureCategory, value: float):
        self._name = name
        self._category = category
        self._value = value

    @property
    def name(self):
        """Return the feature name."""
        return self._name

    @property
    def category(self):
        """Return the feature category."""
        return self._category

    @property
    def value(self):
        """Return the extracted feature value."""
        return self._value

    def __repr__(self):
        """Return the string representation of the feature value."""
        return str(self.value)


class EEGFeature(ABC):
    """Abstract base class for EEG features."""

    name: ClassVar[str]
    category: ClassVar[FeatureCategory]

    @staticmethod
    @abstractmethod
    def _extract(context: FeatureExtractionContext) -> float:
        """Extract the feature value from a feature extraction context."""
        raise NotImplementedError

    @classmethod
    def compute(cls, context: FeatureExtractionContext) -> EEGExtractedFeature:
        """Compute and return the extracted EEG feature."""
        return EEGExtractedFeature(
            name=cls.name,
            category=cls.category,
            value=cls._extract(context),
        )


class RegisteredFeatureProvider:
    """
    Central registry for EEG feature classes.

    This registry stores classes decorated with @register_feature and exposes
    retrieval methods:
    - all features;
    - features by category;
    - features by name.
    """

    _features: list[type[EEGFeature]] = []
    _features_by_category: dict[FeatureCategory, list[type[EEGFeature]]] = defaultdict(list)
    _features_by_name: dict[str, type[EEGFeature]] = {}

    @classmethod
    def register(cls, feature_cls: type[EEGFeature]) -> type[EEGFeature]:
        """
        Register an EEG feature class in the global registry.

        Parameters
        ----------
        feature_cls : type[EEGFeature]
            Feature class to register.

        Returns
        -------
        type[EEGFeature]
            The feature class itself, allowing this method to be used as a decorator.
        """
        if not issubclass(feature_cls, EEGFeature):
            raise TypeError(
                f"{feature_cls.__name__} doit hériter de EEGFeature pour être enregistrée."
            )

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
        """Return all registered EEG feature classes."""
        return list(cls._features)

    @classmethod
    def get_by_category(cls, category: FeatureCategory) -> list[type[EEGFeature]]:
        """Return all registered EEG feature classes for a given category."""
        return list(cls._features_by_category.get(category, []))

    @classmethod
    def get_by_categories(
        cls,
        categories: list[FeatureCategory],
    ) -> list[type[EEGFeature]]:
        """Return all registered EEG feature classes for several categories."""
        selected_categories = set(categories)

        return [
            feature_cls
            for feature_cls in cls._features
            if feature_cls.category in selected_categories
        ]

    @classmethod
    def get_by_name(cls, name: str) -> type[EEGFeature]:
        """Return the registered EEG feature class associated with a name."""
        return cls._features_by_name[name]

    @classmethod
    def clear(cls) -> None:
        """Clear the registry, mainly for unit tests."""
        cls._features.clear()
        cls._features_by_category.clear()
        cls._features_by_name.clear()


def register_feature(feature_cls: type[EEGFeature]) -> type[EEGFeature]:
    """Decorator used to automatically register an EEG feature class."""
    return RegisteredFeatureProvider.register(feature_cls)