"""
Package `features.dataset`

Ce package regroupe toutes les structures de données utilisées pour
représenter les features EEG après extraction, ainsi que les outils
de sélection, filtrage et transformation associés.

Contenu principal
-----------------
SingleParticipantProcessedFeatureDataset
    Dataset sujet-level après extraction complète.

FeaturesDataset
    Dataset global multi-participants avec vues tabulaires (wide / long).

SelectedFeaturesDataset
    Vue restreinte d’un dataset après sélection de colonnes ML.

SampleSelector
    Helper métier pour sélectionner des familles de features, filtrer
    des sujets et effectuer des splits train/test/val.

FeaturesDatasetSelector
    Helper bas niveau pour sélectionner explicitement des colonnes wide.

SingleParticipantProcessedFeatureDatasetFactory
    Factory de construction d’un dataset sujet-level à partir d’un
    résultat complet d’extraction.
"""

from .participant import SingleParticipantProcessedFeatureDataset
from .base import FeaturesDataset
from .selected import SelectedFeaturesDataset, FeaturesDatasetSelector, SelectedFeature, SelectedFeaturesDatasetFactory, SelectedFeaturesConcatEngine
from .selector import SampleSelector
from .factory import SingleParticipantProcessedFeatureDatasetFactory


__all__ = [
    "SingleParticipantProcessedFeatureDataset",
    "FeaturesDataset",
    "SelectedFeaturesDataset",
    "SelectedFeature",
    "SelectedFeaturesDatasetFactory",
    "SelectedFeaturesConcatEngine",
    "FeaturesDatasetSelector",
    "SampleSelector",
    "SingleParticipantProcessedFeatureDatasetFactory",
]