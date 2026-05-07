from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features.dataset import SingleParticipantProcessedFeatureDataset, FeaturesDataset
from eeg.data import EEGProcessedDataIO

class SingleParticipantProcessedFeatureDatasetIO:
    @staticmethod
    def export(dataset: SingleParticipantProcessedFeatureDataset, path: str | Path):
        """
        Exporte un SingleParticipantProcessedFeatureDataset dans un dossier structuré.

        Structure créée
        ----------------
        path/
        └── sub-<id>-rec-XX/
            ├── features.parquet
            ├── psd_band_results.json
            ├── ppc_band_results.json
            └── metadata.json

        XX est automatiquement incrémenté selon les enregistrements déjà présents.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        subject_id = dataset.subject_dico["id"]
        subject_prefix = f"sub-{subject_id}-rec-"

        existing_indices = []

        for folder in path.iterdir():
            if folder.is_dir() and folder.name.startswith(subject_prefix):
                suffix = folder.name.replace(subject_prefix, "")
                if suffix.isdigit():
                    existing_indices.append(int(suffix))

        next_index = 1 if not existing_indices else max(existing_indices) + 1
        recording_key = f"{next_index:02d}"

        export_folder = path / f"{subject_prefix}{recording_key}"
        export_folder.mkdir(parents=True, exist_ok=True)

        # Features
        dataset.features_df.to_parquet(export_folder / "features.parquet")

        # PSD
        with open(export_folder / "psd_band_results.json", "w") as f:
            json.dump(dataset.psd_band_results, f)

        # PPC: conversion explicite ndarray -> list
        ppc_json_ready = {
            band_name: dataset.ppc_matrix(band_name).tolist()
            for band_name in dataset.ppc_band_names
        }
        with open(export_folder / "ppc_band_results.json", "w") as f:
            json.dump(ppc_json_ready, f)

        metadata = {
            "subject_dico": dataset.subject_dico,
            "pipeline_name": dataset.pipeline_name,
            "eeg_info_dico": dataset.eeg_info_dico,
            "recording_key": recording_key,
        }

        with open(export_folder / "metadata.json", "w") as f:
            json.dump(metadata, f)

    @staticmethod
    def load(feature_data_path: str | Path, eeg_data_path: str | Path=None) -> SingleParticipantProcessedFeatureDataset:
        """
        Recharge un SingleParticipantProcessedFeatureDataset depuis un dossier exporté.
        """
        feature_data_path = Path(feature_data_path)

        features_df = pd.read_parquet(feature_data_path / "features.parquet")
        features_df = features_df.astype(np.float32, copy=False)

        with open(feature_data_path / "psd_band_results.json", "r") as f:
            psd_band_results = json.load(f)

        with open(feature_data_path / "ppc_band_results.json", "r") as f:
            raw_ppc_band_results = json.load(f)

        # Conversion list -> ndarray float32
        ppc_band_results = {
            band_name: np.asarray(matrix, dtype=np.float32)
            for band_name, matrix in raw_ppc_band_results.items()
        }

        with open(feature_data_path / "metadata.json", "r") as f:
            metadata = json.load(f)


        eeg = None if eeg_data_path is None else EEGProcessedDataIO.load(eeg_data_path)

        return SingleParticipantProcessedFeatureDataset(
            features_df=features_df,
            psd_band_results=psd_band_results,
            ppc_band_results=ppc_band_results,
            subject_dico=metadata["subject_dico"],
            pipeline_name=metadata["pipeline_name"],
            eeg_info_dico=metadata["eeg_info_dico"],
            _eeg=eeg,
        )


class FeaturesDatasetIO:
    @staticmethod
    def export(dataset: FeaturesDataset, folder_name_path: str | Path):
        folder = Path(folder_name_path)
        folder.mkdir(parents=True, exist_ok=True)

        for participant_dataset in dataset.participant_datasets:
            SingleParticipantProcessedFeatureDatasetIO.export(
                participant_dataset,
                folder,
            )

    @staticmethod
    def load(
        feature_folder_name_path: str | Path,
        eeg_folder_name_path: str | Path | None = None,
    ) -> FeaturesDataset:
        participant_datasets = []

        feature_folder = Path(feature_folder_name_path)

        eeg_folder = (
            Path(eeg_folder_name_path)
            if eeg_folder_name_path is not None
            else None
        )

        for dataset_folder_path in sorted(feature_folder.iterdir()):
            if not dataset_folder_path.is_dir():
                continue

            eeg_dataset_folder_path = (
                eeg_folder / dataset_folder_path.name
                if eeg_folder is not None
                else None
            )

            participant_dataset = SingleParticipantProcessedFeatureDatasetIO.load(
                dataset_folder_path,
                eeg_data_path=eeg_dataset_folder_path,
            )

            participant_datasets.append(participant_dataset)

        return FeaturesDataset(participant_datasets)