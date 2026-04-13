from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from features.dataset import SingleParticipantProcessedFeatureDataset, FeaturesDataset


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

        # Recherche des enregistrements existants
        existing_indices = []

        for folder in path.iterdir():
            if folder.is_dir() and folder.name.startswith(subject_prefix):
                suffix = folder.name.replace(subject_prefix, "")
                if suffix.isdigit():
                    existing_indices.append(int(suffix))

        # Détermine le prochain index disponible
        next_index = 1 if not existing_indices else max(existing_indices) + 1

        recording_key = f"{next_index:02d}"

        export_folder = path / f"{subject_prefix}{recording_key}"
        export_folder.mkdir(parents=True, exist_ok=True)

        # Sauvegarde features
        dataset.features_df.to_parquet(export_folder / "features.parquet")

        # Sauvegarde PSD
        with open(export_folder / "psd_band_results.json", "w") as f:
            json.dump(dataset.psd_band_results, f)

        # Sauvegarde PPC
        with open(export_folder / "ppc_band_results.json", "w") as f:
            json.dump(dataset.ppc_band_results, f)

        metadata = {
            "subject_dico": dataset.subject_dico,
            "pipeline_name": dataset.pipeline_name,
            "eeg_info_dico": dataset.eeg_info_dico,
            "recording_key": recording_key,
        }

        with open(export_folder / "metadata.json", "w") as f:
            json.dump(metadata, f)

    @staticmethod
    def load(path: str | Path) -> SingleParticipantProcessedFeatureDataset:
        """
        Recharge un SingleParticipantProcessedFeatureDataset depuis un dossier exporté.
        """
        path = Path(path)

        features_df = pd.read_parquet(path / "features.parquet")

        with open(path / "psd_band_results.json", "r") as f:
            psd_band_results = json.load(f)

        with open(path / "ppc_band_results.json", "r") as f:
            ppc_band_results = json.load(f)

        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        return SingleParticipantProcessedFeatureDataset(
            features_df=features_df,
            psd_band_results=psd_band_results,
            ppc_band_results=ppc_band_results,
            subject_dico=metadata["subject_dico"],
            pipeline_name=metadata["pipeline_name"],
            eeg_info_dico=metadata["eeg_info_dico"],
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
    def load(folder_name_path: str | Path):
        participant_datasets = []
        folder = Path(folder_name_path)

        for dataset_folder_path in sorted(folder.iterdir()):
            if dataset_folder_path.is_dir():
                participant_dataset = SingleParticipantProcessedFeatureDatasetIO.load(
                    dataset_folder_path
                )
                participant_datasets.append(participant_dataset)

        return FeaturesDataset(participant_datasets)