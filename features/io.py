from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

from features.dataset import SingleParticipantProcessedFeatureDataset, FeaturesDataset


class SingleParticipantProcessedFeatureDatasetIO:

    @staticmethod
    def export(dataset: SingleParticipantProcessedFeatureDataset, path: str | Path):
        """
        Exporte un FeatureDataset dans un dossier structuré.

        Structure créée :
        path/
        ├── features.parquet
        ├── ppc.npy
        └── metadata.json
        """

        path = Path(path) / f"sub-{dataset.subject_dico['id']}"
        path.mkdir(parents=True, exist_ok=True)

        # Save dataframe
        dataset.features_df.to_parquet(path / "features.parquet")

        # Save PPC tensor
        np.save(path / "ppc.npy", dataset.ppc_raw_data)

        # Save metadata
        metadata = {
            "subject_dico": dataset.subject_dico,
            "pipeline_name": dataset.pipeline_name,
            "eeg_info_dico": dataset.eeg_info_dico,
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)

    @staticmethod
    def load(path: str | Path) -> SingleParticipantProcessedFeatureDataset:
        """
        Recharge un FeatureDataset depuis un dossier exporté.
        """

        path = Path(path)

        # Load dataframe
        features_df = pd.read_parquet(path / "features.parquet")

        # Load PPC tensor
        ppc_raw_data = np.load(path / "ppc.npy")

        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        return SingleParticipantProcessedFeatureDataset(
            features_df=features_df,
            ppc_raw_data=ppc_raw_data,
            subject_dico=metadata["subject_dico"],
            pipeline_name=metadata["pipeline_name"],
            eeg_info_dico=metadata["eeg_info_dico"],
        )
    


from pathlib import Path
class FeaturesDatasetIO:
    @staticmethod
    def load(folder_name_path):
        participant_datasets = []
        for dataset_folder_path in Path(folder_name_path).iterdir() :
            if dataset_folder_path.is_dir():
                partipant_dataset = SingleParticipantProcessedFeatureDatasetIO.load(dataset_folder_path)
                participant_datasets.append(partipant_dataset)

        return FeaturesDataset(participant_datasets)
