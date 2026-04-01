from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.dataset import FeatureDataset


import json
import numpy as np
import pandas as pd
from pathlib import Path


class FeatureDatasetIO:

    @staticmethod
    def export(dataset:FeatureDataset, path: str | Path):
        """
        Exporte un FeatureDataset dans un dossier structuré.

        Structure créée :
        path/
        ├── features.parquet
        ├── ppc.npy
        └── metadata.json
        """
        path += f"/subject-{dataset.subject_dico["id"]}/"
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save dataframe
        dataset.features_df.to_parquet(path / "features.parquet")


        # Save PPC tensor
        np.save(path / "ppc.npy", dataset.ppc_raw_data)

        # Save metadata
        metadata = {
            "subject_dico": dataset.subject_dico,
            "pipeline_name": dataset.pipeline_name,
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)


    @staticmethod
    def load(path: str | Path):
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

        

        return FeatureDataset(
            features_df=features_df,
            ppc_raw_data=ppc_raw_data,
            subject_dico=metadata["subject_dico"],
            pipeline_name=metadata["pipeline_name"],
        )

