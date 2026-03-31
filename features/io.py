from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.factory import FeatureExtractionResult


class FeatureExtractionResultIO:
    """
    Classe utilitaire dédiée à l'export et au chargement d'un FeatureExtractionResult.

    Objectif
    --------
    - Exporter un résultat complet dans un dossier structuré
    - Pouvoir reconstruire ensuite exactement un FeatureExtractionResult
    - Garder aussi des fichiers lisibles pour inspection humaine

    Stratégie
    ---------
    On mélange deux approches :
    1. Pickle pour reconstruire fidèlement les objets Python complexes
    2. CSV / JSON / NPZ pour permettre l'inspection manuelle des données exportées

    Structure du dossier exporté
    ----------------------------
    folder/
    ├── manifest.json
    ├── features.csv
    ├── objects/
    │   ├── eeg.pkl
    │   ├── config.pkl
    │   ├── features_dico.pkl
    │   ├── psd_result_dico.pkl
    │   └── ppc_result.pkl
    ├── psd/
    │   ├── Fp1.npz
    │   ├── Fp2.npz
    │   └── ...
    └── ppc/
        ├── delta_matrix.csv
        ├── theta_matrix.csv
        └── ...
    """

    EXPORT_VERSION = "1.0"

    @classmethod
    def export(cls, result: "FeatureExtractionResult", folder_path_name: str) -> Path:
        """
        Exporte un FeatureExtractionResult dans un dossier.

        Parameters
        ----------
        result :
            Objet à exporter.
        folder_path_name :
            Chemin du dossier d'export.

        Returns
        -------
        Path
            Le chemin du dossier exporté.
        """
        base = Path(folder_path_name)
        base.mkdir(parents=True, exist_ok=True)

        objects_dir = base / "objects"
        psd_dir = base / "psd"
        ppc_dir = base / "ppc"

        objects_dir.mkdir(exist_ok=True)
        psd_dir.mkdir(exist_ok=True)
        ppc_dir.mkdir(exist_ok=True)

        # ------------------------------------------------------------------
        # 1) Export des features dans un CSV lisible
        # ------------------------------------------------------------------
        df = result.dataframe.copy()
        df.index.name = "signal_name"
        df.to_csv(base / "features.csv")

        # ------------------------------------------------------------------
        # 2) Export fidèle des objets nécessaires à la reconstruction
        # ------------------------------------------------------------------
        cls._pickle_dump(result.eeg, objects_dir / "eeg.pkl")
        cls._pickle_dump(result.config, objects_dir / "config.pkl")
        cls._pickle_dump(result._features_dico, objects_dir / "features_dico.pkl")
        cls._pickle_dump(result._psd_result_dico, objects_dir / "psd_result_dico.pkl")
        cls._pickle_dump(result.ppc_result, objects_dir / "ppc_result.pkl")

        # ------------------------------------------------------------------
        # 3) Export lisible des PSD par signal
        # ------------------------------------------------------------------
        psd_index = []

        for signal, psd_result in result._psd_result_dico.items():
            signal_name = getattr(signal, "name", str(signal))
            safe_signal_name = cls._sanitize_filename(signal_name)
            file_name = f"{safe_signal_name}.npz"

            np.savez_compressed(
                psd_dir / file_name,
                freqs=np.asarray(psd_result.freqs, dtype=float),
                psd=np.asarray(psd_result.psd, dtype=float),
            )

            psd_index.append(
                {
                    "signal_name": signal_name,
                    "file": f"psd/{file_name}",
                    "n_freqs": int(len(psd_result.freqs)),
                }
            )

        # ------------------------------------------------------------------
        # 4) Export lisible des matrices PPC
        # ------------------------------------------------------------------
        ppc_summary: dict[str, Any] = {}

        if result.ppc_result is not None:
            channel_names = list(result.ppc_result.channel_names)
            band_names = list(result.ppc_result.band_names)

            ppc_summary["channel_names"] = channel_names
            ppc_summary["band_names"] = band_names

            for band_name in band_names:
                matrix = result.ppc_result.band_matrix(band_name)
                matrix_df = pd.DataFrame(
                    matrix,
                    index=channel_names,
                    columns=channel_names,
                )
                matrix_file_name = f"{cls._sanitize_filename(band_name)}_matrix.csv"
                matrix_df.to_csv(ppc_dir / matrix_file_name)

                ppc_summary[band_name] = {
                    "shape": list(matrix.shape),
                    "mean_connectivity": float(
                        result.ppc_result.mean_band_connectivity(band_name)
                    ),
                }

        # ------------------------------------------------------------------
        # 5) Manifest JSON
        # ------------------------------------------------------------------
        manifest = {
            "export_version": cls.EXPORT_VERSION,
            "class_name": "FeatureExtractionResult",
            "n_signals": len(result._features_dico),
            "feature_names": list(result.feature_names),
            "files": {
                "features_csv": "features.csv",
                "eeg_pickle": "objects/eeg.pkl",
                "config_pickle": "objects/config.pkl",
                "features_dico_pickle": "objects/features_dico.pkl",
                "psd_result_dico_pickle": "objects/psd_result_dico.pkl",
                "ppc_result_pickle": "objects/ppc_result.pkl",
            },
            "psd_index": psd_index,
            "ppc_summary": ppc_summary,
        }

        with open(base / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return base

    @classmethod
    def load(cls, folder_path_name: str) -> "FeatureExtractionResult":
        """
        Recharge un FeatureExtractionResult depuis un dossier exporté.

        Parameters
        ----------
        folder_path_name :
            Chemin du dossier exporté.

        Returns
        -------
        FeatureExtractionResult
            Objet reconstruit.
        """
        base = Path(folder_path_name)
        objects_dir = base / "objects"
        manifest_path = base / "manifest.json"

        if not base.exists():
            raise FileNotFoundError(f"Dossier introuvable : {base}")

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Le fichier manifest.json est introuvable dans : {base}"
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        export_version = manifest.get("export_version")
        if export_version != cls.EXPORT_VERSION:
            raise ValueError(
                f"Version d'export non supportée : {export_version}. "
                f"Version attendue : {cls.EXPORT_VERSION}"
            )

        eeg = cls._pickle_load(objects_dir / "eeg.pkl")
        config = cls._pickle_load(objects_dir / "config.pkl")
        features_dico = cls._pickle_load(objects_dir / "features_dico.pkl")
        psd_result_dico = cls._pickle_load(objects_dir / "psd_result_dico.pkl")
        ppc_result = cls._pickle_load(objects_dir / "ppc_result.pkl")

        return FeatureExtractionResult(
            eeg=eeg,
            extraction_config=config,
            features_dico=features_dico,
            psd_result_dico=psd_result_dico,
            ppc_result=ppc_result,
        )

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """
        Transforme une chaîne en nom de fichier sûr.
        """
        name = name.strip().replace(" ", "_")
        name = re.sub(r"[^A-Za-z0-9_.\-]", "_", name)
        return name

    @staticmethod
    def _pickle_dump(obj: Any, path: Path) -> None:
        """
        Sauvegarde un objet Python avec pickle.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _pickle_load(path: Path) -> Any:
        """
        Recharge un objet Python pickle.
        """
        with open(path, "rb") as f:
            return pickle.load(f)