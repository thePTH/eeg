from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FeatureNameHelper:
    """
    Helper pour construire une liste de noms de features à partir d'inputs simples.

    Types de features supportés
    ---------------------------
    1) EEG scalaires
       Convention :
           <CHANNEL>_<EEG_FEATURE>
       Exemple :
           O1_relative_wavelet_energy

    2) Connectivité
       Convention :
           cn_<BAND>_<EDGE>
       Exemple :
           cn_alpha_O1_F7

    3) Subject
       Convention :
           subject_<FIELD>
       Exemples :
           subject_id
           subject_health
           subject_group
           subject_gender
           subject_age
           subject_mmse

    Exemples
    --------
    >>> helper = FeatureNameHelper(available_features)

    >>> helper.build(eeg="relative_wavelet_energy", channels="O1")
    ['O1_relative_wavelet_energy']

    >>> helper.build(cn="alpha", edges="O1_F7")
    ['cn_alpha_O1_F7']

    >>> helper.build(subject=["subject_age", "subject_mmse"])
    ['subject_age', 'subject_mmse']

    >>> helper.build(
    ...     eeg=["variance"],
    ...     channels=["O1", "O2"],
    ...     subject=["subject_age"]
    ... )
    ['O1_variance', 'O2_variance', 'subject_age']
    """

    available_features: list[str]

    def __post_init__(self) -> None:
        if not self.available_features:
            raise ValueError("`available_features` ne peut pas être vide.")

        eeg_channels: set[str] = set()
        eeg_feature_names: set[str] = set()

        cn_bands: set[str] = set()
        cn_edges: set[str] = set()

        subject_features: set[str] = set()

        for feature in self.available_features:
            parts = feature.split("_")

            # -----------------------------------------------------------------
            # Subject features : subject_xxx
            # -----------------------------------------------------------------
            if feature.startswith("subject_"):
                subject_features.add(feature)
                continue

            # -----------------------------------------------------------------
            # Connectivity features : cn_<band>_<edge>
            # Ex: cn_alpha_O1_F7
            # -----------------------------------------------------------------
            if len(parts) >= 4 and parts[0] == "cn":
                band = parts[1]
                edge = "_".join(parts[2:])
                cn_bands.add(band)
                cn_edges.add(edge)
                continue

            # -----------------------------------------------------------------
            # EEG features : <channel>_<feature_name>
            # Ex: O1_relative_wavelet_energy
            # -----------------------------------------------------------------
            if len(parts) >= 2 and parts[0] != "cn":
                channel = parts[0]
                eeg_feature = "_".join(parts[1:])
                eeg_channels.add(channel)
                eeg_feature_names.add(eeg_feature)

        object.__setattr__(self, "_available_set", set(self.available_features))
        object.__setattr__(self, "_eeg_channels", sorted(eeg_channels))
        object.__setattr__(self, "_eeg_feature_names", sorted(eeg_feature_names))
        object.__setattr__(self, "_cn_bands", sorted(cn_bands))
        object.__setattr__(self, "_cn_edges", sorted(cn_edges))
        object.__setattr__(self, "_subject_features", sorted(subject_features))

    @staticmethod
    def _normalize_to_list(
        value: str | Iterable[str] | None,
        field_name: str,
    ) -> list[str] | None:
        """
        Convertit :
        - None -> None
        - str -> [str]
        - iterable[str] -> list[str]
        """
        if value is None:
            return None

        if isinstance(value, str):
            items = [value]
        else:
            items = list(value)

        if len(items) == 0:
            raise ValueError(f"`{field_name}` ne peut pas être vide.")

        for item in items:
            if not isinstance(item, str):
                raise TypeError(
                    f"Tous les éléments de `{field_name}` doivent être des chaînes de caractères."
                )

        return items

    def build(
        self,
        *,
        eeg: str | Iterable[str] | None = None,
        cn: str | Iterable[str] | None = None,
        subject: str | Iterable[str] | None = None,
        channels: str | Iterable[str] | None = None,
        edges: str | Iterable[str] | None = None,
    ) -> list[str]:
        """
        Construit une liste de noms de features existants à partir d'inputs simples.

        Paramètres
        ----------
        eeg:
            Nom ou liste de noms de features EEG.
            Exemples :
                "relative_wavelet_energy"
                ["variance", "spectral_centroid"]

        cn:
            Bande ou liste de bandes de connectivité.
            Exemples :
                "alpha"
                ["delta", "theta", "alpha"]

        subject:
            Nom ou liste de noms de subject features.
            Exemples :
                "subject_age"
                ["subject_age", "subject_mmse"]

        channels:
            Canal ou liste de canaux pour les EEG features.
            Si None et `eeg` est renseigné, tous les canaux disponibles sont utilisés.

        edges:
            Edge ou liste d'edges pour les connectivity features.
            Si None et `cn` est renseigné, toutes les edges disponibles sont utilisées.

        Retour
        ------
        list[str]
            Liste des features existantes, sans doublons, en conservant l'ordre.

        Règles
        ------
        - Il faut renseigner au moins un parmi `eeg`, `cn`, `subject`.
        - `channels` ne peut être utilisé que si `eeg` est renseigné.
        - `edges` ne peut être utilisé que si `cn` est renseigné.
        """
        eeg = self._normalize_to_list(eeg, "eeg")
        cn = self._normalize_to_list(cn, "cn")
        subject = self._normalize_to_list(subject, "subject")
        channels = self._normalize_to_list(channels, "channels")
        edges = self._normalize_to_list(edges, "edges")

        if eeg is None and cn is None and subject is None:
            raise ValueError(
                "Il faut renseigner au moins un des champs `eeg`, `cn` ou `subject`."
            )

        if channels is not None and eeg is None:
            raise ValueError("`channels` ne peut être utilisé que si `eeg` est renseigné.")

        if edges is not None and cn is None:
            raise ValueError("`edges` ne peut être utilisé que si `cn` est renseigné.")

        result: list[str] = []

        # ---------------------------------------------------------------------
        # EEG
        # ---------------------------------------------------------------------
        if eeg is not None:
            unknown_eeg = sorted(set(eeg) - set(self._eeg_feature_names))
            if unknown_eeg:
                raise ValueError(
                    "Features EEG inconnues : "
                    f"{unknown_eeg}. "
                    f"Features EEG disponibles : {self._eeg_feature_names}"
                )

            eeg_channels = self._eeg_channels if channels is None else channels

            unknown_channels = sorted(set(eeg_channels) - set(self._eeg_channels))
            if unknown_channels:
                raise ValueError(
                    "Canaux inconnus : "
                    f"{unknown_channels}. "
                    f"Canaux disponibles : {self._eeg_channels}"
                )

            for channel in eeg_channels:
                for eeg_feature in eeg:
                    feature_name = f"{channel}_{eeg_feature}"
                    if feature_name in self._available_set:
                        result.append(feature_name)

        # ---------------------------------------------------------------------
        # CONNECTIVITY
        # ---------------------------------------------------------------------
        if cn is not None:
            unknown_cn = sorted(set(cn) - set(self._cn_bands))
            if unknown_cn:
                raise ValueError(
                    "Bandes de connectivité inconnues : "
                    f"{unknown_cn}. "
                    f"Bandes disponibles : {self._cn_bands}"
                )

            cn_edges = self._cn_edges if edges is None else edges

            unknown_edges = sorted(set(cn_edges) - set(self._cn_edges))
            if unknown_edges:
                raise ValueError(
                    "Arêtes inconnues : "
                    f"{unknown_edges}. "
                    f"Arêtes disponibles : {self._cn_edges}"
                )

            for band in cn:
                for edge in cn_edges:
                    feature_name = f"cn_{band}_{edge}"
                    if feature_name in self._available_set:
                        result.append(feature_name)

        # ---------------------------------------------------------------------
        # SUBJECT
        # ---------------------------------------------------------------------
        if subject is not None:
            unknown_subject = sorted(set(subject) - set(self._subject_features))
            if unknown_subject:
                raise ValueError(
                    "Subject features inconnues : "
                    f"{unknown_subject}. "
                    f"Subject features disponibles : {self._subject_features}"
                )

            for subject_feature in subject:
                if subject_feature in self._available_set:
                    result.append(subject_feature)

        # Suppression des doublons en conservant l'ordre
        return list(dict.fromkeys(result))

    @property
    def eeg_channels(self) -> list[str]:
        return list(self._eeg_channels)

    @property
    def eeg_feature_names(self) -> list[str]:
        return list(self._eeg_feature_names)

    @property
    def cn_bands(self) -> list[str]:
        return list(self._cn_bands)

    @property
    def cn_edges(self) -> list[str]:
        return list(self._cn_edges)

    @property
    def subject_features(self) -> list[str]:
        return list(self._subject_features)