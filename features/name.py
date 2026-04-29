from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FeatureNameHelper:
    """
    Helper permettant de reconstruire des noms de colonnes de features
    à partir de familles de features.

    Conventions supportées
    ----------------------
    1) EEG :
        <CHANNEL>_<EEG_FAMILY>
        Exemple :
            O1_relative_wavelet_energy

    2) Connectivité :
        cn_<BAND>_<EDGE>
        Exemple :
            cn_alpha_O1_F7

    3) Subject :
        subject_<FIELD>
        Exemple :
            subject_age
            subject_mmse

    Utilisation principale
    ----------------------
    On peut désormais appeler :

    >>> helper.build(family_names="relative_wavelet_energy", channels="O1")
    ['O1_relative_wavelet_energy']

    >>> helper.build(family_names="alpha", edges="O1_F7")
    ['cn_alpha_O1_F7']

    >>> helper.build(family_names="subject_age")
    ['subject_age']

    >>> helper.build(
    ...     family_names=["variance", "alpha", "subject_age"],
    ...     channels=["O1", "O2"],
    ...     edges=["O1_F7"]
    ... )
    ['O1_variance', 'O2_variance', 'cn_alpha_O1_F7', 'subject_age']

    Règles
    ------
    - `family_names` est l'API recommandée.
    - Une famille EEG utilise `channels` si fourni, sinon tous les canaux disponibles.
    - Une famille CN utilise `edges` si fourni, sinon toutes les arêtes disponibles.
    - Une famille subject n'utilise ni `channels` ni `edges`.
    - Si une famille est ambiguë (par ex. même nom existant en EEG et CN), on lève
      une erreur explicite.
    """

    available_features: list[str]

    def __post_init__(self) -> None:
        if not self.available_features:
            raise ValueError("`available_features` ne peut pas être vide.")

        eeg_channels: set[str] = set()
        eeg_family_names: set[str] = set()

        cn_bands: set[str] = set()
        cn_edges: set[str] = set()

        subject_features: set[str] = set()

        # Mapping famille -> types possibles {"eeg", "cn", "subject"}
        family_to_kinds: dict[str, set[str]] = {}

        for feature in self.available_features:
            if not isinstance(feature, str) or not feature.strip():
                raise ValueError("Toutes les features disponibles doivent être des chaînes non vides.")

            parts = feature.split("_")

            # -------------------------------------------------------------
            # SUBJECT : subject_xxx
            # -------------------------------------------------------------
            if feature.startswith("subject_"):
                subject_features.add(feature)
                family_to_kinds.setdefault(feature, set()).add("subject")
                continue

            # -------------------------------------------------------------
            # CONNECTIVITY : cn_<band>_<edge>
            # Exemple : cn_alpha_O1_F7
            # -------------------------------------------------------------
            if len(parts) >= 4 and parts[0] == "cn":
                band = parts[1]
                edge = "_".join(parts[2:])

                cn_bands.add(band)
                cn_edges.add(edge)
                family_to_kinds.setdefault(band, set()).add("cn")
                continue

            # -------------------------------------------------------------
            # EEG : <channel>_<family>
            # Exemple : O1_relative_wavelet_energy
            # -------------------------------------------------------------
            if len(parts) >= 2 and parts[0] != "cn":
                channel = parts[0]
                family_name = "_".join(parts[1:])

                eeg_channels.add(channel)
                eeg_family_names.add(family_name)
                family_to_kinds.setdefault(family_name, set()).add("eeg")
                continue

        object.__setattr__(self, "_available_set", set(self.available_features))
        object.__setattr__(self, "_eeg_channels", sorted(eeg_channels))
        object.__setattr__(self, "_eeg_family_names", sorted(eeg_family_names))
        object.__setattr__(self, "_cn_bands", sorted(cn_bands))
        object.__setattr__(self, "_cn_edges", sorted(cn_edges))
        object.__setattr__(self, "_subject_features", sorted(subject_features))
        object.__setattr__(
            self,
            "_family_to_kinds",
            {family: frozenset(kinds) for family, kinds in family_to_kinds.items()},
        )

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

        cleaned_items: list[str] = []
        for item in items:
            if not isinstance(item, str):
                raise TypeError(
                    f"Tous les éléments de `{field_name}` doivent être des chaînes de caractères."
                )
            stripped = item.strip()
            if not stripped:
                raise ValueError(
                    f"Tous les éléments de `{field_name}` doivent être des chaînes non vides."
                )
            cleaned_items.append(stripped)

        return cleaned_items

    def _resolve_family_kind(self, family_name: str) -> str:
        """
        Détermine automatiquement si une famille correspond à :
        - 'eeg'
        - 'cn'
        - 'subject'

        Lève une erreur si la famille est inconnue ou ambiguë.
        """
        kinds = self._family_to_kinds.get(family_name)

        if kinds is None:
            raise ValueError(
                f"Famille inconnue : '{family_name}'. "
                f"Familles EEG disponibles : {self._eeg_family_names}. "
                f"Familles CN disponibles : {self._cn_bands}. "
                f"Subject features disponibles : {self._subject_features}."
            )

        if len(kinds) > 1:
            raise ValueError(
                f"La famille '{family_name}' est ambiguë : elle peut correspondre à {sorted(kinds)}. "
                "Utilise une méthode plus explicite ou renomme les familles pour lever l'ambiguïté."
            )

        return next(iter(kinds))

    def build(
        self,
        *,
        family_names: str | Iterable[str] | None = None,
        channels: str | Iterable[str] | None = None,
        edges: str | Iterable[str] | None = None,
        eeg: str | Iterable[str] | None = None,
        cn: str | Iterable[str] | None = None,
        subject: str | Iterable[str] | None = None,
    ) -> list[str]:
        """
        Construit une liste de noms de features existants.

        API recommandée
        ---------------
        family_names:
            Nom(s) de famille à résoudre automatiquement.
            Exemples :
                "variance"       -> EEG
                "alpha"          -> CN
                "subject_age"    -> subject

        channels:
            Canaux à utiliser pour les familles EEG.
            Si None, tous les canaux EEG disponibles sont utilisés.

        edges:
            Arêtes à utiliser pour les familles CN.
            Si None, toutes les arêtes CN disponibles sont utilisées.

        Compatibilité ancienne API
        --------------------------
        eeg, cn, subject:
            Toujours supportés pour rester compatible avec l'ancien code.

        Retour
        ------
        list[str]
            Liste des features existantes, sans doublons et en conservant l'ordre.
        """
        family_names = self._normalize_to_list(family_names, "family_names")
        channels = self._normalize_to_list(channels, "channels")
        edges = self._normalize_to_list(edges, "edges")

        eeg = self._normalize_to_list(eeg, "eeg")
        cn = self._normalize_to_list(cn, "cn")
        subject = self._normalize_to_list(subject, "subject")

        if (
            family_names is None
            and eeg is None
            and cn is None
            and subject is None
        ):
            raise ValueError(
                "Il faut renseigner au moins un des champs "
                "`family_names`, `eeg`, `cn` ou `subject`."
            )

        result: list[str] = []

        # ------------------------------------------------------------------
        # Nouvelle API : family_names
        # ------------------------------------------------------------------
        if family_names is not None:
            unknown_channels = []
            if channels is not None:
                unknown_channels = sorted(set(channels) - set(self._eeg_channels))
                if unknown_channels:
                    raise ValueError(
                        "Canaux inconnus : "
                        f"{unknown_channels}. "
                        f"Canaux disponibles : {self._eeg_channels}"
                    )

            unknown_edges = []
            if edges is not None:
                unknown_edges = sorted(set(edges) - set(self._cn_edges))
                if unknown_edges:
                    raise ValueError(
                        "Arêtes inconnues : "
                        f"{unknown_edges}. "
                        f"Arêtes disponibles : {self._cn_edges}"
                    )

            for family_name in family_names:
                kind = self._resolve_family_kind(family_name)

                if kind == "eeg":
                    eeg_channels = self._eeg_channels if channels is None else channels
                    for channel in eeg_channels:
                        feature_name = f"{channel}_{family_name}"
                        if feature_name in self._available_set:
                            result.append(feature_name)

                elif kind == "cn":
                    cn_edges = self._cn_edges if edges is None else edges
                    for edge in cn_edges:
                        feature_name = f"cn_{family_name}_{edge}"
                        if feature_name in self._available_set:
                            result.append(feature_name)

                elif kind == "subject":
                    if family_name in self._available_set:
                        result.append(family_name)

                else:
                    raise RuntimeError(f"Type de famille inattendu : {kind}")

        # ------------------------------------------------------------------
        # Ancienne API explicite : EEG
        # ------------------------------------------------------------------
        if eeg is not None:
            unknown_eeg = sorted(set(eeg) - set(self._eeg_family_names))
            if unknown_eeg:
                raise ValueError(
                    "Features EEG inconnues : "
                    f"{unknown_eeg}. "
                    f"Features EEG disponibles : {self._eeg_family_names}"
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
                for eeg_family in eeg:
                    feature_name = f"{channel}_{eeg_family}"
                    if feature_name in self._available_set:
                        result.append(feature_name)

        # ------------------------------------------------------------------
        # Ancienne API explicite : CONNECTIVITY
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Ancienne API explicite : SUBJECT
        # ------------------------------------------------------------------
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

        return list(dict.fromkeys(result))

    @property
    def eeg_channels(self) -> list[str]:
        return list(self._eeg_channels)

    @property
    def eeg_family_names(self) -> list[str]:
        return list(self._eeg_family_names)

    @property
    def cn_bands(self) -> list[str]:
        return list(self._cn_bands)

    @property
    def cn_edges(self) -> list[str]:
        return list(self._cn_edges)

    @property
    def subject_features(self) -> list[str]:
        return list(self._subject_features)

    @property
    def family_names(self) -> list[str]:
        """
        Toutes les familles connues, tous types confondus.
        """
        return sorted(self._family_to_kinds.keys())

    def family_kind(self, family_name: str) -> str:
        """
        Retourne le type d'une famille :
        - 'eeg'
        - 'cn'
        - 'subject'
        """
        if not isinstance(family_name, str) or not family_name.strip():
            raise ValueError("`family_name` doit être une chaîne non vide.")
        return self._resolve_family_kind(family_name.strip())