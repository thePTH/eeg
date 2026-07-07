from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FeatureNameHelper:
    """
    Helper used to rebuild feature column names from feature families.

    Supported conventions
    ---------------------
    1) EEG:
        <CHANNEL>_<EEG_FAMILY>
        Example:
            O1_relative_wavelet_energy

    2) Connectivity:
        cn_<BAND>_<EDGE>
        Example:
            cn_alpha_O1_F7

    3) Subject:
        subject_<FIELD>
        Examples:
            subject_age
            subject_mmse

    Main usage
    ----------
    >>> helper.build(family_names="relative_wavelet_energy", channels="O1")
    ['O1_relative_wavelet_energy']

    >>> helper.build(family_names="alpha", edges="O1_F7")
    ['cn_alpha_O1_F7']

    >>> helper.build(family_names="subject_age")
    ['subject_age']

    Rules
    -----
    - ``family_names`` is the recommended API.
    - An EEG family uses ``channels`` if provided, otherwise all available channels.
    - A connectivity family uses ``edges`` if provided, otherwise all available edges.
    - A subject family uses neither ``channels`` nor ``edges``.
    - If a family is ambiguous, an explicit error is raised.
    """

    available_features: list[str]

    def __post_init__(self) -> None:
        """Parse available feature names and build lookup indexes."""
        if not self.available_features:
            raise ValueError("`available_features` ne peut pas être vide.")

        eeg_channels: set[str] = set()
        eeg_family_names: set[str] = set()

        cn_bands: set[str] = set()
        cn_edges: set[str] = set()

        subject_features: set[str] = set()

        family_to_kinds: dict[str, set[str]] = {}

        for feature in self.available_features:
            if not isinstance(feature, str) or not feature.strip():
                raise ValueError(
                    "Toutes les features disponibles doivent être des chaînes non vides."
                )

            parts = feature.split("_")

            if feature.startswith("subject_"):
                subject_features.add(feature)
                family_to_kinds.setdefault(feature, set()).add("subject")
                continue

            if len(parts) >= 4 and parts[0] == "cn":
                band = parts[1]
                edge = "_".join(parts[2:])

                cn_bands.add(band)
                cn_edges.add(edge)
                family_to_kinds.setdefault(band, set()).add("cn")
                continue

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
            {
                family: frozenset(kinds)
                for family, kinds in family_to_kinds.items()
            },
        )

    @staticmethod
    def _normalize_to_list(
        value: str | Iterable[str] | None,
        field_name: str,
    ) -> list[str] | None:
        """
        Normalize an optional string or iterable of strings to a list.

        Conversion rules:
        - None -> None
        - str -> [str]
        - Iterable[str] -> list[str]
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
        Resolve whether a family corresponds to EEG, connectivity, or subject data.

        Raises an explicit error if the family is unknown or ambiguous.
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
        Build a list of existing feature names.

        Recommended API
        ---------------
        family_names:
            Family name(s) to resolve automatically.
            Examples:
                "variance"    -> EEG
                "alpha"       -> connectivity
                "subject_age" -> subject

        channels:
            Channels to use for EEG families. If None, all available EEG
            channels are used.

        edges:
            Edges to use for connectivity families. If None, all available
            connectivity edges are used.

        Backward-compatible API
        -----------------------
        eeg, cn, subject:
            Explicit legacy arguments still supported for existing code.

        Returns
        -------
        list[str]
            Existing feature names, without duplicates and preserving order.
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
        """Return the available EEG channels."""
        return list(self._eeg_channels)

    @property
    def eeg_family_names(self) -> list[str]:
        """Return the available EEG feature families."""
        return list(self._eeg_family_names)

    @property
    def cn_bands(self) -> list[str]:
        """Return the available connectivity bands."""
        return list(self._cn_bands)

    @property
    def cn_edges(self) -> list[str]:
        """Return the available connectivity edges."""
        return list(self._cn_edges)

    @property
    def subject_features(self) -> list[str]:
        """Return the available subject-level features."""
        return list(self._subject_features)

    @property
    def family_names(self) -> list[str]:
        """Return all known families across EEG, connectivity, and subject features."""
        return sorted(self._family_to_kinds.keys())

    def family_kind(self, family_name: str) -> str:
        """
        Return the kind of a feature family.

        Possible values:
        - 'eeg'
        - 'cn'
        - 'subject'
        """
        if not isinstance(family_name, str) or not family_name.strip():
            raise ValueError("`family_name` doit être une chaîne non vide.")

        return self._resolve_family_kind(family_name.strip())