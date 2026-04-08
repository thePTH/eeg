from dataclasses import dataclass

from .types import CorrectionKind, PostHocKind


@dataclass(frozen=True, kw_only=True)
class CorrectionSpec:
    """
    Décrit une correction pour comparaisons multiples.

    Parameters
    ----------
    method:
        Méthode de correction, par exemple 'fdr_bh'.
    alpha:
        Niveau alpha.
    family_name:
        Nom métier de la famille de tests corrigée.

    Exemples
    --------
    - 'channels'
    - 'edges'
    - 'frequency_bins'
    """
    method: CorrectionKind = "fdr_bh"
    alpha: float = 0.05
    family_name: str = "default"


@dataclass(frozen=True, kw_only=True)
class PostHocSpec:
    """
    Décrit un post-hoc à exécuter après un test omnibus.
    """
    method: PostHocKind = "tukey_hsd"
    alpha: float = 0.05
    only_if_omnibus_significant: bool = True