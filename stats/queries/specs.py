from dataclasses import dataclass

from .types import CorrectionKind, PostHocKind


@dataclass(frozen=True, kw_only=True)
class CorrectionSpec:
    """
    Specification of a multiple-comparison correction.

    Parameters
    ----------
    method : CorrectionKind
        Multiple-testing correction method (e.g. ``"fdr_bh"``).
    alpha : float
        Significance level used for the correction.
    family_name : str
        Name of the family of statistical tests to be corrected.

    Examples
    --------
    - ``"channels"``
    - ``"edges"``
    - ``"frequency_bins"``
    """

    method: CorrectionKind = "fdr_bh"
    alpha: float = 0.05
    family_name: str = "default"


@dataclass(frozen=True, kw_only=True)
class PostHocSpec:
    """
    Specification of a post-hoc analysis to perform after an omnibus test.

    Parameters
    ----------
    method : PostHocKind
        Post-hoc procedure to apply (e.g. ``"tukey_hsd"``).
    alpha : float
        Significance level used for the post-hoc comparisons.
    only_if_omnibus_significant : bool
        If ``True``, the post-hoc test is performed only when the omnibus
        test is statistically significant.
    """

    method: PostHocKind = "tukey_hsd"
    alpha: float = 0.05
    only_if_omnibus_significant: bool = True