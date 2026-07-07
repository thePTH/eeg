from __future__ import annotations

from abc import ABC, abstractmethod

from stats.results import CorrectedStatisticalResultSet, StatisticalResultSet


class MultipleComparisonCorrector(ABC):
    """Abstract base class for multiple-comparison correction methods."""

    method_name: str

    @abstractmethod
    def correct(
        self,
        result_set: StatisticalResultSet,
        *,
        alpha: float,
        family_name: str,
    ) -> CorrectedStatisticalResultSet:
        """
        Apply a multiple-comparison correction to a statistical result set.

        Parameters
        ----------
        result_set : StatisticalResultSet
            Collection of statistical test results to be corrected.
        alpha : float
            Significance level used for the correction.
        family_name : str
            Name identifying the family of hypotheses being corrected.

        Returns
        -------
        CorrectedStatisticalResultSet
            Statistical results with corrected p-values.
        """
        raise NotImplementedError