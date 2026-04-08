from __future__ import annotations

from abc import ABC, abstractmethod

from stats.results import CorrectedStatisticalResultSet, StatisticalResultSet


class MultipleComparisonCorrector(ABC):
    method_name: str

    @abstractmethod
    def correct(
        self,
        result_set: StatisticalResultSet,
        *,
        alpha: float,
        family_name: str,
    ) -> CorrectedStatisticalResultSet:
        raise NotImplementedError