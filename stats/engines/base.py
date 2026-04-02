from __future__ import annotations

from abc import ABC, abstractmethod

from stats.bundles import SampleBundle
from stats.results import StatisticalTestResult


class StatisticalTestEngine(ABC):
    """
    Un engine sait appliquer une formule statistique à un SampleBundle.
    """
    test_name: str

    @abstractmethod
    def compute(self, bundle: SampleBundle, *, target: str, key: str) -> StatisticalTestResult:
        raise NotImplementedError

    def _build_result(
        self,
        bundle: SampleBundle,
        *,
        statistic: float,
        p_value: float,
        target: str,
        key: str,
    ) -> StatisticalTestResult:
        """
        Construit un résultat standardisé en propageant automatiquement
        les métadonnées du bundle (tailles d'échantillon et libellés).

        Cette méthode centralise la logique commune à tous les engines.
        """
        return StatisticalTestResult(
            statistic=float(statistic),
            p_value=float(p_value),
            target=target,
            key=key,
            test_name=self.test_name,
            n_x=bundle.n_x,
            n_y=bundle.n_y,
            x_name=bundle.x_name,
            y_name=bundle.y_name,
        )