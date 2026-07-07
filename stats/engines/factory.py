from __future__ import annotations

from stats.engines.anova import OneWayANOVAEngine, TwoWayANOVAEngine
from stats.engines.base import StatisticalTestEngine
from stats.engines.spearman import SpearmanEngine
from stats.engines.ttest import TTestEngine
from stats.engines.wilcoxon import WilcoxonRankSumEngine
from stats.queries.base import StatisticalQuery


class StatisticalTestEngineFactory:
    """
    Factory responsible for selecting the appropriate statistical engine
    according to the requested statistical query.
    """

    @staticmethod
    def build(query: StatisticalQuery) -> StatisticalTestEngine:
        """
        Build the statistical engine corresponding to the query.

        Parameters
        ----------
        query : StatisticalQuery
            Statistical query describing the requested test.

        Returns
        -------
        StatisticalTestEngine
            Engine implementing the requested statistical test.

        Raises
        ------
        ValueError
            If the requested test kind is not supported.
        """
        match query.test_kind:
            case "t_test":
                return TTestEngine(equal_var=False)

            case "wilcoxon_rank_sum":
                return WilcoxonRankSumEngine()

            case "spearman":
                return SpearmanEngine()

            case "one_way_anova":
                return OneWayANOVAEngine()

            case "two_way_anova":
                return TwoWayANOVAEngine()

            case _:
                raise ValueError(f"Unsupported test kind: {query.test_kind}")