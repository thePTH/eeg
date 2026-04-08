from __future__ import annotations

from stats.engines.anova import OneWayANOVAEngine, TwoWayANOVAEngine
from stats.engines.base import StatisticalTestEngine
from stats.engines.spearman import SpearmanEngine
from stats.engines.ttest import TTestEngine
from stats.engines.wilcoxon import WilcoxonRankSumEngine
from stats.queries.base import StatisticalQuery


class StatisticalTestEngineFactory:
    """
    Factory interne : choisit le bon engine à partir de la query.
    """

    @staticmethod
    def build(query: StatisticalQuery) -> StatisticalTestEngine:
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