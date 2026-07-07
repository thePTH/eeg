from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from features.dataset import FeaturesDataset
from stats.bundles import OneWayANOVASampleBundle, SampleBundleFactory
from stats.correction.fdr import FDRCorrector
from stats.engines.factory import StatisticalTestEngineFactory
from stats.engines.tukey import TukeyHSDPostHocEngine
from stats.queries.base import StatisticalQuery
from stats.results import StatisticalAnalysisOutcome, StatisticalResultSet


class StatisticalTestRunner:
    """
    Main entry point of the statistical framework.

    Workflow:
    1. The query describes the domain-level intent.
    2. The factory builds the sample bundles.
    3. The engine applies the primary statistical test.
    4. The runner optionally applies:
       - multiple-comparison correction;
       - post-hoc analysis.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def run(
        query: StatisticalQuery | dict[Any, StatisticalQuery],
        dataset: FeaturesDataset,
    ) -> StatisticalAnalysisOutcome | dict[Any, StatisticalAnalysisOutcome]:
        """Run a statistical query or a dictionary of statistical queries."""
        if isinstance(query, Mapping):
            return {
                key: StatisticalTestRunner.run(subquery, dataset)
                for key, subquery in query.items()
            }

        primary_results = StatisticalTestRunner.run_primary(query, dataset)
        corrected_results = StatisticalTestRunner._maybe_correct(query, primary_results)
        posthoc_results = StatisticalTestRunner._maybe_run_posthoc(
            query,
            dataset,
            primary_results,
        )

        return StatisticalAnalysisOutcome(
            primary_results=primary_results,
            corrected_results=corrected_results,
            posthoc_results=posthoc_results,
        )

    @staticmethod
    def run_primary(
        query: StatisticalQuery,
        dataset: FeaturesDataset,
    ) -> StatisticalResultSet:
        """Run only the primary statistical test for a query."""
        engine = StatisticalTestEngineFactory.build(query)
        keys = SampleBundleFactory.list_keys(query, dataset)

        results = {}

        for key in keys:
            bundle = SampleBundleFactory.build(query, dataset, key)
            result = engine.compute(bundle, target=query.target_name, key=key)
            results[key] = result

        return StatisticalResultSet(
            results=results,
            test_name=engine.test_name,
            target=query.target_name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_correct(
        query: StatisticalQuery,
        primary_results: StatisticalResultSet,
    ):
        """Apply multiple-comparison correction when requested by the query."""
        if query.correction is None:
            return None

        match query.correction.method:
            case "fdr_bh":
                corrector = FDRCorrector()
            case _:
                raise ValueError(
                    f"Unsupported correction method: {query.correction.method}"
                )

        return corrector.correct(
            primary_results,
            alpha=query.correction.alpha,
            family_name=query.correction.family_name,
        )

    @staticmethod
    def _maybe_run_posthoc(
        query: StatisticalQuery,
        dataset: FeaturesDataset,
        primary_results: StatisticalResultSet,
    ) -> dict[str, Any] | None:
        """Run post-hoc analysis when requested by the query."""
        posthoc_spec = getattr(query, "posthoc", None)

        if posthoc_spec is None:
            return None

        if query.test_kind != "one_way_anova":
            raise ValueError(
                "Post-hoc is currently supported only after one_way_anova. "
                "For two_way_anova, add a dedicated estimated-marginal-means / "
                "simple-effects layer later."
            )

        match posthoc_spec.method:
            case "tukey_hsd":
                engine = TukeyHSDPostHocEngine()
            case _:
                raise ValueError(f"Unsupported posthoc method: {posthoc_spec.method}")

        posthoc_results = {}
        keys = SampleBundleFactory.list_keys(query, dataset)

        for key in keys:
            primary_result = primary_results.results[key]

            if posthoc_spec.only_if_omnibus_significant:
                if (
                    not hasattr(primary_result, "p_value")
                    or primary_result.p_value >= posthoc_spec.alpha
                ):
                    continue

            bundle = SampleBundleFactory.build(query, dataset, key)

            if not isinstance(bundle, OneWayANOVASampleBundle):
                raise TypeError(
                    "Tukey HSD expects a OneWayANOVASampleBundle. "
                    "This indicates an inconsistency between query and bundle construction."
                )

            posthoc_results[key] = engine.compute(
                bundle,
                target=query.target_name,
                key=key,
                alpha=posthoc_spec.alpha,
            )

        return posthoc_results or None