from __future__ import annotations

from stats.queries import StatisticalQuery
from stats.results import StatisticalTestResultSet
from stats.bundles import SampleBundleFactory
from stats.engines.factory import StatisticalTestEngineFactory
from features.dataset import FeaturesDataset
from collections.abc import Mapping
from typing import Union, Any


class StatisticalTestRunner:
    """
    Point d'entrée unique du framework statistique.

    Supporte :
    - une query simple -> renvoie un StatisticalTestResultSet
    - un dict[str, query] -> renvoie un dict[str, StatisticalTestResultSet]
    """

    @staticmethod
    def run(query:Union[StatisticalQuery, dict[Any, StatisticalQuery]], dataset: FeaturesDataset) -> Union[StatisticalTestResultSet, dict[Any, StatisticalTestResultSet]]:
        if isinstance(query, Mapping):
            return {
                key: StatisticalTestRunner.run(subquery, dataset)
                for key, subquery in query.items()
            }

        engine = StatisticalTestEngineFactory.build(query)
        keys = SampleBundleFactory.list_keys(query, dataset)

        results = {}

        for key in keys:
            bundle = SampleBundleFactory.build(query, dataset, key)
            result = engine.compute(bundle, target=query.target_name, key=key)
            results[key] = result

        return StatisticalTestResultSet(
            results=results,
            test_name=engine.test_name,
            target=query.target_name,
        )