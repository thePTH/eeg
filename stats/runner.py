from __future__ import annotations

from stats.queries import StatisticalQuery
from stats.results import StatisticalTestResultSet
from stats.bundles import SampleBundleFactory
from stats.engines.factory import StatisticalTestEngineFactory
from features.dataset import FeaturesDataset

class StatisticalTestRunner:
    """
    Point d'entrée unique du framework statistique.

    Responsabilités :
    - recevoir une query + un dataset
    - déterminer les keys à analyser
    - construire les SampleBundle
    - choisir le bon TestEngine
    - exécuter le test sur chaque key
    - renvoyer un ResultSet homogène
    """


    @staticmethod
    def run(query: StatisticalQuery, dataset: FeaturesDataset) -> StatisticalTestResultSet:
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