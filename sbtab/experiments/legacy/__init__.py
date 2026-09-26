"""
LEGACY namespace -- frozen historical experiment scripts and result artifacts.

Everything below this package is superseded by the canonical stages
``python -m sbtab.experiments.{prepare_splits,tune,cross_validate,calculate_metrics,
aggregate_results}`` and the canonical metric package ``sbtab.evaluation``
(``sbtab.metrics/1``). Metric definitions used here are ``legacy/0`` and live in
``sbtab.experiments.legacy.legacy_metrics``. Legacy and canonical results must never
be mixed. See ``README.md`` in this directory for status and provenance.

Nothing is imported here on purpose: importing the package must stay free of
solver / baseline side effects.
"""
