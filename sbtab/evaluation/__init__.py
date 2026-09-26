"""
Canonical metric package of the benchmark protocol (``sbtab.metrics/1``).

Everything that is learned (histogram edges, training supports, conditioning
definitions, MMD scales / bandwidth / row selection) is learned by
``MetricContext.fit`` from the TRAINING rows only and then applied unchanged to the
real held-out rows and to every generated table of that fold.
"""
from ._common import json_safe
from .association import association_metrics
from .conditional import conditional_metrics
from .context import MetricContext
from .marginal import marginal_metrics, tuning_objective
from .mmd import mmd_metrics
from .spec import METRIC_VERSION, STATUSES, MetricConfig
from .utility import resolve_utility_params, utility_gap, utility_reference, utility_tstr
from .validity import check_validity

__all__ = [
    "METRIC_VERSION", "STATUSES", "MetricConfig", "MetricContext",
    "check_validity", "tuning_objective", "marginal_metrics", "association_metrics",
    "conditional_metrics", "mmd_metrics",
    "resolve_utility_params", "utility_reference", "utility_tstr", "utility_gap", "json_safe",
]
