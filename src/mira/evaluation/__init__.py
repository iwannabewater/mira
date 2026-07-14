"""Exact and approximate retrieval evaluation with statistical comparisons."""

from mira.evaluation.evaluators import (
    EvaluationReport,
    ExactSearchEvaluator,
    ExternalANNEvaluator,
    FaissEvaluator,
    SearchResult,
)
from mira.evaluation.metrics import MetricResult, compute_retrieval_metrics
from mira.evaluation.statistics import (
    ConfidenceInterval,
    PairedBootstrapResult,
    bootstrap_confidence_interval,
    build_slice_masks,
    paired_bootstrap,
    slice_metrics,
)

__all__ = [
    "ConfidenceInterval",
    "EvaluationReport",
    "ExactSearchEvaluator",
    "ExternalANNEvaluator",
    "FaissEvaluator",
    "MetricResult",
    "PairedBootstrapResult",
    "SearchResult",
    "bootstrap_confidence_interval",
    "build_slice_masks",
    "compute_retrieval_metrics",
    "paired_bootstrap",
    "slice_metrics",
]
