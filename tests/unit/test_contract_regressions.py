"""Small cross-package regressions found during adversarial review."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from mira.data.schema import RetrievalMetadata
from mira.observability import CollapseThresholds, analyze_embedding_health


def test_sampling_weight_must_be_finite() -> None:
    with pytest.raises(ValidationError, match="finite number"):
        RetrievalMetadata(
            modality_pair="text_to_text",
            dataset_name="test",
            sampling_weight=float("inf"),
        )


def test_health_report_detects_identical_vector_collapse() -> None:
    report = analyze_embedding_health(
        torch.ones(4, 3),
        thresholds=CollapseThresholds(minimum_effective_rank_ratio=0.5),
    )
    assert report.collapse_score > 0.9
    assert "effective_rank_below_threshold" in report.alerts
    assert "pairwise_cosine_above_threshold" in report.alerts
