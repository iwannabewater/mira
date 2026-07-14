from __future__ import annotations

import pytest
import torch

from mira.core import ConfigurationError, NumericalError
from mira.evaluation import (
    bootstrap_confidence_interval,
    build_slice_masks,
    paired_bootstrap,
    slice_metrics,
)


def test_bootstrap_is_seeded_and_paired() -> None:
    baseline = torch.tensor([0.1, 0.2, 0.3, 0.4])
    candidate = baseline + 0.1

    first = bootstrap_confidence_interval(candidate, n_resamples=500, seed=7)
    second = bootstrap_confidence_interval(candidate, n_resamples=500, seed=7)
    paired = paired_bootstrap(baseline, candidate, n_resamples=500, seed=7)

    assert first == second
    assert first.low <= first.estimate <= first.high
    assert paired.delta == pytest.approx(0.1)
    assert paired.low == pytest.approx(0.1)
    assert paired.high == pytest.approx(0.1)
    assert 0 < paired.p_value <= 1


def test_slice_masks_and_aggregation() -> None:
    masks = build_slice_masks(
        {
            "language": ("en", "zh", "en"),
            "modality": ("text", "image", "text"),
        }
    )
    sliced = slice_metrics(
        {
            "recall@1": torch.tensor([1.0, 0.0, 0.5]),
            "average_precision": torch.tensor([1.0, 0.25, 0.5]),
            "first_relevant_rank": torch.tensor([1.0, 4.0, 2.0]),
        },
        {"english": masks["language=en"]},
    )

    assert sliced["english"]["recall@1"] == pytest.approx(0.75)
    assert sliced["english"]["map"] == pytest.approx(0.75)
    assert sliced["english"]["median_rank"] == pytest.approx(1.5)


def test_statistics_validate_alignment_and_options() -> None:
    with pytest.raises(ConfigurationError, match="identical"):
        paired_bootstrap(torch.ones(2), torch.ones(3), n_resamples=10)
    with pytest.raises(ConfigurationError, match="at least two"):
        bootstrap_confidence_interval(torch.ones(1), n_resamples=10)
    with pytest.raises(NumericalError, match="NaN"):
        bootstrap_confidence_interval(torch.tensor([1.0, float("nan")]), n_resamples=10)
    with pytest.raises(ConfigurationError, match="at least 2"):
        bootstrap_confidence_interval(torch.ones(2), n_resamples=1)
    with pytest.raises(ConfigurationError, match="between"):
        bootstrap_confidence_interval(torch.ones(2), confidence=1.0, n_resamples=2)


def test_slice_validation_is_explicit() -> None:
    with pytest.raises(ConfigurationError, match="must not be empty"):
        build_slice_masks({})
    with pytest.raises(ConfigurationError, match="non-zero"):
        build_slice_masks({"a": (1,), "b": (1, 2)})
    with pytest.raises(ConfigurationError, match="must not be empty"):
        slice_metrics({}, {"all": torch.ones(1, dtype=torch.bool)})
    with pytest.raises(NumericalError, match="rank-1"):
        slice_metrics(
            {"recall": torch.tensor([[1.0, 0.0]])},
            {"all": torch.ones(2, dtype=torch.bool)},
        )
    with pytest.raises(ConfigurationError, match="selects no"):
        slice_metrics(
            {"recall": torch.ones(2)},
            {"empty": torch.zeros(2, dtype=torch.bool)},
        )
    with pytest.raises(ConfigurationError, match="equal length"):
        slice_metrics(
            {"recall": torch.ones(2), "mrr": torch.ones(3)},
            {"all": torch.ones(2, dtype=torch.bool)},
        )
    with pytest.raises(ConfigurationError, match="boolean vector"):
        slice_metrics(
            {"recall": torch.ones(2)},
            {"bad": torch.ones(2)},
        )
