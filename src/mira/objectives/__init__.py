"""Mathematically validated embedding objectives and composition."""

from mira.objectives.base import EmbeddingObjective
from mira.objectives.composer import ObjectiveComposer, ObjectiveTerm
from mira.objectives.contrastive import MultiPositiveInfoNCE, SymmetricMultiPositiveInfoNCE
from mira.objectives.matryoshka import MatryoshkaLoss, MatryoshkaObjective, MRLObjective
from mira.objectives.scheduling import (
    ConstantSchedule,
    CosineSchedule,
    EpochSchedule,
    LinearWarmupSchedule,
    PerTaskSchedule,
    StepSchedule,
    WeightSchedule,
)

__all__ = [
    "ConstantSchedule",
    "CosineSchedule",
    "EmbeddingObjective",
    "EpochSchedule",
    "LinearWarmupSchedule",
    "MRLObjective",
    "MatryoshkaLoss",
    "MatryoshkaObjective",
    "MultiPositiveInfoNCE",
    "ObjectiveComposer",
    "ObjectiveTerm",
    "PerTaskSchedule",
    "StepSchedule",
    "SymmetricMultiPositiveInfoNCE",
    "WeightSchedule",
]
