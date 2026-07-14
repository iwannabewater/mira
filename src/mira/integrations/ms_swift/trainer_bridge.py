"""Minimal constructor bridge to MS-SWIFT's Trainer implementation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

from mira.core.exceptions import ContractError

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn


@dataclass(slots=True)
class SwiftTrainerSpec:
    """Objects forwarded unchanged into the fixed upstream Trainer."""

    model: nn.Module
    arguments: Any
    train_dataset: Any
    data_collator: Callable[[list[Any]], dict[str, Any]]
    compute_loss: Callable[..., Any]
    eval_dataset: Any | None = None


def create_swift_trainer(spec: SwiftTrainerSpec) -> Any:
    """Create an upstream Trainer while retaining MIRA's loss and collator."""
    trainers = import_module("swift.trainers")
    trainer_class = getattr(trainers, "Trainer", None)
    if trainer_class is None:
        raise ContractError("MS-SWIFT does not expose swift.trainers.Trainer")
    trainer = trainer_class(
        model=spec.model,
        args=spec.arguments,
        train_dataset=spec.train_dataset,
        eval_dataset=spec.eval_dataset,
        data_collator=spec.data_collator,
        compute_loss_func=spec.compute_loss,
    )
    if not hasattr(trainer, "train"):
        raise ContractError("MS-SWIFT returned an incompatible Trainer object")
    return trainer
