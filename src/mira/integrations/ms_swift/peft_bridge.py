"""Lazy PEFT construction for Qwen component-aware LoRA training."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from torch import nn

from mira.core.exceptions import ConfigurationValidationError, ContractError


@dataclass(frozen=True, slots=True)
class LoraSpec:
    """Auditable LoRA values passed to PEFT without hidden defaults."""

    rank: int
    alpha: float
    dropout: float
    target_modules: str | tuple[str, ...]
    use_dora: bool = False
    use_rslora: bool = False

    def __post_init__(self) -> None:
        """Validate LoRA topology before modifying model parameters."""
        if self.rank <= 0 or self.alpha <= 0:
            raise ConfigurationValidationError("LoRA rank and alpha must be positive")
        if self.dropout < 0 or self.dropout >= 1:
            raise ConfigurationValidationError("LoRA dropout must satisfy 0 <= p < 1")
        if isinstance(self.target_modules, str):
            if self.target_modules != "all-linear":
                raise ConfigurationValidationError(
                    "string LoRA target_modules must be the explicit 'all-linear' selector"
                )
        elif not self.target_modules or any(not value.strip() for value in self.target_modules):
            raise ConfigurationValidationError("LoRA target_modules must be non-empty")


def apply_lora(model: nn.Module, spec: LoraSpec) -> nn.Module:
    """Apply PEFT LoRA and fail if none of the declared modules can be matched."""
    if not isinstance(spec.target_modules, str):
        available_suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        missing = tuple(
            target for target in spec.target_modules if target not in available_suffixes
        )
        if missing:
            raise ContractError(f"LoRA target modules are absent from the model: {missing}")
    peft = import_module("peft")
    config_type = getattr(peft, "LoraConfig", None)
    get_peft_model = getattr(peft, "get_peft_model", None)
    if config_type is None or not callable(get_peft_model):
        raise ContractError("installed PEFT does not expose the LoRA construction API")
    config = config_type(
        r=spec.rank,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=(
            spec.target_modules
            if isinstance(spec.target_modules, str)
            else list(spec.target_modules)
        ),
        use_dora=spec.use_dora,
        use_rslora=spec.use_rslora,
        task_type="FEATURE_EXTRACTION",
    )
    wrapped = get_peft_model(model, config)
    if not isinstance(wrapped, nn.Module):
        raise ContractError("PEFT returned an invalid model wrapper")
    return wrapped
