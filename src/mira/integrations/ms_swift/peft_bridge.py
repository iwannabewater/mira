"""Backward-compatible PEFT imports for the MS-SWIFT integration."""

from __future__ import annotations

from mira.integrations.peft import LoraSpec, apply_lora

__all__ = ["LoraSpec", "apply_lora"]
