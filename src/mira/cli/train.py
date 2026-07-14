"""Single-device Phase-1 training vertical slice."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
import torch
import typer

from mira.cli._common import emit_json, file_sha256
from mira.cli._media import resolve_record_media
from mira.cli._model import apply_configured_lora
from mira.config import (
    DistributedStrategy,
    MatryoshkaConfig,
    MultiPositiveInfoNCEConfig,
    SamplerType,
    TuningMethod,
    resolve_config,
)
from mira.core.contracts import (
    BatchMetadata,
    EmbeddingBatch,
    ObjectiveContext,
    RepresentationRequest,
)
from mira.core.exceptions import ConfigurationValidationError, ContractError
from mira.data import (
    DeterministicNoDuplicateBatchSampler,
    build_group_layout,
    read_jsonl,
)
from mira.data.fingerprint import records_checksum
from mira.integrations.ms_swift import SwiftLoadSpec, SwiftTemplateSpec, load_qwen3_vl_embedding
from mira.integrations.ms_swift.compatibility import EXPECTED_MS_SWIFT_VERSION
from mira.integrations.ms_swift.template_bridge import validate_qwen_message_roles
from mira.models.base import ParameterGroupSpec, TuningSpec
from mira.objectives import (
    MatryoshkaObjective,
    MultiPositiveInfoNCE,
    ObjectiveComposer,
    ObjectiveTerm,
)
from mira.training import (
    CheckpointComponents,
    CheckpointManager,
    CheckpointMetadata,
    CheckpointRestoreTargets,
    CheckpointSnapshot,
    TrainingState,
    TrainingStep,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mira.config import MiraConfig
    from mira.data.schema import EmbeddingRecord
    from mira.models.adapters import Qwen3VLEmbeddingAdapter
    from mira.objectives.base import EmbeddingObjective


def train_command(  # noqa: PLR0912, PLR0913, PLR0915 - explicit lifecycle controls
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    max_steps: Annotated[int, typer.Option("--max-steps", min=1)] = 1,
    overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
    resume_from: Annotated[
        Path | None,
        typer.Option("--resume-from", exists=True, file_okay=False, readable=True),
    ] = None,
    run_steps: Annotated[int | None, typer.Option("--run-steps", min=1)] = None,
    expected_manifest_sha256: Annotated[
        str | None,
        typer.Option("--expected-manifest-sha256"),
    ] = None,
) -> None:
    """Train a real Qwen adapter for bounded steps and save a full checkpoint."""
    resolved = resolve_config(config, overrides=tuple(overrides or ()))
    experiment = resolved.config
    _validate_cli_runtime(experiment)
    _seed_everything(experiment.project.seed)
    records = _load_training_records(experiment)
    validation_records = _load_validation_records(experiment)
    overlapping_ids = sorted(
        {record.sample_id for record in records}.intersection(
            record.sample_id for record in validation_records
        )
    )
    if overlapping_ids:
        raise ConfigurationValidationError(
            f"training and validation sample IDs overlap: {tuple(overlapping_ids)}"
        )
    media_root = Path(experiment.data.train[0].path).parent
    dataset_checksum = records_checksum(records, media_root=media_root)
    validation_media_root = Path(experiment.data.validation.path).parent
    validation_checksum = records_checksum(
        validation_records,
        media_root=validation_media_root,
    )
    sampler = DeterministicNoDuplicateBatchSampler(
        records,
        batch_size=experiment.runtime.train_batch_size_per_device,
        seed=experiment.project.seed,
        duplicate_keys=experiment.data.sampler.duplicate_keys,
        homogeneous_keys=(
            ("task_id", "dataset_name", "modality_pair")
            if experiment.data.sampler.homogeneous_global_batch
            else ()
        ),
        media_root=media_root,
    )
    maximum_steps = _configured_step_budget(sampler, experiment.runtime.epochs)
    if max_steps > maximum_steps:
        raise ConfigurationValidationError(
            f"max_steps={max_steps} exceeds the configured {experiment.runtime.epochs} "
            f"epoch budget of {maximum_steps} steps"
        )
    template_spec = SwiftTemplateSpec()
    assets = load_qwen3_vl_embedding(
        SwiftLoadSpec(
            model_id=experiment.model.name_or_path,
            revision=experiment.model.revision,
            torch_dtype=_precision_name(experiment.runtime.precision),
            attention=experiment.runtime.attention,
            image_max_tokens=experiment.data.media.image_max_tokens,
            embedding_dimension=experiment.model.hidden_size,
        ),
        template_spec,
    )
    apply_configured_lora(assets.adapter, experiment)
    _configure_gradient_checkpointing(assets.adapter, experiment)
    objective = _objective(experiment)
    groups = assets.adapter.get_parameter_groups(_tuning_spec(experiment))
    optimizer = _optimizer(
        groups,
        objective=objective,
        objective_learning_rate=_objective_learning_rate(experiment),
    )
    scheduler = _scheduler(optimizer, experiment, max_steps=max_steps)
    step = TrainingStep(
        assets.adapter,
        objective,
        optimizer,
        RepresentationRequest(normalize=experiment.model.representation.normalize),
        max_gradient_norm=1.0,
    )
    audit = _parameter_audit(assets.adapter, objective, groups)
    emit_json({"event": "parameter_audit", **audit})

    step_number = 0
    epoch = 0
    preserve_sampler_cursor = False
    if resume_from is not None:
        resume_manifest = CheckpointManager().verify(
            resume_from,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        _validate_resume_provenance(
            resume_manifest,
            config_checksum=resolved.checksum,
            dataset_checksum=dataset_checksum,
            validation_checksum=validation_checksum,
            model_name_or_path=experiment.model.name_or_path,
            planned_max_steps=max_steps,
        )
        loaded = CheckpointManager().load(
            resume_from,
            model=assets.adapter,
            mode="resume_full",
            targets=CheckpointRestoreTargets(
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                objective=objective,
            ),
            expected_manifest_sha256=expected_manifest_sha256,
        )
        if loaded.training_state is None:
            raise ContractError("full resume returned no training state")
        step_number = loaded.training_state.global_step
        epoch = loaded.training_state.epoch
        preserve_sampler_cursor = True
        if step_number >= max_steps:
            raise ConfigurationValidationError(
                f"resume checkpoint is already at step {step_number}, which is not below "
                f"max_steps={max_steps}"
            )
    elif expected_manifest_sha256 is not None:
        raise ConfigurationValidationError(
            "expected_manifest_sha256 is only valid together with resume_from"
        )
    stop_step = max_steps if run_steps is None else min(max_steps, step_number + run_steps)
    last_result = None
    while step_number < stop_step:
        if epoch >= math.ceil(experiment.runtime.epochs):
            raise ConfigurationValidationError(
                "training state would exceed the configured epoch budget before reaching "
                f"stop_step={stop_step}"
            )
        if not preserve_sampler_cursor:
            sampler.set_epoch(epoch)
        preserve_sampler_cursor = False
        epoch_batch_limit = len(sampler.global_batches())
        full_epochs = math.floor(experiment.runtime.epochs)
        fractional_epoch = experiment.runtime.epochs - full_epochs
        if fractional_epoch and epoch == full_epochs:
            epoch_batch_limit = math.ceil(epoch_batch_limit * fractional_epoch)
        if sampler.cursor > epoch_batch_limit:
            raise ConfigurationValidationError(
                "resume sampler cursor exceeds the configured fractional-epoch budget"
            )
        for indices in sampler:
            if sampler.cursor > epoch_batch_limit:
                break
            selected = tuple(records[index] for index in indices)
            batch = _collate_batch(
                assets.adapter,
                assets.template,
                selected,
                template_spec,
                _retrieval_config(experiment),
            )
            last_result = step.run(
                batch,
                ObjectiveContext(global_step=step_number, epoch=float(epoch)),
            )
            scheduler.step()
            step_number += 1
            if step_number == stop_step:
                break
        if step_number == stop_step:
            break
        epoch += 1
    if last_result is None:
        raise ContractError("training sampler produced no batches")

    checkpoint = output_dir / f"checkpoint-step-{step_number}"
    manifest = CheckpointManager().save(
        checkpoint,
        snapshot=CheckpointSnapshot(
            model=assets.adapter,
            training_state=TrainingState(global_step=step_number, epoch=sampler.epoch),
            resolved_config=experiment.model_dump(mode="json"),
            components=CheckpointComponents(
                optimizer=optimizer,
                scheduler=scheduler,
                sampler_state=sampler.state_dict(),
                objective_state=objective.state_dict(),
            ),
            metadata=CheckpointMetadata(
                manifests={
                    "config_checksum": resolved.checksum,
                    "datasets": {source.name: dataset_checksum for source in experiment.data.train},
                    "dataset_licenses": {
                        source.name: source.licenses for source in experiment.data.train
                    },
                    "model_name_or_path": experiment.model.name_or_path,
                    "model_revision": experiment.model.revision,
                    "model_licenses": experiment.model.licenses,
                    "planned_max_steps": max_steps,
                    "template_fingerprint": assets.template.fingerprint,
                    "validation_dataset": {
                        "checksum": validation_checksum,
                        "evaluation_status": "NOT RUN",
                        "licenses": experiment.data.validation.licenses,
                        "name": experiment.data.validation.name,
                        "path": experiment.data.validation.path,
                        "task_id": experiment.data.validation.task_id,
                    },
                },
                source_revisions={"ms-swift": EXPECTED_MS_SWIFT_VERSION},
            ),
        ),
    )
    manifest_sha256 = file_sha256(checkpoint / "manifest.json")
    emit_json(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_version": manifest["checkpoint_version"],
            "config_checksum": resolved.checksum,
            "gradient_norm": last_result.gradient_norm,
            "loss": last_result.loss,
            "manifest_sha256": manifest_sha256,
            "status": "complete",
            "steps": step_number,
        }
    )


def _validate_cli_runtime(config: MiraConfig) -> None:
    if config.runtime.precision == "fp16":
        raise ConfigurationValidationError(
            "the Phase-1 CLI does not implement FP16 loss scaling; use bf16 or fp32"
        )
    if config.runtime.distributed.strategy is not DistributedStrategy.SINGLE:
        raise ConfigurationValidationError(
            "the `mira train` vertical slice currently accepts runtime.distributed.strategy="
            "single; use the tested distributed primitives from a custom launcher"
        )
    if config.runtime.gradient_accumulation_steps != 1:
        raise ConfigurationValidationError(
            "the CLI vertical slice requires gradient_accumulation_steps=1"
        )
    if config.runtime.padding_free:
        raise ConfigurationValidationError(
            "the MIRA item collator does not yet expose padding_free training"
        )
    if not config.data.media.validate_on_startup:
        raise ConfigurationValidationError(
            "the Phase-1 CLI requires data.media.validate_on_startup=true"
        )
    if config.data.sampler.type is SamplerType.MODALITY_BALANCED_NO_DUPLICATE:
        raise ConfigurationValidationError(
            "the Phase-1 CLI does not implement weighted modality-balanced sampling"
        )
    if len(config.data.train) != 1:
        raise ConfigurationValidationError(
            "the Phase-1 CLI requires exactly one training source; weighted mixtures need "
            "a mixture-aware sampler"
        )
    source = config.data.train[0]
    if source.weight != 1.0 or source.negative_semantics is not None:
        raise ConfigurationValidationError(
            "the Phase-1 CLI requires one source with weight=1 and no custom negative_semantics"
        )
    if config.model.tuning.method is TuningMethod.QLORA:
        raise ConfigurationValidationError(
            "QLoRA requires an explicit quantized load contract that the Phase-1 CLI "
            "does not expose"
        )
    retrieval = _retrieval_config(config)
    if retrieval.false_negative_mask.similarity_margin is not None:
        raise ConfigurationValidationError(
            "similarity-margin false-negative masking is not implemented in the Phase-1 CLI"
        )


def _configured_step_budget(
    sampler: DeterministicNoDuplicateBatchSampler,
    epochs: float,
) -> int:
    """Count the exact deterministic batches allowed by full and fractional epochs."""
    original_state = sampler.state_dict()
    full_epochs = math.floor(epochs)
    fractional_epoch = epochs - full_epochs
    budget = 0
    try:
        for epoch in range(full_epochs):
            sampler.set_epoch(epoch)
            budget += len(sampler.global_batches())
        if fractional_epoch:
            sampler.set_epoch(full_epochs)
            budget += math.ceil(len(sampler.global_batches()) * fractional_epoch)
    finally:
        sampler.load_state_dict(original_state)
    return budget


def _load_training_records(config: MiraConfig) -> tuple[EmbeddingRecord, ...]:
    records: list[EmbeddingRecord] = []
    for source in config.data.train:
        dataset = read_jsonl(source.path)
        mismatched = tuple(
            record.sample_id for record in dataset if record.task_id != source.task_id
        )
        if mismatched:
            raise ConfigurationValidationError(
                f"dataset {source.name!r} has records with unexpected task_id: {mismatched}"
            )
        mislabeled = tuple(
            record.sample_id for record in dataset if record.metadata.dataset_name != source.name
        )
        if mislabeled:
            raise ConfigurationValidationError(
                f"dataset {source.name!r} has records with a different metadata.dataset_name: "
                f"{mislabeled}"
            )
        if config.data.media.validate_on_startup:
            dataset = tuple(
                resolve_record_media(record, Path(source.path).parent) for record in dataset
            )
        records.extend(dataset)
    if not records:
        raise ConfigurationValidationError("training datasets contain no records")
    _validate_qwen_records(records)
    weighted_records = tuple(
        record.sample_id for record in records if record.metadata.sampling_weight != 1.0
    )
    if weighted_records:
        raise ConfigurationValidationError(
            "record-level sampling_weight is not implemented in the Phase-1 sampler; "
            f"offending samples: {weighted_records}"
        )
    hard_negative_count = _retrieval_config(config).negatives.explicit_hard_negatives
    mismatched_negative_counts = tuple(
        (record.sample_id, len(record.negatives))
        for record in records
        if len(record.negatives) != hard_negative_count
    )
    if mismatched_negative_counts:
        raise ConfigurationValidationError(
            "record explicit-negative counts must exactly match "
            f"objectives.retrieval.negatives.explicit_hard_negatives={hard_negative_count}; "
            f"observed {mismatched_negative_counts}"
        )
    return tuple(records)


def _load_validation_records(config: MiraConfig) -> tuple[EmbeddingRecord, ...]:
    source = config.data.validation
    records = read_jsonl(source.path)
    if not records:
        raise ConfigurationValidationError("validation dataset contains no records")
    if source.task_id is not None:
        mismatched = tuple(
            record.sample_id for record in records if record.task_id != source.task_id
        )
        if mismatched:
            raise ConfigurationValidationError(
                f"validation dataset has records with unexpected task_id: {mismatched}"
            )
    if source.name is not None:
        mislabeled = tuple(
            record.sample_id for record in records if record.metadata.dataset_name != source.name
        )
        if mislabeled:
            raise ConfigurationValidationError(
                f"validation records have a different metadata.dataset_name: {mislabeled}"
            )
    if config.data.media.validate_on_startup:
        records = tuple(
            resolve_record_media(record, Path(source.path).parent) for record in records
        )
    _validate_qwen_records(records)
    return records


def _validate_qwen_records(records: Sequence[EmbeddingRecord]) -> None:
    unsupported_audio = tuple(
        item.item_id
        for record in records
        for item in (record.query, *record.positives, *record.negatives)
        if item.audios
    )
    if unsupported_audio:
        raise ConfigurationValidationError(
            "Qwen3-VL-Embedding does not declare audio support; offending items: "
            f"{unsupported_audio}"
        )
    unsupported_instructions = tuple(
        item.item_id
        for record in records
        for item in (record.query, *record.positives, *record.negatives)
        if item.instruction not in {None, SwiftTemplateSpec().instruction}
    )
    if unsupported_instructions:
        raise ConfigurationValidationError(
            "training items must use the configured Qwen instruction; offending items: "
            f"{unsupported_instructions}"
        )
    for record in records:
        for item in (record.query, *record.positives, *record.negatives):
            validate_qwen_message_roles(item)


def _validate_resume_provenance(  # noqa: PLR0913 - every identity axis is explicit
    manifest: dict[str, Any],
    *,
    config_checksum: str,
    dataset_checksum: str,
    validation_checksum: str,
    model_name_or_path: str,
    planned_max_steps: int,
) -> None:
    values = manifest.get("manifests")
    if not isinstance(values, dict):
        raise ConfigurationValidationError("resume checkpoint has no provenance manifest")
    expected = {
        "config_checksum": config_checksum,
        "model_name_or_path": model_name_or_path,
        "planned_max_steps": planned_max_steps,
    }
    for name, value in expected.items():
        if values.get(name) != value:
            raise ConfigurationValidationError(
                f"resume checkpoint {name} mismatch: checkpoint={values.get(name)!r}, "
                f"current={value!r}"
            )
    datasets = values.get("datasets")
    if not isinstance(datasets, dict) or set(datasets.values()) != {dataset_checksum}:
        raise ConfigurationValidationError("resume checkpoint dataset checksum mismatch")
    validation = values.get("validation_dataset")
    if not isinstance(validation, dict) or validation.get("checksum") != validation_checksum:
        raise ConfigurationValidationError("resume checkpoint validation dataset checksum mismatch")


def _configure_gradient_checkpointing(
    adapter: Qwen3VLEmbeddingAdapter,
    config: MiraConfig,
) -> None:
    if not config.runtime.gradient_checkpointing:
        return
    enable = getattr(adapter.model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise ConfigurationValidationError(
            "gradient_checkpointing=true, but the loaded model exposes no enable method"
        )
    enable()


def _tuning_spec(config: MiraConfig) -> TuningSpec:
    values: dict[str, float] = {}
    for name, component in config.model.tuning.components:
        if component.trainable:
            if component.learning_rate is None:
                raise ConfigurationValidationError(
                    f"trainable component {name!r} requires an explicit learning_rate"
                )
            values[name] = component.learning_rate
    return TuningSpec(values, weight_decay=config.runtime.optimizer.weight_decay)


def _optimizer(
    groups: Sequence[ParameterGroupSpec],
    *,
    objective: EmbeddingObjective | None = None,
    objective_learning_rate: float | None = None,
) -> torch.optim.AdamW:
    values = [
        {
            "lr": group.learning_rate,
            "params": list(group.parameters),
            "weight_decay": group.weight_decay,
        }
        for group in groups
    ]
    objective_parameters = (
        []
        if objective is None
        else [parameter for parameter in objective.parameters() if parameter.requires_grad]
    )
    if objective_parameters:
        if objective_learning_rate is None:
            raise ConfigurationValidationError(
                "trainable objective parameters require an explicit learning rate"
            )
        values.append(
            {
                "lr": objective_learning_rate,
                "params": objective_parameters,
                "weight_decay": 0.0,
            }
        )
    if not values:
        raise ConfigurationValidationError("configuration produced no optimizer groups")
    return torch.optim.AdamW(values)


def _objective(config: MiraConfig) -> EmbeddingObjective:
    retrieval = _retrieval_config(config)
    retrieval_objective = _infonce(retrieval)
    terms: list[ObjectiveTerm] = [
        ObjectiveTerm("retrieval", retrieval_objective, weight=retrieval.weight)
    ]
    mrl = next(
        (item for item in config.objectives if isinstance(item, MatryoshkaConfig)),
        None,
    )
    if mrl is not None:
        terms.append(
            ObjectiveTerm(
                "matryoshka",
                MatryoshkaObjective(retrieval_objective, mrl.dimensions),
                weight=mrl.weight,
            )
        )
    return ObjectiveComposer(terms)


def _retrieval_config(config: MiraConfig) -> MultiPositiveInfoNCEConfig:
    return next(item for item in config.objectives if isinstance(item, MultiPositiveInfoNCEConfig))


def _objective_learning_rate(config: MiraConfig) -> float | None:
    return _retrieval_config(config).temperature.learning_rate


def _scheduler(
    optimizer: torch.optim.Optimizer,
    config: MiraConfig,
    *,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = math.floor(config.runtime.scheduler.warmup_ratio * max_steps)

    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _infonce(config: MultiPositiveInfoNCEConfig) -> MultiPositiveInfoNCE:
    temperature = config.temperature
    return MultiPositiveInfoNCE(
        temperature=temperature.value,
        learnable_temperature=temperature.type == "learnable",
        min_temperature=temperature.minimum or temperature.value,
        max_temperature=temperature.maximum or temperature.value,
    )


def _collate_batch(
    adapter: Qwen3VLEmbeddingAdapter,
    template: Any,
    records: Sequence[EmbeddingRecord],
    template_spec: SwiftTemplateSpec,
    retrieval: MultiPositiveInfoNCEConfig,
) -> EmbeddingBatch:
    built = build_group_layout(
        records,
        include_in_batch_negatives=retrieval.negatives.in_batch,
        apply_metadata_false_negative_mask=(
            retrieval.false_negative_mask.enabled and retrieval.false_negative_mask.metadata_rules
        ),
    )
    encoded = [template.encode_item(item) for item in built.flattened_items]
    collated = template.collate_items(encoded)
    non_tensors = tuple(
        name for name, value in collated.items() if not isinstance(value, torch.Tensor)
    )
    if non_tensors:
        raise ContractError(f"Swift collator returned unsupported non-tensor fields: {non_tensors}")
    device = _model_device(adapter.model)
    model_inputs = {name: value.to(device) for name, value in collated.items()}
    layout = built.layout.to(device)
    return EmbeddingBatch(
        model_inputs=model_inputs,
        layout=layout,
        metadata=BatchMetadata(
            sample_ids=tuple(record.sample_id for record in records),
            task_ids=layout.task_ids,
            dataset_ids=layout.dataset_ids,
            extras={
                "instruction": template_spec.instruction,
                "template_fingerprint": template.fingerprint,
            },
        ),
    )


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ContractError("loaded model contains no parameters") from error


def _parameter_audit(
    adapter: Qwen3VLEmbeddingAdapter,
    objective: EmbeddingObjective,
    groups: Sequence[ParameterGroupSpec],
) -> dict[str, object]:
    total = sum(parameter.numel() for parameter in adapter.parameters())
    trainable = sum(
        parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
    )
    objective_parameters = sum(
        parameter.numel() for parameter in objective.parameters() if parameter.requires_grad
    )
    return {
        "groups": {
            group.name: {
                "learning_rate": group.learning_rate,
                "parameters": sum(parameter.numel() for parameter in group.parameters),
            }
            for group in groups
        },
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_objective_parameters": objective_parameters,
        "trainable_percentage": 100.0 * trainable / total,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _precision_name(value: str) -> Literal["bfloat16", "float16", "float32"]:
    if value == "bf16":
        return "bfloat16"
    if value == "fp16":
        return "float16"
    return "float32"
