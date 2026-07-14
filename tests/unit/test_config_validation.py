from __future__ import annotations

import math
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from mira.config import (
    config_checksum,
    interpolate_environment,
    load_config,
    validate_config_mapping,
)
from mira.core import ConfigurationValidationError

FIXTURE = Path(__file__).parents[1] / "fixtures" / "configs" / "qwen3_vl_ecommerce.yaml"


def _mapping() -> dict[str, Any]:
    return load_config(FIXTURE).model_dump(mode="json")


def _invalid(document: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationValidationError):
        validate_config_mapping(document)


def _set_path(document: dict[str, Any], path: tuple[str | int, ...], value: Any) -> None:
    current: Any = document
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def test_extra_fields_are_forbidden_at_every_level() -> None:
    document = _mapping()
    document["model"]["representation"]["mystery"] = True
    _invalid(document)


def test_mrl_dimensions_must_be_ordered_positive_weighted_and_within_hidden_size() -> None:
    reversed_dimensions = _mapping()
    reversed_dimensions["objectives"][1]["dimensions"] = {1024: 0.5, 512: 0.5}
    _invalid(reversed_dimensions)

    zero_weights = _mapping()
    zero_weights["objectives"][1]["dimensions"] = {512: 0.0, 1024: 0.0}
    _invalid(zero_weights)

    too_large = _mapping()
    too_large["model"]["hidden_size"] = 2048
    _invalid(too_large)


def test_temperature_bounds_fail_during_configuration_loading() -> None:
    document = _mapping()
    document["objectives"][0]["temperature"]["value"] = 0
    _invalid(document)

    learnable = _mapping()
    learnable["objectives"][0]["temperature"] = {
        "type": "learnable",
        "value": 0.07,
        "minimum": 0.1,
        "maximum": 0.5,
    }
    _invalid(learnable)


def test_model_capabilities_are_checked_before_training() -> None:
    document = _mapping()
    document["model"]["representation"]["type"] = "multi_vector"
    _invalid(document)


@pytest.mark.parametrize("method", ["qlora", "adapter_only", "projection_only"])
def test_planned_tuning_methods_fail_during_config_preflight(method: str) -> None:
    document = _mapping()
    document["model"]["tuning"]["method"] = method
    if method != "qlora":
        document["model"]["tuning"].pop("lora")
    _invalid(document)


def test_component_learning_rate_must_match_trainability() -> None:
    frozen_with_rate = _mapping()
    frozen_with_rate["model"]["tuning"]["components"]["vision"]["learning_rate"] = 1e-4
    _invalid(frozen_with_rate)

    trainable_without_rate = _mapping()
    trainable_without_rate["model"]["tuning"]["components"]["aligner"].pop("learning_rate")
    _invalid(trainable_without_rate)


def test_qwen_vision_or_aligner_lora_requires_all_linear_targets() -> None:
    aligner_targets_miss_merger = _mapping()
    aligner_targets_miss_merger["model"]["tuning"]["lora"]["target_modules"] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ]
    _invalid(aligner_targets_miss_merger)

    language_only = _mapping()
    language_only["model"]["tuning"]["components"]["aligner"] = {"trainable": False}
    language_only["model"]["tuning"]["lora"]["target_modules"] = ["q_proj", "v_proj"]
    validate_config_mapping(language_only)


def test_sampler_and_in_batch_objective_must_share_negative_semantics() -> None:
    document = _mapping()
    document["data"]["sampler"]["homogeneous_global_batch"] = False
    _invalid(document)


def test_cross_device_negatives_require_distributed_runtime() -> None:
    document = _mapping()
    document["runtime"]["distributed"]["strategy"] = "single"
    _invalid(document)


def test_dataset_weights_and_objective_names_are_validated() -> None:
    weights = _mapping()
    weights["data"]["train"][0]["weight"] = 0.5
    _invalid(weights)

    names = _mapping()
    names["objectives"][1]["name"] = names["objectives"][0]["name"]
    _invalid(names)

    missing_license = _mapping()
    missing_license["data"]["train"][0].pop("licenses")
    _invalid(missing_license)

    missing_validation_license = _mapping()
    missing_validation_license["data"]["validation"].pop("licenses")
    _invalid(missing_validation_license)


def test_evaluation_dimensions_must_be_trained() -> None:
    document = _mapping()
    document["evaluation"]["dimensions"].append(8192)
    _invalid(document)


def test_boolean_values_cannot_coerce_into_numeric_or_boolean_config_fields() -> None:
    objective_weight = _mapping()
    objective_weight["objectives"][0]["weight"] = True
    _invalid(objective_weight)

    epochs = _mapping()
    epochs["runtime"]["epochs"] = True
    _invalid(epochs)

    seed = _mapping()
    seed["project"]["seed"] = 1.0
    _invalid(seed)

    trainable = _mapping()
    trainable["model"]["tuning"]["components"]["aligner"]["trainable"] = 1
    _invalid(trainable)

    mrl_key = _mapping()
    mrl_key["objectives"][1]["dimensions"] = {True: 1.0}
    _invalid(mrl_key)

    mrl_weight = _mapping()
    mrl_weight["objectives"][1]["dimensions"] = {512: True, 1024: 1.0}
    _invalid(mrl_weight)

    evaluation_dimension = _mapping()
    evaluation_dimension["evaluation"]["dimensions"] = [True, 1024, 2048, 4096]
    _invalid(evaluation_dimension)


def test_valid_mapping_is_not_mutated_by_validation() -> None:
    document = _mapping()
    original = deepcopy(document)
    validate_config_mapping(document)
    assert document == original


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("project", "seed"), "42"),
        (("runtime", "epochs"), "1.5"),
        (("runtime", "gradient_accumulation_steps"), b"1"),
        (("project", "name"), b"experiment"),
        (("backend", "type"), b"ms_swift"),
        (("model", "licenses"), [b"Apache-2.0"]),
        (("data", "train", 0, "weight"), "0.6"),
        (("objectives", 0, "temperature", "value"), "0.07"),
    ],
)
def test_scalar_fields_reject_string_and_bytes_coercion(
    path: tuple[str | int, ...], value: object
) -> None:
    document = _mapping()
    _set_path(document, path, value)
    _invalid(document)


@pytest.mark.parametrize("value", [Decimal("0.5"), Fraction(1, 2)])
def test_float_fields_reject_non_builtin_numeric_coercion(value: object) -> None:
    document = _mapping()
    document["objectives"][0]["temperature"]["value"] = value
    _invalid(document)


@pytest.mark.parametrize(
    "path",
    [
        ("project", "name"),
        ("data", "train", 0, "path"),
        ("objectives", 0, "name"),
        ("evaluation", "slices", 0),
    ],
)
def test_configuration_strings_reject_edge_whitespace(path: tuple[str | int, ...]) -> None:
    document = _mapping()
    if path == ("evaluation", "slices", 0):
        document["evaluation"]["slices"] = ["slice"]
    current: Any = document
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = f" {current[path[-1]]} "
    _invalid(document)


def test_cyclic_direct_mappings_fail_as_configuration_errors() -> None:
    document = _mapping()
    cycle: dict[str, Any] = {}
    cycle["self"] = cycle
    document["unexpected_cycle"] = cycle
    _invalid(document)


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_all_nonfinite_numeric_inputs_fail_as_configuration_errors(nonfinite: float) -> None:
    for path in (
        ("runtime", "epochs"),
        ("data", "train", 0, "weight"),
        ("objectives", 0, "temperature", "value"),
        ("objectives", 1, "dimensions", 512),
    ):
        document = _mapping()
        _set_path(document, path, nonfinite)
        _invalid(document)


def test_extreme_dataset_weights_are_validation_errors_not_math_overflow() -> None:
    document = _mapping()
    for dataset in document["data"]["train"]:
        dataset["weight"] = 1e308
    _invalid(document)


@pytest.mark.parametrize(
    "dimensions",
    [
        [],
        {True: 1.0},
        {512.0: 1.0},
        {"0512": 1.0},
        {512: "1.0"},
        {512: 0.5, "512": 0.5},
    ],
)
def test_matryoshka_dimension_input_shape_is_exact_and_errors_are_wrapped(
    dimensions: object,
) -> None:
    document = _mapping()
    document["objectives"][1]["dimensions"] = dimensions
    with pytest.raises(ConfigurationValidationError):
        validate_config_mapping(document)


def test_json_dimension_keys_are_canonicalized_explicitly_then_frozen() -> None:
    config = validate_config_mapping(_mapping())
    mrl = config.objectives[1]
    assert mrl.type == "matryoshka"
    assert tuple(mrl.dimensions) == (512, 1024, 2048, 4096)
    checksum = config_checksum(config)

    with pytest.raises(TypeError, match="immutable"):
        mrl.dimensions[512] = 999
    with pytest.raises(TypeError, match="immutable"):
        mrl.dimensions.update({512: 999})
    assert config_checksum(config) == checksum


def test_validation_errors_hide_interpolated_environment_values() -> None:
    document = _mapping()
    document["project"]["seed"] = "${SECRET_VALUE}"
    environment_value = "must-not-appear-in-validation-error"
    interpolated = interpolate_environment(document, environ={"SECRET_VALUE": environment_value})

    with pytest.raises(ConfigurationValidationError) as captured:
        validate_config_mapping(interpolated)
    assert environment_value not in str(captured.value)


def test_tuning_payload_and_component_preflight_covers_each_invalid_shape() -> None:
    missing_lora = _mapping()
    missing_lora["model"]["tuning"].pop("lora")
    _invalid(missing_lora)

    full_with_lora = _mapping()
    full_with_lora["model"]["tuning"]["method"] = "full"
    _invalid(full_with_lora)

    none_trainable = _mapping()
    for component in none_trainable["model"]["tuning"]["components"].values():
        component["trainable"] = False
        component.pop("learning_rate", None)
    _invalid(none_trainable)

    empty_target = _mapping()
    empty_target["model"]["tuning"]["lora"]["target_modules"] = ""
    _invalid(empty_target)

    empty_tuple_target = _mapping()
    empty_tuple_target["model"]["tuning"]["lora"]["target_modules"] = ["q_proj", ""]
    _invalid(empty_tuple_target)

    planned_component = _mapping()
    planned_component["model"]["tuning"]["components"]["identity_head"] = {
        "trainable": True,
        "learning_rate": 1e-5,
    }
    _invalid(planned_component)


def test_model_dataset_and_sampler_identifiers_are_unique_and_nonempty() -> None:
    model_licenses = _mapping()
    model_licenses["model"]["licenses"] = ["Apache-2.0", "Apache-2.0"]
    _invalid(model_licenses)

    dataset_licenses = _mapping()
    dataset_licenses["data"]["train"][0]["licenses"] = [""]
    _invalid(dataset_licenses)

    validation_licenses = _mapping()
    validation_licenses["data"]["validation"]["licenses"] = ["dataset", "dataset"]
    _invalid(validation_licenses)

    duplicate_datasets = _mapping()
    duplicate_datasets["data"]["train"][1]["name"] = duplicate_datasets["data"]["train"][0]["name"]
    _invalid(duplicate_datasets)

    duplicate_keys = _mapping()
    duplicate_keys["data"]["sampler"]["duplicate_keys"] = ["id", "id"]
    _invalid(duplicate_keys)

    empty_key = _mapping()
    empty_key["data"]["sampler"]["duplicate_keys"] = [""]
    _invalid(empty_key)

    missing_rules = _mapping()
    missing_rules["data"]["sampler"]["duplicate_keys"] = []
    _invalid(missing_rules)


def test_model_identity_and_qwen_license_are_exact() -> None:
    unknown = _mapping()
    unknown["model"]["name_or_path"] = "Qwen/unknown"
    _invalid(unknown)

    wrong_revision = _mapping()
    wrong_revision["model"]["revision"] = "0" * 40
    _invalid(wrong_revision)

    wrong_license = _mapping()
    wrong_license["model"]["licenses"] = ["MIT"]
    _invalid(wrong_license)


def test_temperature_negative_and_false_negative_modes_are_exact() -> None:
    fixed_bounds = _mapping()
    fixed_bounds["objectives"][0]["temperature"]["minimum"] = 0.01
    _invalid(fixed_bounds)

    reversed_bounds = _mapping()
    reversed_bounds["objectives"][0]["temperature"] = {
        "type": "learnable",
        "value": 0.2,
        "minimum": 0.5,
        "maximum": 0.1,
        "learning_rate": 1e-5,
    }
    _invalid(reversed_bounds)

    outside_bounds = _mapping()
    outside_bounds["objectives"][0]["temperature"] = {
        "type": "learnable",
        "value": 0.05,
        "minimum": 0.1,
        "maximum": 0.5,
        "learning_rate": 1e-5,
    }
    _invalid(outside_bounds)

    cross_device_none = _mapping()
    cross_device_none["objectives"][0]["negatives"]["cross_device_gradient"] = "none"
    _invalid(cross_device_none)

    local_remote_gradient = _mapping()
    local_remote_gradient["objectives"][0]["negatives"].update(
        {"cross_device": False, "cross_device_gradient": "detached_remote"}
    )
    _invalid(local_remote_gradient)

    no_mask_rule = _mapping()
    no_mask_rule["objectives"][0]["false_negative_mask"] = {
        "enabled": True,
        "metadata_rules": False,
        "similarity_margin": None,
    }
    _invalid(no_mask_rule)


def test_valid_learnable_temperature_and_full_tuning_configs_parse() -> None:
    document = _mapping()
    document["objectives"][0]["temperature"] = {
        "type": "learnable",
        "value": 0.2,
        "minimum": 0.1,
        "maximum": 0.5,
        "learning_rate": 1e-5,
    }
    validate_config_mapping(document)

    full = _mapping()
    full["model"]["tuning"]["method"] = "full"
    full["model"]["tuning"].pop("lora")
    validate_config_mapping(full)


def test_matryoshka_weights_cover_empty_negative_order_and_zero_cases() -> None:
    for dimensions in (
        {},
        {512: -0.1, 1024: 1.1},
        {1024: 0.5, 512: 0.5},
        {512: 0.0, 1024: 0.0},
    ):
        document = _mapping()
        document["objectives"][1]["dimensions"] = dimensions
        _invalid(document)


def test_evaluation_metrics_dimensions_and_slices_use_canonical_ordered_values() -> None:
    for metrics in (["Recall@10"], ["auc"], ["mrr@10", "mrr@10"]):
        document = _mapping()
        document["evaluation"]["metrics"] = metrics
        _invalid(document)

    for dimensions in ([512.0], [-1], [1024, 512], [512, 512]):
        document = _mapping()
        document["evaluation"]["dimensions"] = dimensions
        _invalid(document)

    for slices in (["category", "category"], [""]):
        document = _mapping()
        document["evaluation"]["slices"] = slices
        _invalid(document)


def test_objective_cardinality_and_mrl_evaluation_contracts_fail_early() -> None:
    no_retrieval = _mapping()
    no_retrieval["objectives"] = [no_retrieval["objectives"][1]]
    _invalid(no_retrieval)

    two_retrieval = _mapping()
    two_retrieval["objectives"].append(deepcopy(two_retrieval["objectives"][0]))
    two_retrieval["objectives"][-1]["name"] = "retrieval-two"
    _invalid(two_retrieval)

    two_mrl = _mapping()
    two_mrl["objectives"].append(deepcopy(two_mrl["objectives"][1]))
    two_mrl["objectives"][-1]["name"] = "mrl-two"
    _invalid(two_mrl)

    no_mrl_multiple_dimensions = _mapping()
    no_mrl_multiple_dimensions["objectives"] = [no_mrl_multiple_dimensions["objectives"][0]]
    _invalid(no_mrl_multiple_dimensions)

    no_mrl_one_dimension = _mapping()
    no_mrl_one_dimension["objectives"] = [no_mrl_one_dimension["objectives"][0]]
    no_mrl_one_dimension["evaluation"]["dimensions"] = [4096]
    validate_config_mapping(no_mrl_one_dimension)
