from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from os import stat_result

import pytest
import yaml

import mira.config.loader as config_loader
from mira.config import (
    apply_overrides,
    config_checksum,
    deep_merge,
    interpolate_environment,
    interpolate_string,
    load_config,
    migrate_config,
    resolve_config,
    write_resolved_config,
)
from mira.core import (
    ConfigurationFileError,
    ConfigurationValidationError,
    InterpolationError,
    MigrationError,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "configs" / "qwen3_vl_ecommerce.yaml"


def test_master_prompt_ecommerce_yaml_parses_as_real_config() -> None:
    config = load_config(FIXTURE)

    assert config.schema_version == 1
    assert config.model.name_or_path == "Qwen/Qwen3-VL-Embedding-8B"
    assert config.model.tuning.lora is not None
    assert config.model.tuning.lora.rank == 16
    assert [objective.type for objective in config.objectives] == [
        "multi_positive_infonce",
        "matryoshka",
    ]
    assert config.evaluation.dimensions == (512, 1024, 2048, 4096)


def test_extends_compose_env_and_cli_overrides_are_deterministic(tmp_path: Path) -> None:
    base_text = FIXTURE.read_text(encoding="utf-8").replace(
        "data/train.jsonl", "${DATA_ROOT}/train.jsonl"
    )
    (tmp_path / "base.yaml").write_text(base_text, encoding="utf-8")
    (tmp_path / "runtime.yaml").write_text("runtime:\n  epochs: 2\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\ncompose:\n  - runtime.yaml\nproject:\n  name: composed\n",
        encoding="utf-8",
    )

    first = resolve_config(
        tmp_path / "child.yaml",
        environ={"DATA_ROOT": "/datasets"},
        overrides=("runtime.epochs=3", "project.seed=7"),
    )
    second = resolve_config(
        tmp_path / "child.yaml",
        environ={"DATA_ROOT": "/datasets"},
        overrides=("runtime.epochs=3", "project.seed=7"),
    )

    assert first.config.project.name == "composed"
    assert first.config.project.seed == 7
    assert first.config.runtime.epochs == 3
    assert first.config.data.train[0].path == "/datasets/train.jsonl"
    assert first.yaml == second.yaml
    assert first.checksum == second.checksum == config_checksum(first.config)
    assert len(first.source_files) == 3


def test_resolved_yaml_round_trips_and_checksum_matches_written_bytes(tmp_path: Path) -> None:
    resolved = resolve_config(FIXTURE)
    output = tmp_path / "resolved_config.yaml"
    checksum = write_resolved_config(resolved.config, output)

    assert output.read_text(encoding="utf-8") == resolved.yaml
    assert checksum == resolved.checksum
    assert load_config(output) == resolved.config

    second_output = tmp_path / "resolved-copy.yaml"
    resolved.write(second_output)
    assert second_output.read_text(encoding="utf-8") == resolved.yaml


def test_missing_environment_variable_fails_with_location(tmp_path: Path) -> None:
    text = FIXTURE.read_text(encoding="utf-8").replace(
        "data/train.jsonl", "${MISSING_ROOT}/train.jsonl"
    )
    path = tmp_path / "missing-env.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(InterpolationError, match="MISSING_ROOT"):
        load_config(path, environ={})


def test_duplicate_yaml_keys_and_composition_cycles_fail(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    with pytest.raises(ConfigurationFileError, match="duplicate YAML key"):
        load_config(duplicate)

    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigurationFileError, match="cycle"):
        load_config(tmp_path / "a.yaml")


def test_invalid_and_duplicate_cli_overrides_fail(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationFileError, match=r"dotted.path=value"):
        load_config(FIXTURE, overrides=("runtime.epochs",))
    with pytest.raises(ConfigurationFileError, match="duplicate CLI override"):
        load_config(FIXTURE, overrides=("runtime.epochs=2", "runtime.epochs=3"))
    with pytest.raises(ConfigurationValidationError, match="extra_forbidden"):
        load_config(FIXTURE, overrides=("runtime.unknown=true",))


def test_v0_migration_handles_documented_renames(tmp_path: Path) -> None:
    current = load_config(FIXTURE).model_dump(mode="json")
    current["schema_version"] = 0
    current["backend"] = "ms_swift"
    model = current["model"]
    model["path"] = model.pop("name_or_path")
    runtime = current["runtime"]
    runtime["per_device_train_batch_size"] = runtime.pop("train_batch_size_per_device")
    data = current["data"]
    data["val"] = data.pop("validation")
    current["loss"] = current.pop("objectives")
    path = tmp_path / "v0.yaml"
    path.write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")

    migrated = load_config(path)
    assert migrated.schema_version == 1
    assert migrated.model.name_or_path == "Qwen/Qwen3-VL-Embedding-8B"
    assert migrated.runtime.train_batch_size_per_device == 2


def test_composition_references_are_confined_to_selected_root_tree(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    child = root / "child.yaml"

    for reference in ("../outside.yaml", str(outside)):
        child.write_text(f"extends: {reference}\n", encoding="utf-8")
        with pytest.raises(ConfigurationFileError, match="selected root") as captured:
            load_config(child)
        assert str(outside) not in str(captured.value)

    symlink = root / "linked.yaml"
    symlink.symlink_to(outside)
    child.write_text("extends: linked.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigurationFileError, match="selected root"):
        load_config(child)


def test_composed_sources_reject_internal_file_and_directory_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real.yaml"
    real.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    linked = tmp_path / "linked.yaml"
    linked.symlink_to(real)
    child = tmp_path / "child.yaml"
    child.write_text("extends: linked.yaml\n", encoding="utf-8")

    with pytest.raises(ConfigurationFileError, match=r"symbolic link.*selected root"):
        load_config(child)

    fragments = tmp_path / "fragments"
    fragments.mkdir()
    (fragments / "base.yaml").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    linked_directory = tmp_path / "linked-directory"
    linked_directory.symlink_to(fragments, target_is_directory=True)
    child.write_text("extends: linked-directory/base.yaml\n", encoding="utf-8")

    with pytest.raises(ConfigurationFileError, match=r"symbolic link.*selected root"):
        load_config(child)


def test_composed_source_replacement_after_secure_read_cannot_change_parsed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    safe_name = "descriptor-bound-safe"
    outside_name = "path-replacement-outside"
    source = tmp_path / "base.yaml"
    source.write_text(
        FIXTURE.read_text(encoding="utf-8").replace(
            "name: mira-qwen3-vl-embedding-ecommerce", f"name: {safe_name}"
        ),
        encoding="utf-8",
    )
    outside = tmp_path.parent / f"{tmp_path.name}-outside.yaml"
    outside.write_text(
        FIXTURE.read_text(encoding="utf-8").replace(
            "name: mira-qwen3-vl-embedding-ecommerce", f"name: {outside_name}"
        ),
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\n", encoding="utf-8")
    saved = tmp_path / "opened-base.yaml"
    original = config_loader._resolve_reference
    swapped = False

    def resolve_then_swap(
        declaring_source: config_loader._ConfigSource,
        reference: str,
        *,
        context: config_loader._CompositionContext,
    ) -> config_loader._ConfigSource:
        nonlocal swapped
        selected = original(
            declaring_source,
            reference,
            context=context,
        )
        if not swapped:
            source.replace(saved)
            source.symlink_to(outside)
            swapped = True
        return selected

    monkeypatch.setattr(config_loader, "_resolve_reference", resolve_then_swap)

    config = load_config(child)

    assert swapped
    assert config.project.name == safe_name
    assert config.project.name != outside_name


def test_composed_directory_replacement_stays_bound_to_open_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    safe_name = "directory-descriptor-safe"
    outside_name = "directory-replacement-outside"
    fragments = tmp_path / "fragments"
    fragments.mkdir()
    (fragments / "base.yaml").write_text(
        FIXTURE.read_text(encoding="utf-8").replace(
            "name: mira-qwen3-vl-embedding-ecommerce", f"name: {safe_name}"
        ),
        encoding="utf-8",
    )
    outside_directory = tmp_path.parent / f"{tmp_path.name}-outside-fragments"
    outside_directory.mkdir()
    (outside_directory / "base.yaml").write_text(
        FIXTURE.read_text(encoding="utf-8").replace(
            "name: mira-qwen3-vl-embedding-ecommerce", f"name: {outside_name}"
        ),
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text("extends: fragments/base.yaml\n", encoding="utf-8")
    held_directory = tmp_path / "opened-fragments"
    original = config_loader._open_bound_component
    swapped = False

    def open_then_swap(
        parent_fd: int,
        component: str,
        *,
        require_directory: bool,
        path_label: str,
    ) -> tuple[int, stat_result]:
        nonlocal swapped
        opened = original(
            parent_fd,
            component,
            require_directory=require_directory,
            path_label=path_label,
        )
        if require_directory and component == "fragments" and not swapped:
            fragments.replace(held_directory)
            fragments.symlink_to(outside_directory, target_is_directory=True)
            swapped = True
        return opened

    monkeypatch.setattr(config_loader, "_open_bound_component", open_then_swap)

    config = load_config(child)

    assert swapped
    assert config.project.name == safe_name
    assert config.project.name != outside_name


def test_interpolated_composition_escape_does_not_echo_environment_value(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    secret_path = str(tmp_path / "do-not-echo-this-secret.yaml")
    child = root / "child.yaml"
    child.write_text("extends: ${SECRET_INCLUDE}\n", encoding="utf-8")

    with pytest.raises(ConfigurationFileError, match="selected root") as captured:
        load_config(child, environ={"SECRET_INCLUDE": secret_path})
    assert secret_path not in str(captured.value)

    missing_name = "must-not-echo-missing-file-name.yaml"
    with pytest.raises(ConfigurationFileError, match="cannot read composed") as captured:
        load_config(child, environ={"SECRET_INCLUDE": missing_name})
    assert missing_name not in str(captured.value)


def test_mapping_form_compose_cannot_modify_sibling_sections(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "runtime.yaml").write_text(
        "runtime:\n  epochs: 2\nproject:\n  name: must-not-leak\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: base.yaml\ncompose:\n  runtime: runtime.yaml\n",
        encoding="utf-8",
    )

    config = load_config(child)
    assert config.runtime.epochs == 2
    assert config.project.name != "must-not-leak"


def test_yaml_aliases_and_normalized_key_collisions_fail_before_validation(
    tmp_path: Path,
) -> None:
    alias = tmp_path / "alias.yaml"
    alias.write_text("schema_version: 1\nloop: &loop [*loop]\n", encoding="utf-8")
    with pytest.raises(ConfigurationFileError, match="aliases are not supported"):
        load_config(alias)

    collision = tmp_path / "collision.yaml"
    collision.write_text(
        'schema_version: 1\nobjectives:\n- dimensions:\n    512: 0.4\n    "512": 0.6\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationFileError, match="normalize identically"):
        load_config(collision)


def test_invalid_file_encoding_and_invalid_root_path_are_wrapped(tmp_path: Path) -> None:
    invalid_utf8 = tmp_path / "invalid.yaml"
    invalid_utf8.write_bytes(b"\xff\xfe")
    with pytest.raises(ConfigurationFileError, match="cannot read selected root"):
        load_config(invalid_utf8)

    with pytest.raises(ConfigurationFileError, match="invalid root configuration path"):
        load_config("embedded\x00null.yaml")


def test_directives_require_nonempty_string_paths(tmp_path: Path) -> None:
    for directive in ("extends: ''\n", "compose: ['']\n", "compose:\n  '': file.yaml\n"):
        path = tmp_path / "invalid-directive.yaml"
        path.write_text(directive, encoding="utf-8")
        with pytest.raises(ConfigurationFileError, match=r"path string|compose section"):
            load_config(path)


def test_override_paths_cannot_overlap_and_values_reject_duplicate_keys() -> None:
    document = {"runtime": {"epochs": 1}}
    with pytest.raises(ConfigurationFileError, match="conflicting CLI override"):
        apply_overrides(document, ("runtime={epochs: 2}", "runtime.epochs=3"))
    with pytest.raises(ConfigurationFileError, match="conflicting CLI override"):
        apply_overrides(document, ("runtime.epochs=3", "runtime={epochs: 2}"))
    with pytest.raises(ConfigurationFileError, match="duplicate YAML key"):
        apply_overrides(document, ("runtime={epochs: 2, epochs: 3}",))
    with pytest.raises(ConfigurationFileError, match="must be strings"):
        apply_overrides(document, cast("tuple[str, ...]", (1,)))
    with pytest.raises(ConfigurationFileError, match="invalid CLI override path"):
        apply_overrides(document, ("runtime..epochs=2",))
    with pytest.raises(ConfigurationFileError, match="is not a mapping"):
        apply_overrides(document, ("runtime.epochs.value=2",))

    created = apply_overrides({}, ("new_section.value=2",))
    assert created == {"new_section": {"value": 2}}


def test_bare_section_fragment_is_scoped_and_nested_relative_include_stays_confined(
    tmp_path: Path,
) -> None:
    (tmp_path / "base.yaml").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    fragments = tmp_path / "fragments"
    fragments.mkdir()
    (fragments / "runtime-values.yaml").write_text("epochs: 2\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: base.yaml\ncompose:\n  runtime: fragments/runtime-values.yaml\n",
        encoding="utf-8",
    )
    assert load_config(child).runtime.epochs == 2

    (fragments / "nested.yaml").write_text("extends: ../base.yaml\n", encoding="utf-8")
    child.write_text("extends: fragments/nested.yaml\n", encoding="utf-8")
    assert load_config(child).project.name == load_config(FIXTURE).project.name


def test_malformed_yaml_root_keys_and_mapping_keys_fail_cleanly(tmp_path: Path) -> None:
    cases = (
        ("malformed.yaml", "schema_version: [\n", "invalid YAML"),
        ("sequence.yaml", "- schema_version\n- 1\n", "YAML mapping"),
        ("root-key.yaml", "1: value\n", "root keys must be strings"),
        ("unhashable-key.yaml", "? [a, b]\n: value\n", "unhashable YAML mapping key"),
    )
    for name, content, message in cases:
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ConfigurationFileError, match=message):
            load_config(path)

    with pytest.raises(ConfigurationFileError, match="mapping keys must be strings"):
        deep_merge({}, cast("dict[str, Any]", {1: "value"}))
    with pytest.raises(ConfigurationFileError, match="strings or integer dimensions"):
        apply_overrides(cast("dict[str, Any]", {(): "value"}), ())


def test_invalid_interpolated_reference_and_atomic_write_errors_are_redacted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid-reference.yaml"
    path.write_text("extends: ${REFERENCE}\n", encoding="utf-8")
    reference = "must-not-echo\x00reference"
    with pytest.raises(ConfigurationFileError, match="invalid configuration reference") as captured:
        load_config(path, environ={"REFERENCE": reference})
    assert reference not in str(captured.value)

    resolved = resolve_config(FIXTURE)
    with pytest.raises(ConfigurationFileError, match="cannot write resolved configuration"):
        resolved.write(tmp_path)


def test_environment_interpolation_is_recursive_literal_and_non_mutating() -> None:
    source: dict[str, Any] = {
        "path": "${ROOT}/data",
        "literal": "$ROOT",
        "nested": ["${ROOT}", ("${ROOT}/x",)],
    }
    resolved = interpolate_environment(source, environ={"ROOT": "/srv"})

    assert resolved == {
        "path": "/srv/data",
        "literal": "$ROOT",
        "nested": ["/srv", ("/srv/x",)],
    }
    assert source["path"] == "${ROOT}/data"
    with pytest.raises(InterpolationError, match="invalid environment interpolation"):
        interpolate_string("${INVALID-NAME}", environ={})
    with pytest.raises(InterpolationError, match="invalid environment interpolation"):
        interpolate_string("${VALUE}", environ={"VALUE": "${NESTED}"})
    with pytest.raises(InterpolationError, match="key"):
        interpolate_environment(cast("dict[str, Any]", {1: "value"}), environ={})


@pytest.mark.parametrize("version", [None, True, 1.0, "1", -1, 2])
def test_migration_rejects_missing_or_inexact_schema_versions(version: object) -> None:
    document: dict[str, Any] = {} if version is None else {"schema_version": version}
    with pytest.raises(MigrationError):
        migrate_config(document)


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [
        ("project", None, "project must be a mapping"),
        ("backend", None, "backend must be a string or mapping"),
        ("model", None, "model must be a string or mapping"),
        ("objectives", 1, "objectives must be a mapping or sequence"),
        ("data", None, "data must be a mapping"),
        ("runtime", None, "runtime must be a mapping"),
    ],
)
def test_v0_migration_rejects_explicitly_malformed_sections(
    section: str, value: object, message: str
) -> None:
    with pytest.raises(MigrationError, match=message):
        migrate_config({"schema_version": 0, section: value})


def test_v0_model_adapter_move_is_explicit_and_ambiguous_forms_fail() -> None:
    migrated = migrate_config(
        {
            "schema_version": 0,
            "model": {"path": "model-id"},
            "model_adapter": "adapter-id",
        }
    )
    assert migrated["model"] == {"name_or_path": "model-id", "adapter": "adapter-id"}

    with pytest.raises(MigrationError, match=r"both model_adapter and model\.adapter"):
        migrate_config(
            {
                "schema_version": 0,
                "model": {"path": "model-id", "adapter": "adapter-id"},
                "model_adapter": "adapter-id",
            }
        )
    with pytest.raises(MigrationError, match="model_adapter must be a string"):
        migrate_config({"schema_version": 0, "model": {"path": "model-id"}, "model_adapter": 1})


@pytest.mark.parametrize(
    "document",
    [
        {"schema_version": 0, "loss": [], "objectives": []},
        {"schema_version": 0, "project": {"name": "x"}, "project_name": "x"},
        {"schema_version": 0, "project": {"seed": 1}, "seed": 1},
        {"schema_version": 0, "model": {"path": "a", "name_or_path": "b"}},
        {"schema_version": 0, "data": {"val": {}, "validation": {}}},
        {
            "schema_version": 0,
            "runtime": {
                "per_device_train_batch_size": 1,
                "train_batch_size_per_device": 1,
            },
        },
    ],
)
def test_v0_migration_rejects_all_ambiguous_old_and_new_spellings(
    document: dict[str, Any],
) -> None:
    with pytest.raises(MigrationError, match="both"):
        migrate_config(document)


def test_current_migration_returns_an_independent_copy() -> None:
    source = {"schema_version": 1, "nested": {"value": 1}}
    migrated = migrate_config(source)
    cast("dict[str, Any]", migrated["nested"])["value"] = 2
    assert source["nested"]["value"] == 1


def _padded_yaml(source: str, byte_count: int) -> str:
    encoded_size = len(source.encode("utf-8"))
    assert encoded_size < byte_count
    padding = byte_count - encoded_size
    if padding == 1:
        return f"{source}\n"
    return f"{source}#{'x' * (padding - 2)}\n"


def _nested_flow_sequence(depth: int) -> str:
    return f"{'[' * depth}0{']' * depth}"


def _write_composition_chain(directory: Path, *, file_count: int) -> Path:
    directory.mkdir()
    for index in range(file_count - 1):
        (directory / f"chain-{index}.yaml").write_text(
            f"extends: chain-{index + 1}.yaml\n", encoding="utf-8"
        )
    (directory / f"chain-{file_count - 1}.yaml").write_text(
        FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return directory / "chain-0.yaml"


def test_root_and_composed_file_byte_limit_boundaries(tmp_path: Path) -> None:
    limit = config_loader._MAX_CONFIG_SOURCE_BYTES
    root = tmp_path / "root.yaml"
    root.write_text(_padded_yaml(FIXTURE.read_text(encoding="utf-8"), limit), encoding="utf-8")
    assert load_config(root).schema_version == 1

    root.write_text(_padded_yaml(FIXTURE.read_text(encoding="utf-8"), limit + 1), encoding="utf-8")
    with pytest.raises(ConfigurationFileError, match=rf"{limit}-byte limit") as captured:
        load_config(root)
    assert str(root) not in str(captured.value)

    fragment = tmp_path / "fragment.yaml"
    fragment.write_text(_padded_yaml("project:\n  name: fragment\n", limit), encoding="utf-8")
    root.write_text(
        f"extends: {fragment.name}\n{FIXTURE.read_text(encoding='utf-8')}", encoding="utf-8"
    )
    assert load_config(root).schema_version == 1

    fragment.write_text(_padded_yaml("project:\n  name: fragment\n", limit + 1), encoding="utf-8")
    with pytest.raises(ConfigurationFileError, match=rf"{limit}-byte limit") as captured:
        load_config(root)
    assert str(fragment) not in str(captured.value)


def test_override_source_byte_limit_is_exact_and_uses_utf8_bytes() -> None:
    limit = config_loader._MAX_CONFIG_SOURCE_BYTES
    boundary = "x" * limit
    assert apply_overrides({"value": ""}, (f"value={boundary}",))["value"] == boundary

    with pytest.raises(ConfigurationFileError, match=rf"{limit}-byte limit"):
        apply_overrides({"value": ""}, (f"value={boundary}x",))
    with pytest.raises(ConfigurationFileError, match=rf"{limit}-byte limit"):
        apply_overrides({"value": ""}, (f"value={'é' * (limit // 2 + 1)}",))


def test_root_and_override_nesting_limits_fail_before_python_recursion(tmp_path: Path) -> None:
    limit = config_loader._MAX_CONTAINER_DEPTH
    root = tmp_path / "nested.yaml"
    root.write_text(
        f"schema_version: 1\nnested: {_nested_flow_sequence(limit - 1)}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationValidationError):
        load_config(root)

    root.write_text(
        f"schema_version: 1\nnested: {_nested_flow_sequence(limit)}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationFileError, match=rf"{limit}-level nesting limit"):
        load_config(root)

    root.write_text(
        f"schema_version: 1\nnested: {_nested_flow_sequence(1_200)}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationFileError, match=rf"{limit}-level nesting limit"):
        load_config(root)

    value = _nested_flow_sequence(limit - 1)
    assert apply_overrides({"value": 0}, (f"value={value}",))["value"]
    with pytest.raises(ConfigurationFileError, match=rf"{limit}-level depth limit"):
        apply_overrides({"value": 0}, (f"value=[{value}]",))


def test_composition_chain_depth_limit_boundary_and_one_over(tmp_path: Path) -> None:
    limit = config_loader._MAX_COMPOSITION_DEPTH
    boundary = _write_composition_chain(tmp_path / "boundary", file_count=limit)
    assert load_config(boundary).schema_version == 1

    over = _write_composition_chain(tmp_path / "over", file_count=limit + 1)
    with pytest.raises(ConfigurationFileError, match=rf"{limit}-file depth limit"):
        load_config(over)


@pytest.mark.parametrize("reference_mode", ["fanout", "repeated"])
def test_composed_file_budget_counts_fanout_and_repeated_references_per_resolution(
    tmp_path: Path, reference_mode: str
) -> None:
    limit = config_loader._MAX_COMPOSED_FILE_LOADS
    repeated_reference = reference_mode == "repeated"
    directory = tmp_path / ("repeated" if repeated_reference else "fanout")
    directory.mkdir()
    if repeated_reference:
        names = ["fragment.yaml"] * limit
        (directory / "fragment.yaml").write_text("project:\n  name: fragment\n", encoding="utf-8")
    else:
        names = [f"fragment-{index}.yaml" for index in range(limit)]
        for index, name in enumerate(names):
            (directory / name).write_text(f"project:\n  name: fragment-{index}\n", encoding="utf-8")

    root = directory / "root.yaml"

    def write_root(references: list[str]) -> None:
        directives = "".join(f"  - {name}\n" for name in references)
        root.write_text(
            f"extends:\n{directives}{FIXTURE.read_text(encoding='utf-8')}",
            encoding="utf-8",
        )

    write_root(names[: limit - 1])
    assert len(resolve_config(root).source_files) == (2 if repeated_reference else limit)

    write_root(names[:limit])
    with pytest.raises(ConfigurationFileError, match=rf"{limit}-file limit"):
        load_config(root)

    assert load_config(FIXTURE).schema_version == 1


@pytest.mark.skipif(
    not hasattr(config_loader.os, "mkfifo"), reason="host has no POSIX FIFO support"
)
def test_composed_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "fragment.yaml"
    config_loader.os.mkfifo(fifo)
    root = tmp_path / "root.yaml"
    root.write_text(
        f"extends: {fifo.name}\n{FIXTURE.read_text(encoding='utf-8')}",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationFileError, match="regular file"):
        load_config(root)
