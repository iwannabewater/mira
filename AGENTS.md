# MIRA engineering guide

## Project map

- `src/mira/core`: dependency-light contracts, errors, registries, and manifests.
- `src/mira/config`: versioned Pydantic schema, composition, interpolation, migration, and checksums.
- `src/mira/data`: records, validation, conversion, collators, and deterministic sampling.
- `src/mira/objectives`: mathematical objectives that consume `GroupLayout`.
- `src/mira/distributed`: tensor collectives and loss-scaling rules.
- `src/mira/training`: backend-independent engine and checkpoint lifecycle.
- `src/mira/evaluation`: exact retrieval and metrics.
- `src/mira/integrations`: optional, versioned backend adapters.
- `tests`: unit, property, distributed, integration, regression, and smoke evidence.

## Boundaries

Core, data, objectives, mining, and evaluation must not import MS-SWIFT. The MS-SWIFT adapter is optional and pinned to an audited release. Model-name checks belong in model adapters, never in objectives or the trainer.

`GroupLayout` is the only source of query, positive, negative, and mask semantics. Do not infer groups from labels or flattened positions. Configuration errors and unsupported capability combinations fail before training. Do not add silent fallbacks.

Unimplemented work stays in documentation as `planned`. Do not register empty algorithms, commands, or adapters. Hardware or model checks that did not run remain `NOT RUN` in manifests and release notes.

## Verification

Run `make check` before review. Changes to distributed code also require the two-rank command documented in `docs/distributed/ddp.md`. Changes to packaging or CLI behavior require an isolated wheel-install smoke test. Release readiness additionally requires `make audit`, the coverage gate, the smoke-status ledger, and artifact inspection.

## Long-running work

Stop and report state when two consecutive checkpoints make no progress, the same failure appears three times, the declared time or compute budget is exhausted, or an external credential, merge conflict, or unavailable service blocks progress. Do not loop on the same hypothesis.

## Ownership

The contracts in `core`, the tensor semantics in `objectives`, and the collective order in `distributed` are high-risk boundaries. Changes there require mathematical tests and a focused review. The command named in this file is the stable verifier for agents and maintainers.

The checkpoint lifecycle in `src/mira/training/checkpoint.py` and its focused
regression suite in `tests/unit/test_checkpoint.py` form one security-sensitive
hotspot. Preserve the manifest schema, atomic save behavior, trusted-digest
boundary, resume-mode separation, and all-or-nothing RNG restoration. Any
change to either file requires `uv run pytest tests/unit/test_checkpoint.py`
followed by `make check`.

The strict experiment contract in `src/mira/config/schema.py` is owned together
with `tests/unit/test_config_validation.py` and
`tests/unit/test_config_loader.py`. Preserve exact scalar typing, immutable
validated state, capability checks, and early rejection of ambiguous input.
Changes require `uv run pytest tests/unit/test_config_validation.py
tests/unit/test_config_loader.py` followed by `make check`.

The adversarial data suite in `tests/unit/test_data_validation.py` owns the
untrusted JSONL, media-path, fingerprint, and sampling boundary across
`src/mira/data`. Preserve lossless user content while rejecting malformed
identifiers, path escapes, aliasing, and non-finite tensor inputs. Changes
require `uv run pytest tests/unit/test_data_validation.py` followed by
`make check`.

The cross-module contract suite in
`tests/unit/test_runtime_contract_coverage.py` owns runtime validation for
core contracts, objective results, observability records, and training-step
mutation ordering. Keep its assertions behavioral and paired with the source
contract they protect. Changes require `uv run pytest
tests/unit/test_runtime_contract_coverage.py` followed by `make check`.

The MS-SWIFT boundary suite in `tests/unit/test_ms_swift_boundary.py` owns the
optional-integration isolation contract. Preserve lazy imports, exact audited
version checks, one-time model construction, full Qwen3-VL-Embedding naming on
public surfaces, and the upstream-only `qwen3_vl_emb` implementation identifier.
Changes to the MS-SWIFT adapter or this suite require `uv run pytest
tests/unit/test_ms_swift_boundary.py` followed by `make check`.

The CLI model lifecycle suite in
`tests/integration/test_cli_model_lifecycle.py` owns command-level preflight,
model-source trust, checkpoint/export separation, and encode artifact
publication. Preserve fail-before-load validation, immutable remote model
identity, descriptor-bound local reads, and atomic output behavior. Changes to
model lifecycle commands or this suite require `uv run pytest
tests/integration/test_cli_model_lifecycle.py` followed by `make check` and an
isolated wheel-install smoke test.

The encoded artifact protocol in `src/mira/inference/artifacts.py` and
`tests/unit/test_inference_artifacts.py` is a security-sensitive I/O hotspot.
Preserve the v2 manifest schema, descriptor-bound no-symlink reads, exact file
set and checksum verification, bounded snapshots, global item-ID and role
semantics, normalization checks, one-shard reader memory bound, full shard
validation before budgeted materialization, and atomic no-replace publication.
Changes require `uv run pytest
tests/unit/test_inference_artifacts.py tests/integration/test_cli_workflows.py`
followed by `make check` and an isolated wheel-install smoke test.
