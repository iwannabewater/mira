# MIRA

MIRA is a framework for post-training multimodal embedding and retrieval models. It keeps retrieval data semantics, objectives, distributed collectives, evaluation, and reproducibility independent from any one model-training backend.

The 0.1 release scope is a CPU reference framework and a versioned MS-SWIFT
integration boundary. Publication requires every pre-tag gate in the
[release ledger](https://github.com/iwannabewater/mira/blob/main/docs/release-status.md) to pass. MIRA 0.1 does not publish an
MS-SWIFT dependency extra because the 4.4.1 dependency graph is
security-blocked. Qwen3-VL-Embedding execution, CUDA, NCCL, DeepSpeed,
FlashAttention, training-time validation, and model-quality evaluation remain
`NOT RUN`.

## Design boundary

MIRA owns the parts of retrieval training that determine mathematical and data correctness:

- typed, versioned configuration with a saved checksum
- multi-positive records and explicit `GroupLayout` tensors
- InfoNCE, MRL, false-negative masks, and distributed loss scaling
- deterministic sampling, checksummed checkpoint manifests, and explicit resume modes
- exact retrieval metrics and export semantics

The dependency-lazy MS-SWIFT facade covers model and processor loading, Qwen templates, multimodal preprocessing, upstream trainer construction, and PEFT. It binds checkpoint save and resume to MIRA's `CheckpointManager` and exports through MIRA's model adapter; it does not read or write Swift-native checkpoints. Core packages do not import it. The source boundary is available for audit, but this release has no supported backend installation command.

The Phase-1 training CLI records validation-dataset provenance but does not run
a training-time validation loop. `mira evaluate` is a separate exact evaluator
for precomputed, provenance-bound artifacts. No Qwen validation run has been
executed.

## Current compatibility target

| Surface | Boundary | Evidence state |
| --- | --- | --- |
| Python | declared target: 3.11 and 3.12 | 902 tests passed on Python 3.12.13 and in a fresh locked Python 3.11.15 environment |
| Operating system | Linux and macOS; Windows unsupported | POSIX descriptor-relative security boundary; macOS local gates and Ubuntu 24.04 CI target |
| PyTorch | `>=2.8,<3` | PyTorch 2.13.0 CPU/Gloo and integrated CPU reference gates `PASS` |
| MS-SWIFT | source boundary pinned to 4.4.1 | source-audited at commit `98a09c18...`; dependency installation security-blocked |
| Qwen3-VL-Embedding-2B and 8B | experimental adapter targets | model-weight smoke tests `NOT RUN` |
| CUDA, NCCL, DeepSpeed, FlashAttention | experimental | `NOT RUN` on the current host |

Read the [upstream audit](https://github.com/iwannabewater/mira/blob/main/docs/architecture/upstream-audit.md), [compatibility matrix](https://github.com/iwannabewater/mira/blob/main/docs/architecture/compatibility-matrix.md), and [integration ADR](https://github.com/iwannabewater/mira/blob/main/docs/adr/0001-upstream-integration-strategy.md) before enabling an external backend.

## Start here

Use the [installation guide](https://github.com/iwannabewater/mira/blob/main/docs/getting-started/installation.md) to create the
locked environment, then follow the [quickstart](https://github.com/iwannabewater/mira/blob/main/docs/getting-started/quickstart.md)
for configuration inspection, dataset validation, exact evaluation, and the
bounded reference lifecycle. The [data schema](https://github.com/iwannabewater/mira/blob/main/docs/data/schema.md) defines the
multi-positive JSONL contract, and the [release ledger](https://github.com/iwannabewater/mira/blob/main/docs/release-status.md)
separates pre-tag readiness from unavailable hardware, model, and
post-publication lanes. Version-specific notes are in the
[0.1.0 release document](https://github.com/iwannabewater/mira/blob/main/docs/releases/v0.1.0.md).

## Development setup

```bash
make sync
make check
```

`uv.lock` is the dependency record. Ruff owns formatting and linting; mypy runs in strict mode. Pytest enforces branch coverage, mathematical checks, property tests, distributed tests, and the tiny end-to-end reference loop.

## Release honesty

MIRA uses `stable`, `experimental`, and `planned` capability states. A stable entry needs a schema, implementation, mathematical and gradient tests, an integration test, documentation, and a compatibility statement. Missing hardware or model weights produce `NOT RUN`, never a simulated pass.

## License

MIRA is licensed under Apache License 2.0. See `LICENSE` and `NOTICE`. Training
checkpoint provenance records model licenses and training plus validation
dataset licenses. Export manifests record the model revision and model licenses;
they do not transfer dataset rights. `NOASSERTION` in an example dataset config
means that no license conclusion has been established; it is not permission to
use or redistribute the data.
