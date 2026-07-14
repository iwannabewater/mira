# Release status

Snapshot date: 2026-07-15.

This ledger records what ran on the release host. `PASS` requires direct
evidence, `NOT RUN` means the lane did not execute, and `PENDING` means the
required evidence has not yet been recorded. A skipped or mocked lane cannot
become `PASS`.

## Required pre-tag gates

| Gate | Status | Evidence or required update |
| --- | --- | --- |
| Git repository initialized on `main` | `PASS` | Local repository state |
| uv lock and frozen base sync | `PASS` | Python 3.12.13 environment resolved from `uv.lock` |
| Upstream MS-SWIFT and Qwen source audit | `PASS` | Architecture audit and ADR with immutable revisions |
| Two-rank CPU/Gloo distributed verifier | `PASS` | Two torchrun tests passed with `uv run pytest --no-cov -m distributed` |
| Integrated Python 3.12 `make check`, including coverage | `PASS` | 902 tests passed; 94.66% total branch coverage; Ruff, strict mypy, import contracts, and vulture passed |
| Python 3.11 verification | `PASS` | 902 tests passed on Python 3.11.15 in a fresh isolated environment resolved from `uv.lock` |
| Dependency, import, and dead-code audit | `PASS` | Both import contracts held; vulture passed; current-platform exports and a marker-independent audit of all 95 registry pins in `uv.lock`, including FAISS and Linux-only CUDA pins, had no known advisories |
| Documentation consistency, links, and punctuation | `PASS` | All 35 documentation files passed punctuation and byte checks; all 34 Markdown files passed local-link, anchor, fence, and JSON-example validation |
| Wheel and source archive build | `PASS` | Hatch built `mira_embed-0.1.0-py3-none-any.whl` and `mira_embed-0.1.0.tar.gz` through uv |
| Isolated wheel-install smoke test | `PASS` | Python 3.12 isolated install, dependency check, import, version, root help, and all nine subcommand help surfaces passed |
| Artifact contents, checksums, license, and notice inspection | `PASS` | Wheel and source archive inventories were inspected; SHA-256 digests were computed; wheel metadata contains `LICENSE` and `NOTICE` |

The pre-tag readiness score is:

```text
100 * (required pre-tag gates marked PASS / all required pre-tag gates)
```

`PENDING`, `FAIL`, and `NOT RUN` rows contribute zero. The release threshold is
strictly greater than 99.9%. Because the eleven required gates have equal,
binary weight, every one must be `PASS`. This snapshot has 11 of 11 required
gates marked `PASS`, for a score of 100.00%. The threshold is met.

Do not tag while any required row remains `PENDING`, `FAIL`, or `NOT RUN`.

## Post-publication verification

These checks happen only after the candidate passes every pre-tag gate. They
are excluded from the readiness score so repository publication, hosted CI,
tagging, and release creation do not become circular prerequisites for
themselves. Their status remains `NOT RUN` in the immutable source snapshot;
the published GitHub release and its hosted checks provide the later evidence.

| Check | Source-snapshot status | Required external evidence |
| --- | --- | --- |
| Public GitHub repository and `origin` remote | `NOT RUN` | Repository URL and remote commit identity |
| Hosted CI at the release commit | `NOT RUN` | Successful workflow URLs for the exact commit |
| Annotated `v0.1.0` tag | `NOT RUN` | Tag object and peeled commit identity |
| GitHub release and uploaded artifacts | `NOT RUN` | Release URL, asset list, and published checksums |
| Published artifact download verification | `NOT RUN` | Fresh downloads matching the published checksums |

## Runtime execution matrix

| Lane | Status | Scope of evidence |
| --- | --- | --- |
| CPU/Gloo two-rank collectives | `PASS` | Unequal rows, detached-remote and autograd gather, DDP global mean, mismatch detection |
| Integrated CPU reference suite | `PASS` | All 902 tests passed on Python 3.12.13 and again on Python 3.11.15 in a fresh environment |
| Phase-1 training-time validation | `NOT RUN` | CLI records validation-source provenance but does not execute a validation loop |
| MS-SWIFT 4.4.1 installation | `NOT RUN` | Dependency graph security-blocked; MIRA 0.1 publishes no backend extra |
| CUDA | `NOT RUN` | No CUDA device on the snapshot host |
| NCCL | `NOT RUN` | Requires CUDA environment |
| Multi-node or multi-GPU Qwen training | `NOT RUN` | No compatible backend, model weights, or accelerator runtime record |
| DeepSpeed ZeRO-2 | `NOT RUN` | No compatible accelerator runtime record |
| DeepSpeed ZeRO-3 | `NOT RUN` | No compatible accelerator runtime record |
| FlashAttention 2 | `NOT RUN` | No compatible accelerator runtime record |
| Qwen3-VL-Embedding-2B weights | `NOT RUN` | No model download, forward, training, export, or reload record |
| Qwen3-VL-Embedding-8B weights | `NOT RUN` | No model download, forward, training, export, or reload record |
| Real Qwen PEFT injection or merge | `NOT RUN` | No compatible backend installation or loaded Qwen model |
| Training or retrieval model quality | `NOT RUN` | No trained model artifact and versioned evaluation dataset report |
| FAISS optional evaluator | `NOT RUN` | No optional-dependency parity record in this snapshot |

CPU/Gloo evidence does not support a CUDA, NCCL, multi-node, DeepSpeed, FlashAttention, or Qwen model claim.

## Capability states

The release candidate implements these backend-independent surfaces:

- versioned strict configuration and checksums
- versioned multi-positive multimodal records and JSONL validation
- explicit `GroupLayout`, no-duplicate sampling, and false-negative masks
- multi-positive InfoNCE and MRL reference objectives
- variable-row tensor collectives and global-query loss scaling
- exact retrieval metrics, confidence intervals, and slice reports
- reference training step, atomic checkpoint lifecycle, inference transforms, and exact mining
- a dependency-lazy MS-SWIFT and Qwen source adapter boundary pinned for audit

Implementation alone is not enough to declare a release stable. The test,
coverage, audit, wheel, and artifact gates above provide the recorded evidence
for this CPU reference release.

The MS-SWIFT runtime path is excluded from the installable release because its
4.4.1 dependency graph is security-blocked. Qwen model-weight execution is
also `NOT RUN`. Identity objectives, reranker distillation, token-level late
interaction, DeepSpeed production launch, and unverified hardware paths remain
`planned` or experimental as their individual documents state.

## Release scope

The first release claims a verified CPU reference framework because every
required pre-tag gate passed. It retains the `NOT RUN` accelerator, Qwen,
training-validation, and model-quality rows in its release notes. It cannot
claim trained Qwen quality, accelerator speed, production DeepSpeed support,
or retrieval improvements without new evidence.

If a required gate cannot run, keep the release blocked or narrow the published claim explicitly. Do not replace missing evidence with a simulated pass.
