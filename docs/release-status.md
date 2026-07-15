# Release status

Snapshot date: 2026-07-15. Candidate: MIRA 0.2.0.

`PASS` requires direct evidence. `NOT RUN` means the lane did not execute. A
mock, configuration check, or declared dependency range cannot replace runtime
evidence.

## Required pre-tag gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Repository and release identity | `PASS` | `main` tracks `origin/main`; package and CLI report 0.2.0 |
| Frozen dependency lock | `PASS` | uv resolved 113 registry pins and `uv lock --check` passed |
| Python 3.12 integrated verifier | `PASS` | 972 tests, 92.68% branch coverage, Ruff, Linux/macOS strict mypy, import contracts, and vulture |
| Python 3.11 isolated verifier | `PASS` | Fresh 3.11.15 environment passed the same 972 tests at 92.68% branch coverage and the static gates |
| Two-rank distributed verifier | `PASS` | Two CPU/Gloo torchrun tests passed with the documented command |
| Real Qwen3-VL-Embedding-2B inference | `PASS` | Immutable remote revision passed text and image inference on Apple MPS/FP16 |
| Model lifecycle and adversarial contracts | `PASS` | Tiny train/resume/export/reload/encode lifecycle plus source, media, checkpoint, and artifact tamper tests |
| Dependency and architecture audit | `PASS` | All registry pins and base, dev, FAISS, and inference graphs had no known advisories; both import contracts held |
| Documentation and naming consistency | `PASS` | Link, reference, punctuation, and full Qwen3-VL-Embedding naming audits passed |
| Wheel and source archive build | `PASS` | `mira_embed-0.2.0` wheel and source archive built from the candidate tree |
| Isolated installed-package smoke | `PASS` | Fresh Python 3.12 environment installed 49 packages, generated a packaged config, passed inference doctor, import, inspect, version, and 12 command help checks |
| Artifact contents and checksums | `PASS` | Wheel and source inventories, metadata, packaged config identity, license files, notice, sizes, and SHA-256 digests were inspected |

The readiness score is:

```text
100 * (required pre-tag gates marked PASS / all required pre-tag gates)
```

Every required gate has equal binary weight. All 12 of 12 gates are `PASS`, so
the local readiness score is 100.00%, above the required 99.9% threshold.

## Runtime evidence

| Lane | Status | Evidence boundary |
| --- | --- | --- |
| Qwen3-VL-Embedding-2B direct text inference | `PASS` | MPS/FP16, shape `[1, 512]`, finite, normalized, official revision and template |
| Qwen3-VL-Embedding-2B direct image plus text inference | `PASS` | MPS/FP16 CLI encode into a verified v2 document artifact |
| Qwen3-VL-Embedding-2B real export or PEFT merge | `NOT RUN` | No trained real-model checkpoint |
| Qwen3-VL-Embedding-8B weights | `NOT RUN` | Construction contract only; no weight execution |
| CPU/Gloo two-rank collectives | `PASS` | Mathematical and gradient semantics only |
| CUDA and NCCL | `NOT RUN` | No CUDA device on the release host |
| DeepSpeed and FlashAttention | `NOT RUN` | No compatible accelerator evidence |
| MS-SWIFT 4.4.1 installation | `NOT RUN` | Audited dependency graph remains security-blocked |
| Real Qwen3-VL-Embedding training and model quality | `NOT RUN` | No trained artifact or versioned quality report |
| FAISS parity and throughput | `NOT RUN` | Dependency graph audited; runtime lane not executed |

The implemented video contract is not real-video execution evidence. The 2B
result is not evidence for 8B, CUDA, training, export, or retrieval quality.

## Distribution artifacts

| Artifact | Local inspection |
| --- | --- |
| `mira_embed-0.2.0-py3-none-any.whl` | Metadata, entry point, package data, license, notice, and installed runtime `PASS` |
| `mira_embed-0.2.0.tar.gz` | Source, tests, docs, configs, agent guide, license, and notice inventory `PASS` |

Embedding a source archive's own digest in an included document would be
self-referential. Exact local digests are recorded in the release execution
report; hosted artifact digests are recorded in the GitHub release after fresh
download verification.

## Post-publication verification

These checks occur only after the local candidate reaches 100.00%. They do not
inflate the pre-tag score.

| Check | Source-snapshot status | Required evidence |
| --- | --- | --- |
| Push release commit to public `main` | `NOT RUN` | Remote commit identity equals the reviewed local commit |
| Hosted CI for the release commit | `NOT RUN` | Every required GitHub Actions job succeeds for that SHA |
| Annotated `v0.2.0` tag | `NOT RUN` | Tag object peels to the verified release commit |
| GitHub release and artifacts | `NOT RUN` | Release URL and exact asset inventory |
| Published artifact download verification | `NOT RUN` | Fresh downloads match hosted checksums and installed version |

Do not tag or publish while a required local gate is not `PASS`. Do not mark a
post-publication row `PASS` inside the immutable source snapshot.
