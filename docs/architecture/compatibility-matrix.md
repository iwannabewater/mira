# Compatibility matrix

This matrix separates a declared upstream range from a configuration or runtime that MIRA has executed. A version range is not test evidence.

## Release baseline

| Surface | MIRA boundary | Source status | Runtime status |
| --- | --- | --- | --- |
| Python | declared target: 3.11 and 3.12 | MS-SWIFT classifies support through 3.12; its all-dependencies helper recommends 3.10 or 3.11 | 902 tests passed on Python 3.12.13 and in a fresh locked Python 3.11.15 environment |
| Operating system | Linux and macOS; Windows unsupported | Secure config, media, and local-export reads require POSIX `dir_fd`, `O_DIRECTORY`, and `O_NOFOLLOW` | macOS 26.5.1 local gates and Ubuntu 24.04 CI target; unsupported hosts fail closed |
| MS-SWIFT | source boundary pinned to exactly 4.4.1 | Official tag `98a09c18cdf95ff07051324b9b8cc90f5184b24b` | `NOT RUN`; installation security-blocked by its Gradio 5 dependency graph |
| PyTorch | `>=2.8,<3` | Qwen model card uses 2.8.0 | PyTorch 2.13.0 CPU/Gloo lane `PASS`; CUDA `NOT RUN` |
| Transformers | audited intersection `>=4.57,<5.13` | Intersection of Qwen `>=4.57` and MS-SWIFT `<5.13` | Not installed by MIRA 0.1; real adapter import `NOT RUN` |
| PEFT | `>=0.11,<0.20` | MS-SWIFT requirement | LoRA injection `NOT RUN` |
| Accelerate | no released MIRA dependency range | MS-SWIFT does not declare a lower or upper bound | Not required by the CPU/Gloo lane; backend runtime `NOT RUN` |
| DeepSpeed | audited upstream hint `<0.19`; no released MIRA extra | MS-SWIFT all-dependencies helper | ZeRO-2 and ZeRO-3 `NOT RUN` |
| qwen-vl-utils | audited upstream minimum `>=0.0.14`; no released MIRA extra | Model registration and official model card | Media preprocessing `NOT RUN` |
| decord | required by the audited MS-SWIFT registration; no released MIRA extra | MS-SWIFT Qwen3-VL registration | Video decoding `NOT RUN` |

## Model capability boundary

| Model | Global dimension | Declared modalities | MRL configuration ceiling | MIRA release state |
| --- | ---: | --- | ---: | --- |
| Qwen3-VL-Embedding-2B | 2,048 | text, image, screenshot, video, mixed | 2,048 | Config validation available; model weights `NOT RUN` and backend installation security-blocked |
| Qwen3-VL-Embedding-8B | 4,096 | text, image, screenshot, video, mixed | 4,096 | Config validation available; model weights `NOT RUN` and backend installation security-blocked |

Audio is part of MIRA's domain contract but is not declared by the audited Qwen3-VL-Embedding model. Configuration validation must reject an audio request for this adapter.

## Execution matrix

The project records each lane as `PASS`, `FAIL`, or `NOT RUN`. A skipped job cannot become `PASS`.

| Lane | Required evidence | Current status |
| --- | --- | --- |
| Core unit, mathematical, and property tests | Final uv-managed pytest and coverage output | `PASS`; 902 tests and 94.66% branch coverage on Python 3.12.13 |
| Two-rank Gloo gather and target construction | `uv run pytest --no-cov -m distributed` | `PASS` on CPU/Gloo |
| Tiny end-to-end train, save, resume, evaluate, export, reload | Integration test with a deterministic tiny adapter | `PASS` within the integrated CPU reference suite |
| Qwen3-VL-Embedding-2B single-GPU LoRA | CUDA runtime log and smoke manifest | `NOT RUN`, no CUDA device or weights |
| Qwen3-VL-Embedding-2B two-GPU DDP | Two-rank CUDA runtime log and smoke manifest | `NOT RUN`, no CUDA devices or weights |
| Qwen3-VL-Embedding-8B config | `mira inspect` on the immutable reference config | `PASS` on the release host |
| Qwen3-VL-Embedding-8B forward or training | CUDA runtime log and smoke manifest | `NOT RUN`, no CUDA device or weights |
| Phase-1 training-time validation | Validation metrics tied to a training checkpoint and validation dataset | `NOT RUN`, the CLI records provenance but has no validation loop |
| DeepSpeed ZeRO-2 and ZeRO-3 | Backend-specific integration logs | `NOT RUN`, non-CUDA host |
| FlashAttention 2 | Qwen forward and backward smoke record | `NOT RUN`, no compatible accelerator runtime |
| Exact search against FAISS flat inner product | Shared fixture with identical ranking | `NOT RUN`, optional dependency lane not recorded |
| Trained-checkpoint model quality | Versioned dataset, relevance policy, trained artifact, and query-level metrics | `NOT RUN`, no trained MIRA model artifact or quality report |

The release notes and model registry must use these states verbatim. The
MS-SWIFT security blocker and absence of hardware reduce the supported scope;
neither condition justifies a simulated pass.
