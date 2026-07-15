# Compatibility matrix

This matrix separates implemented contracts from executed evidence. A version
range or passing mock test is not evidence that a model or accelerator lane ran.

## Runtime baseline

| Surface | MIRA 0.2 boundary | Executed evidence |
| --- | --- | --- |
| Python | `>=3.11,<3.13` | 972 tests passed on Python 3.12.13 and in a fresh Python 3.11.15 environment |
| Operating system | Linux and macOS; Windows unsupported | macOS 26.5.1 local gates passed; Ubuntu 24.04 hosted CI is a post-publication gate |
| PyTorch | `>=2.8,<3` | PyTorch 2.13.0 on CPU/Gloo and Apple MPS |
| Transformers | `>=5.13.1,<5.14` | 5.13.1 imported and loaded the real Qwen3-VL-Embedding-2B revision |
| Accelerate | `>=1.14,<1.15` | 1.14.0 dependency and import checks passed |
| PEFT | `>=0.19.1,<0.20` | 0.19.1 dependency and unit integration checks passed; real adapter injection is `NOT RUN` |
| qwen-vl-utils | `>=0.0.14,<0.0.15` | 0.0.14 real image preprocessing passed |
| torchvision | `>=0.28,<0.29` | 0.28.0 dependency and import checks passed |
| safetensors | `>=0.8,<0.9` | 0.8.0 model loading and encoded artifact lanes passed |
| MS-SWIFT | exactly 4.4.1 at audited commit `98a09c18cdf95ff07051324b9b8cc90f5184b24b` | Source boundary tested; installation remains security-blocked and `NOT RUN` |

The direct inference tuple and the audited MS-SWIFT training tuple do not share
one supported environment. MIRA never changes dependency versions or backend
selection after a failed load.

## Model capability boundary

| Model | Immutable revision | Global dimension | Implemented input contract | Weight evidence |
| --- | --- | ---: | --- | --- |
| Qwen3-VL-Embedding-2B | `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` | 2,048 | text, image, video, and mixed global embeddings | Real text and image inference `PASS` on Apple MPS/FP16; video, training, export, and PEFT merge `NOT RUN` |
| Qwen3-VL-Embedding-8B | `2c4565515e0f265c6511776e7193b22c0968ddc7` | 4,096 | text, image, video, and mixed global embeddings | Config and construction contract `PASS`; weight execution `NOT RUN` |

Audio is not declared by either Qwen3-VL-Embedding target. Token and hybrid
representations remain `planned`. The direct backend rejects silent truncation,
remote code, unverified media, unsupported message roles, unavailable devices,
and unsupported dtype combinations.

## Source compatibility

| Source | Required trust input | Preprocessing profile | Status |
| --- | --- | --- | --- |
| Audited Hugging Face model | Exact model ID and config revision | `official` | Real 2B remote load and forward `PASS` |
| Native MIRA checkpoint | Config plus optional trusted checkpoint manifest SHA-256 | `mira_v1` | Tiny lifecycle `PASS`; real trained checkpoint `NOT RUN` |
| Local MIRA export | Required trusted export manifest checksum | `mira_v1` | Tiny export and reload `PASS`; real 2B export and reload `NOT RUN` |

The export reader accepts only one legacy semantic mapping: the exact v0.1 pair
`pooling=last_eos` and `template=qwen3_vl_emb`. Every new public manifest uses
`last_token` and `qwen3_vl_embedding`.

## Execution matrix

| Lane | Current status | Scope |
| --- | --- | --- |
| Integrated Python 3.12 suite | `PASS` | 972 tests and 92.68% branch coverage |
| Fresh Python 3.11 suite | `PASS` | 972 tests and 92.68% branch coverage |
| Two-rank CPU/Gloo verifier | `PASS` | Unequal rows, gather gradients, loss scaling, and collective ordering |
| Tiny train, save, resume, export, reload, encode, mine, and evaluate lifecycle | `PASS` | Deterministic weight-free integration boundary |
| Real Qwen3-VL-Embedding-2B text and image inference | `PASS` | Apple MPS/FP16, official immutable revision |
| Qwen3-VL-Embedding-8B weight execution | `NOT RUN` | No 8B download or suitable release-host accelerator evidence |
| CUDA and NCCL | `NOT RUN` | No CUDA device on the release host |
| DeepSpeed ZeRO-2 and ZeRO-3 | `NOT RUN` | No compatible accelerator runtime |
| FlashAttention 2 | `NOT RUN` | No CUDA runtime evidence |
| Real Qwen3-VL-Embedding PEFT injection, training, merge, export, or reload | `NOT RUN` | No trained MIRA checkpoint |
| Training-time validation and retrieval quality | `NOT RUN` | No versioned trained-model evaluation report |
| FAISS parity and performance | `NOT RUN` | Optional dependency was audited but no parity record was produced |

`NOT RUN` lanes remain outside the release claim. The source tree, release
notes, and published metadata must not convert them into implied support.
