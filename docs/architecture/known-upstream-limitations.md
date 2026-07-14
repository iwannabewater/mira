# Known upstream limitations

This document records behavior in MS-SWIFT v4.4.1 that MIRA must isolate. It is not a general assessment of MS-SWIFT.

| Upstream behavior | Consequence for retrieval training | MIRA boundary |
| --- | --- | --- |
| `_embedding_encode` selects `positive[0]` | Additional positives disappear before the loss | Parse MIRA records first; expand explicitly for a Swift-only backend or use the MIRA collator |
| Groups are reconstructed from positional labels | Packing, malformed labels, or a different collator can change supervision | Carry an explicit `GroupLayout` through the batch |
| InfoNCE parameters live in environment variables | A run cannot be reproduced from one resolved algorithm config | Put every objective field in the typed YAML schema and manifest |
| Remote gathered embeddings are detached | The gradient meaning is fixed implicitly | Require `none`, `detached_remote`, or `autograd` in config and test each mode |
| MRL dimensions above hidden size are skipped with a warning | A misspelled dimension silently changes the objective | Reject invalid dimensions during configuration loading |
| Hard-negative shortages can be filled by random duplication | Duplicate negatives change weighting and depend on untracked NumPy state | Let the sampler construct a deterministic candidate set and record its state |
| False-negative masking uses a similarity threshold only | Known product families can still become negatives | Build metadata masks before similarity-aware filtering |
| Loss configuration does not express task compatibility | Unrelated tasks may share an in-batch denominator | Require task-homogeneous global batches unless compatibility is declared |
| The official Qwen helper catches media preprocessing errors and substitutes `NULL` text | Corrupt media can train against the wrong content | Raise a media exception with record ID, URI, and decoder context |
| Generic model conversion relies on a forward hook | Output shape and normalization are implicit | Adapt model output once and attach explicit representation metadata |
| The Megatron example omits optimizer and RNG state | Its checkpoint cannot reproduce an interrupted run | Reject `resume_full` when any required state is absent |
| Core package requirements install a broad application stack | A research library inherits unrelated dependency and vulnerability surface | Do not publish a backend extra until the complete pinned graph passes security audit |

## Unsupported claims

MIRA does not claim that MS-SWIFT 4.4.1 supports multi-positive training, autograd-safe remote negatives, global-query loss scaling with unequal local batches, or full-resume semantics for the audited Megatron example. MIRA implements those contracts in its own modules. The facade uses Swift for model, processor, template, and trainer entry points, and it uses the PEFT package for LoRA construction. It delegates checkpoint save and resume to MIRA's `CheckpointManager` and export to MIRA's Qwen adapter; it does not claim Swift-native checkpoint support.

MIRA also does not claim Qwen3-VL-Embedding audio support. The core schema accepts audio so another adapter can declare it, while the Qwen adapter rejects it at configuration validation.

MIRA 0.1 does not publish an MS-SWIFT extra. The 4.4.1 dependency graph keeps
known advisories in the compatible Gradio, Pillow, and Starlette versions, so
backend installation is security-blocked. The source-level interoperability
audit remains useful, but it is not runtime evidence.
