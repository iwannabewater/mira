# MS-SWIFT upstream audit

Audit date: 2026-07-14

MIRA integrates with a released MS-SWIFT version through an optional adapter. This audit uses immutable revisions so that a future reader can reproduce every source-level claim.

## Audited revisions

| Component | Revision | Evidence | Use in MIRA |
| --- | --- | --- | --- |
| MS-SWIFT release | `v4.4.1`, commit `98a09c18cdf95ff07051324b9b8cc90f5184b24b` | [release](https://github.com/modelscope/ms-swift/releases/tag/v4.4.1), [version file](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/swift/version.py#L1-L5) | Compatibility baseline |
| MS-SWIFT `main` | `9938a463946beb66d6b502d6f3a7dc64845c4df1` at audit time | [commit](https://github.com/modelscope/ms-swift/commit/9938a463946beb66d6b502d6f3a7dc64845c4df1) | Drift signal only, never a release dependency |
| Qwen3-VL-Embedding-2B | `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` | [model repository](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/tree/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda) | Prompt, pooling, and capability reference |
| Qwen3-VL-Embedding-8B | `2c4565515e0f265c6511776e7193b22c0968ddc7` | [model repository](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/tree/2c4565515e0f265c6511776e7193b22c0968ddc7) | Configuration and smoke-test target |

The latest release was resolved through the GitHub API during the audit. Search indexes still returned v4.2.2, which demonstrates why MIRA records immutable revisions instead of relying on indexed documentation.

## Dependency surface

MS-SWIFT 4.4.1 declares Python `>=3.8` and lists classifiers through Python 3.12 in [`setup.py`](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/setup.py#L120-L163). Its framework requirements include `transformers>=4.33,<5.13.0`, `peft>=0.11,<0.20`, `datasets>=3.0,<4.8.5`, and an unbounded `accelerate` dependency in [`requirements/framework.txt`](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/requirements/framework.txt). The upstream all-dependencies helper recommends Python 3.10 or 3.11, CUDA 12, and `deepspeed<0.19` in [`requirements/install_all.sh`](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/requirements/install_all.sh#L1-L12).

The Qwen3-VL-Embedding model card requires `transformers>=4.57.0`, `qwen-vl-utils>=0.0.14`, and `torch==2.8.0` for its Transformers example. The 2B config was produced with Transformers 4.57.1 and has a 2,048-dimensional text hidden state. The 8B model exposes up to 4,096 dimensions. MIRA therefore narrows its Swift integration to Python 3.11 and 3.12, Transformers `>=4.57,<5.13`, PEFT `>=0.11,<0.20`, and MS-SWIFT 4.4.1. CUDA, DeepSpeed, FlashAttention, and model-weight checks remain runtime probes because package metadata cannot prove hardware compatibility.

### Security resolution snapshot

The release audit resolved the MS-SWIFT framework requirements together with
the Qwen-specific `transformers>=4.57,<5.13`, `qwen-vl-utils>=0.0.14`, and
`decord` requirements for CPython 3.11 on x86-64 manylinux. The resolver was
uv 0.11.21 in highest-version mode. It produced 151 exact pins, including
Gradio 5.50.0, Pillow 11.3.0, and Starlette 0.52.1. pip-audit 2.10.1 emitted
29 vulnerability records across those three packages, representing 23
distinct PYSEC IDs after duplicate IDs within a package were collapsed.

| Package | Resolved version | Distinct audit IDs | Fixed versions reported by pip-audit |
| --- | --- | --- | --- |
| Gradio | 5.50.0 | `PYSEC-2026-63`, `PYSEC-2026-64`, `PYSEC-2026-65`, `PYSEC-2026-66`, `PYSEC-2026-211`, `PYSEC-2026-2178`, `PYSEC-2026-2179` | 6.6.0, 6.7.0, 6.15.0, and 6.16.0; no fixed version was listed for `PYSEC-2026-211` |
| Pillow | 11.3.0 | `PYSEC-2026-165`, `PYSEC-2026-2249`, `PYSEC-2026-2250`, `PYSEC-2026-2251`, `PYSEC-2026-2252`, `PYSEC-2026-2253`, `PYSEC-2026-2254`, `PYSEC-2026-2255`, `PYSEC-2026-2256`, `PYSEC-2026-2257`, `PYSEC-2026-2874` | 12.1.1, 12.2.0, and 12.3.0, depending on the advisory |
| Starlette | 0.52.1 | `PYSEC-2026-161`, `PYSEC-2026-248`, `PYSEC-2026-249`, `PYSEC-2026-2280`, `PYSEC-2026-2281` | 1.0.1, 1.1.0, 1.3.0, and 1.3.1, depending on the advisory |

The exact release-host commands were:

```bash
uv pip compile \
  /tmp/mira-ms-swift-v4.4.1/requirements.txt \
  /tmp/mira-qwen-overlay.in \
  --python-version 3.11 \
  --python-platform x86_64-manylinux_2_28 \
  --resolution highest \
  --no-header \
  --no-annotate \
  --output-file /tmp/mira-ms-swift-qwen-resolved.txt
uv run pip-audit \
  --strict \
  --no-deps \
  --disable-pip \
  --requirement /tmp/mira-ms-swift-qwen-resolved.txt \
  --format json \
  --output /tmp/mira-ms-swift-qwen-audit.json
```

The overlay contained only the three Qwen-specific requirements named above.
The pinned-resolution SHA-256 was
`f1bae58d1bec495b697c42f0400643b074b2a47f0073a4159f9fbfb795b9fe0f`;
the JSON audit-report SHA-256 was
`552aaa6c336641c6c15b1c7466cb8880c8af19b526bec83f9cc135f8f04950f6`.
The audit command intentionally used `--no-deps` after uv had produced the
fully pinned graph, so pip-audit queried those exact 151 versions without a
second resolver changing them.

Gradio 5.50.0 itself requires `pillow<12.0` and `starlette<1.0`; MS-SWIFT
4.4.1 requires `gradio<6.0`. The reported fixed releases therefore cannot be
selected inside this upstream graph. MIRA excludes the backend dependency
extra instead of suppressing these findings.

## Model registration and Qwen output path

MS-SWIFT registers both Qwen3-VL embedding sizes under `qwen3_vl_emb`. The registration uses `Qwen3VLEmbLoader`, the Qwen3-VL architecture, and explicit requirements for Transformers 4.57, `qwen_vl_utils`, and `decord` in [`swift/model/models/qwen.py`](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/swift/model/models/qwen.py#L2092-L2115).

The Swift template adds an empty assistant turn when needed, sets the default system instruction to `Represent the user's input.`, and terminates with `<|endoftext|>` in [`swift/template/templates/qwen.py`](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/swift/template/templates/qwen.py#L648-L662). Training and inference must preserve all three details.

Swift converts a causal language model into an embedding model through a forward hook. The hook selects the last valid hidden state from `output.logits`, applies L2 normalization, and returns it as `last_hidden_state` in [`swift/model/patcher.py`](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/swift/model/patcher.py#L75-L102). The official Qwen script instead returns token-level hidden states, pools the last valid token using the attention mask, then normalizes in its `process` method. MIRA treats the pooled vector and its normalization state as explicit adapter output metadata so objectives never guess whether normalization already happened.

## Template, dataset, and collator path

`Template.encode` dispatches `task_type='embedding'` to `_embedding_encode`, and `Template.data_collator` dispatches to `_embedding_data_collator` in [`swift/template/base.py`](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/swift/template/base.py#L600-L643). The embedding encoder performs these steps:

1. Encode the anchor.
2. Read `inputs.positive`; if it is a list, select element zero.
3. Encode every explicit negative.
4. Emit positional labels with one positive marker followed by zero-valued negatives.

The implementation is visible in [`_embedding_encode`](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/swift/template/base.py#L509-L546). The collator flattens each record into anchor, positive, and numbered negative entries, then concatenates the positional labels in [`_embedding_data_collator`](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/swift/template/base.py#L1769-L1799). The upstream guide also states that embedding training currently accepts exactly one positive per record in [`Embedding.md`](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/docs/source_en/BestPractices/Embedding.md).

MIRA cannot reuse those labels as its internal batch contract. Its collator must produce a `GroupLayout` with explicit query positions, positive pairs, explicit negatives, invalid pairs, relevance weights, task IDs, and dataset IDs. Conversion to the Swift JSON format either expands a multi-positive record into several records or routes it through the MIRA collator. It never discards extra positives.

The public load specification accepts only the two model IDs and immutable
revisions in the audit table, their official 2,048 or 4,096 dimensions, the
`qwen3_vl_emb` model type, embedding task, and Hugging Face source. These checks
run before the optional MS-SWIFT package is imported. A local MIRA export must
match an externally supplied export-manifest checksum. MIRA copies its exact
descriptor-bound, checksummed, code-free artifact set into a private snapshot
and gives that snapshot, not the mutable source directory, to the upstream
loader.

## Loss, gather, and MRL behavior

MS-SWIFT's InfoNCE implementation reads temperature, in-batch-negative behavior, hard-negative count, and false-negative controls from environment variables. `_parse_multi_negative_sentences` reconstructs groups from labels and may duplicate hard negatives with NumPy sampling. The distributed path gathers objects, replaces the local tensor to retain local gradients, and detaches every remote tensor. The implementation supports a detached-remote gradient policy only. See [`swift/loss/embedding.py`](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/swift/loss/embedding.py#L88-L320).

`EmbeddingTrainer` implements MRL by slicing each requested prefix, normalizing it, and summing weighted losses. A dimension larger than the hidden size is logged and skipped in [`swift/trainers/embedding_trainer.py`](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/swift/trainers/embedding_trainer.py#L11-L35). MIRA rejects the entire configuration before training when a dimension is invalid, weights are negative, weights are all zero, or the dimension order is not strictly increasing.

MIRA owns the objective implementation and exposes `none`, `detached_remote`, and `autograd` gather modes. Variable local shapes require a shape gather, padding, tensor gather, unpadding, and global index reconstruction. Loss reduction defaults to a global-query mean rather than a mean of per-rank means.

## Tuning, packing, and parallelism

MS-SWIFT exposes `freeze_llm`, `freeze_vit`, and `freeze_aligner`, with defaults that train the language model while freezing the visual tower and aligner. It also exposes LoRA, RS-LoRA, and DoRA settings in [`swift/arguments/tuner_args.py`](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/swift/arguments/tuner_args.py#L36-L62). Training arguments provide separate `vit_lr` and `aligner_lr` values in [`swift/trainers/arguments.py`](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/swift/trainers/arguments.py#L55-L73).

The generic template supports packing and padding-free operation for embedding tasks. Those transforms flatten or repack sequences before the model forward pass. A valid embedding layout therefore cannot use token positions as record boundaries after packing. MIRA keeps group identity outside model token tensors and validates any backend combination before the first collective call.

The Hugging Face trainer, native Swift trainer, and Megatron-SWIFT trainer do not share checkpoint or distributed semantics. The Megatron Qwen3-VL embedding example enables sequence parallelism but explicitly disables optimizer and RNG checkpoint state in [`examples/megatron/embedding/qwen3_vl_emb.sh`](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/examples/megatron/embedding/qwen3_vl_emb.sh#L1-L36). Such a checkpoint cannot satisfy MIRA's `resume_full` contract.

## Checkpoint, resume, and export boundary

The Hugging Face path distinguishes full resume from `resume_only_model`; the export pipeline has a separate LoRA merge path. Megatron-SWIFT uses its own checkpoint conversion and export modules. MIRA does not claim these formats are interchangeable.

MIRA writes `manifest.json`, `resolved-config.json`, and `state.pt` in an atomic
checkpoint directory. The manifest records file checksums, component-presence
flags, dependency versions, the MS-SWIFT source version, the resolved config
checksum, dataset checksum, model ID and revision, model and dataset licenses,
planned maximum steps, and template fingerprint. The state file carries the
model, optimizer, scheduler, RNG, sampler, objective, and training counters
saved by the Phase-1 CLI.

The MS-SWIFT facade binds its cached Qwen model to this generic MIRA checkpoint
manager. It does not call Swift checkpoint handlers or claim compatibility with
Swift-native checkpoint formats. Export also runs through MIRA's Qwen adapter;
any PEFT merge is an explicit MIRA CLI step before that export.

`resume_full` verifies the checkpoint and provenance, then restores those
components. The generic checkpoint manager also supports scaler, streaming
dataset, gradient-cache, and curriculum state when both the saved state and a
restore target are supplied. `resume_model_only` restores only the full model.
`load_adapter_for_new_stage` requires separately saved adapter state. All three
modes fail when required state is missing or incompatible. An optional manifest
SHA-256 obtained through a trusted channel can authenticate the manifest before
parsing; internal file checksums alone detect accidental corruption. The real
MS-SWIFT and Qwen resume lane is `NOT RUN` because backend installation is
security-blocked and model weights were not loaded.

## License findings

MS-SWIFT and both Qwen3-VL-Embedding model repositories declare Apache License 2.0. MIRA uses the same project license. MIRA source is original unless a file header says otherwise. No MS-SWIFT implementation is copied into the core package. `NOTICE` identifies MS-SWIFT and Qwen sources used for interoperability research. Training checkpoint provenance keeps model and training plus validation dataset licenses. Export manifests keep the model revision and model licenses, not dataset rights.

## Verification state

The source paths, revisions, declared dependency ranges, prompt template,
positional grouping behavior, detached-remote gather, and MRL skip behavior
were inspected at the immutable revisions above. The installed MS-SWIFT lane
is security-blocked. GPU execution, DeepSpeed, FlashAttention, NCCL,
Qwen3-VL-Embedding-2B training, Qwen3-VL-Embedding-8B training, training-time
validation, and LoRA merge are `NOT RUN`. MIRA must not label those paths
stable until their smoke records exist.
