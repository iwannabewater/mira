# Changelog

This project follows Semantic Versioning. Release entries describe user-visible behavior and record any required `NOT RUN` hardware or model checks.

## 0.2.0 - 2026-07-15

MIRA 0.2 completes the strict direct inference lifecycle for
Qwen3-VL-Embedding and turns encoded vectors into portable, checksummed v2
artifacts. The 2B remote model path passed real text and image inference on
Apple MPS. The 8B weight path, CUDA, NCCL, DeepSpeed, FlashAttention, and
trained-model quality remain `NOT RUN`.

### Added

- Backend-neutral `load_encoder`, `Encoder`, and runtime descriptors for
  audited remote models, native MIRA checkpoints, and standalone exports.
- Direct Transformers loading for Qwen3-VL-Embedding with exact revisions,
  verified media, official prompt construction, last-token pooling, MRL
  prefixes, batching, and explicit device and dtype policy.
- The `inference-qwen3-vl-embedding` package extra and an inference-specific `mira doctor`
  workflow with fail-fast dependency checks.
- Streaming `mira.encoded.v2` safetensors artifacts with row-aligned JSONL,
  query or document role binding, per-file checksums, trusted manifest digests,
  atomic publication, and bounded-memory reading.
- `mira init-config` for packaged reference configs and `mira migrate-encoded`
  for explicit v1-to-v2 migration.
- Bounded-memory exact hard-negative mining that preserves deterministic dense
  top-k results and corpus-order tie breaking.

### Changed

- Every public model, config, CLI, registry, and documentation surface now uses
  the full Qwen3-VL-Embedding name. `Qwen3VLModel` appears only as the upstream
  Transformers architecture class, and `qwen3_vl_emb` only as a documented
  MS-SWIFT or v0.1 compatibility identifier.
- `mira encode` now writes v2 artifact directories instead of legacy tensor
  bundles. Mining and evaluation enforce artifact roles and accept trusted
  manifest digests.
- Normal global inference consumes `last_hidden_state` without retaining every
  model layer. Full hidden states are requested only for explicit layer
  representations.
- Exact mining computes scores in query and corpus blocks instead of allocating
  the full query-by-corpus matrix.

### Compatibility

- The public `QwenEmbeddingSemantics` name is replaced by
  `Qwen3VLEmbeddingSemantics`.
- Reference config files now include `qwen3_vl_embedding` in their full names.
- The export reader accepts the exact paired v0.1 labels `last_eos` and
  `qwen3_vl_emb`; new exports always write `last_token` and
  `qwen3_vl_embedding`.
- See [the v0.2 migration guide](docs/migration/v0.2.md) for commands and API
  mappings.

## 0.1.0 - 2026-07-15

First release of MIRA's CPU reference framework for multimodal embedding
post-training. All eleven required pre-tag gates in the release ledger passed
for the CPU reference scope.

### Added

- Strict, versioned experiment configuration with composition, environment
  interpolation, migration, validation, and canonical checksums.
- Versioned multimodal retrieval records with multi-positive supervision,
  item-local media validation, portable dataset fingerprints, and deterministic
  no-duplicate sampling.
- Explicit `GroupLayout` batch semantics for positives, hard negatives,
  in-batch negatives, false-negative masks, relevance weights, tasks, and
  datasets.
- Multi-positive InfoNCE, Matryoshka Representation Learning, scheduled
  objective composition, gradient checks, and embedding-collapse diagnostics.
- Variable-row distributed tensor gathering with detached-remote and autograd
  policies, plus global-query loss scaling verified on two CPU/Gloo ranks.
- Exact retrieval evaluation with Recall, Precision, Hit Rate, MRR, MAP, NDCG,
  per-query evidence, slice reports, and bootstrap confidence intervals.
- Deterministic hard-negative mining with positive-map and output checksums.
- Atomic, checksummed checkpoints with explicit full-resume, model-only, and
  adapter-for-new-stage modes.
- Provenance-bound encoding and export manifests with model, adapter, template,
  tokenizer, processor, representation, normalization, and file identities.
- Nine CLI workflows: `inspect`, `validate-data`, `benchmark`, `doctor`, `mine`,
  `evaluate`, `encode`, `export`, and `train`.
- A dependency-lazy MS-SWIFT 4.4.1 facade and Qwen3-VL-Embedding adapter boundary
  based on audited immutable upstream revisions.

### Evidence boundary

- The release scope is the CPU reference framework. Python 3.11 and 3.12 are
  the declared compatibility targets and passed the pre-tag test gates.
- MS-SWIFT installation is `NOT RUN` because its compatible dependency graph is
  security-blocked, so no backend extra is published.
- Qwen3-VL-Embedding model weights, CUDA, NCCL, DeepSpeed, FlashAttention, multi-node training,
  training-time validation, and model-quality benchmarks remain `NOT RUN`.
- No claim of trained retrieval quality, accelerator performance, or
  Swift-native checkpoint compatibility is part of this release.
