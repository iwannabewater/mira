# Changelog

This project follows Semantic Versioning. Release entries describe user-visible behavior and record any required `NOT RUN` hardware or model checks.

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
- Qwen model weights, CUDA, NCCL, DeepSpeed, FlashAttention, multi-node training,
  training-time validation, and model-quality benchmarks remain `NOT RUN`.
- No claim of trained retrieval quality, accelerator performance, or
  Swift-native checkpoint compatibility is part of this release.
