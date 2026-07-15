# E-commerce retrieval recipe

Status: config and data semantics implemented; Phase-1 CLI execution
unsupported; MS-SWIFT dependency installation security-blocked; 8B accelerator
run `NOT RUN`.

The reference experiment trains a Qwen3-VL-Embedding global representation for product retrieval across image, OCR, mixed-modal, and text replay data. It is an auditable target config, not a measured production preset.

## Reference config

Use:

```text
configs/experiments/qwen3_vl_embedding_ecommerce.yaml
```

Its declared dataset mixture is:

| Dataset | Path | Task | Weight |
| --- | --- | --- | ---: |
| `ecommerce-image-to-product` | `data/train.jsonl` | `image_to_multimodal` | 0.60 |
| `general-replay` | `data/general_replay.jsonl` | `mixed_general` | 0.20 |
| `ocr-and-screenshot` | `data/ocr.jsonl` | `image_text_to_multimodal` | 0.10 |
| `text-retrieval-replay` | `data/text_retrieval.jsonl` | `text_to_text` | 0.10 |

Validation reads `data/validation.jsonl`. Dataset weights must sum to exactly 1 within the schema tolerance.

Every reference source uses `licenses: [NOASSERTION]`. This is an explicit
placeholder for an unresolved dataset license. It does not grant permission to
use, train on, or redistribute the data. Replace it with the applicable SPDX
identifier or a documented `LicenseRef-*` before using a real dataset.

## Product identity and false negatives

Populate trusted identity fields whenever the source can support them:

- `canonical_product_id` for canonical catalog identity
- `spu_id` and `sku_id` for product and variant identity
- `image_hash_group` for known image duplicates
- `positive_group_ids` and `negative_group_ids` for candidate-level relationship groups

The reference sampler excludes duplicate `canonical_product_id` and `image_hash_group` values from a global batch. The layout masks matching product and hash identities before an in-batch candidate reaches the InfoNCE denominator.

Do not populate a stronger identity from a weak heuristic. A wrong canonical ID can hide a true negative just as a missing ID can repel a false negative. Version the identity source and measure mask rates by dataset.

## Validate the experiment inputs

```bash
uv run mira inspect \
  --config configs/experiments/qwen3_vl_embedding_ecommerce.yaml

uv run mira validate-data \
  --config configs/experiments/qwen3_vl_embedding_ecommerce.yaml

mkdir -p runs/doctor-qwen3-vl-embedding-ecommerce
uv run mira doctor \
  --config configs/experiments/qwen3_vl_embedding_ecommerce.yaml \
  --output-dir runs/doctor-qwen3-vl-embedding-ecommerce \
  > runs/doctor-qwen3-vl-embedding-ecommerce/report.json
```

`mira doctor` writes JSON to standard output. `--output-dir` checks
writeability and does not store the report. It exits with status 1 when a
required backend or accelerator check fails, which is expected in the base
environment.

Media validation must happen before training. Record checksum, content type, decode result, dimensions, and any preprocessing policy for each referenced asset. Never replace a failed image or video with text and continue.

## Objective and dimensions

The config combines multi-positive InfoNCE with MRL at 512, 1024, 2048, and 4096 dimensions. Positive weights default to 1 and are not divided by positive count. The candidate pool includes explicit, in-batch, and detached-remote cross-device negatives after false-negative masking.

The runtime declares Qwen3-VL-Embedding-8B, BF16, FlashAttention 2, ZeRO-2, and a train batch size of two per device. None of those accelerator settings ran on the release host.

## Phase-1 CLI boundary

The reference config is an auditable target, not a runnable Phase-1 training
recipe. The current `mira train` CLI rejects it before model loading because it
declares all of these unsupported combinations:

- `deepspeed_zero2` instead of the supported single-process strategy
- padding-free training
- modality-balanced sampling
- a weighted four-source mixture
- similarity-margin false-negative masking

Do not present `mira train --config configs/experiments/qwen3_vl_embedding_ecommerce.yaml`
as an executable smoke command. A future launcher must implement and verify each
declared capability. It must also use an MS-SWIFT dependency graph that passes
security review and record the Qwen3-VL-Embedding, DeepSpeed, CUDA, NCCL, and FlashAttention
tuple. A reduced config should be a separate file with a separate checksum, not
an unrecorded runtime override.

## Evaluation policy

The config requests Recall at 1, 5, 10, and 50, MRR at 10, and NDCG at 10 for every MRL dimension. It also declares category, modality pair, source, and head-tail slices.

Before accepting a checkpoint, define the relevance policy for exact product, variant, substitute, and visually similar results. Report the complete query count and slice support with each metric. A high aggregate score can hide a failed tail or modality slice.

When a verified checkpoint exists, run:

```bash
uv run mira evaluate \
  --checkpoint runs/qwen3-vl-embedding-ecommerce-smoke \
  --config configs/experiments/qwen3_vl_embedding_ecommerce.yaml \
  --queries artifacts/ecommerce-validation-queries.pt \
  --candidates artifacts/ecommerce-validation-candidates.pt \
  --relevance artifacts/ecommerce-validation-relevance.pt \
  --output artifacts/ecommerce-validation-report.json
```

No retrieval-quality result is published for this recipe in the current release.
