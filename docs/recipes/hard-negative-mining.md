# Hard-negative mining recipe

Status: deterministic exact inner-product mining implemented on precomputed embeddings.

The reference miner removes known positives, applies optional metadata and positive-aware filters, selects the highest-scoring retained rows, and records a complete manifest. It fails when the filters leave fewer than the requested number of negatives.

## Command-line workflow

Prepare two safe PyTorch bundles. Each bundle contains `item_ids`, an embedding matrix, and optional metadata aligned on the item axis. Prepare a JSON mapping from query ID to known positive corpus IDs.

Run:

```bash
uv run mira mine \
  --config configs/experiments/qwen3_vl_2b_single_gpu_lora.yaml \
  --queries queries.pt \
  --corpus corpus.pt \
  --positives positives.json \
  --output mined.jsonl \
  --manifest mining-manifest.json \
  --top-k 8 \
  --retrieve-k 100 \
  --metadata-key canonical_product_id \
  --metadata-key spu_id \
  --metadata-key sku_id \
  --metadata-key product_family_id \
  --metadata-key image_hash_group \
  --metadata-key duplicate_listing_group \
  --similarity-alpha 0.9 \
  --model-checkpoint checkpoints/qwen3-vl-2b
```

The config supplies model and instruction provenance. The command checksums the
checkpoint and encoded collections. Repeat `--metadata-key` for each identity
field that the run should enforce. `--similarity-alpha` enables the
positive-aware rule. If you omit these flags, the CLI applies neither optional
filter. Use `retrieve-k` larger than `top-k` so known positives and filtered
false negatives do not exhaust the retrieved set.

Do not load untrusted pickle content. Mining bundles must use the command's safe tensor-and-metadata shape, and their item IDs must be unique. Embeddings must be finite floating-point matrices with matching dimensions.

## Python reference

The Python API shows the exact algorithm without a model dependency:

```python
import os
from pathlib import Path

import torch

from mira.mining import (
    EncodedCollection,
    MetadataFalseNegativeFilter,
    MiningPipeline,
    MiningRequest,
    PositiveAwareSimilarityFilter,
)

queries = EncodedCollection(
    item_ids=("q1",),
    embeddings=torch.tensor([[1.0, 0.0]]),
    metadata=({"canonical_product_id": "query-product"},),
)
corpus = EncodedCollection(
    item_ids=("p1", "duplicate", "hard", "easy"),
    embeddings=torch.tensor(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.80, 0.20],
            [0.10, 0.90],
        ]
    ),
    metadata=(
        {"canonical_product_id": "positive"},
        {"canonical_product_id": "query-product"},
        {"canonical_product_id": "hard"},
        {"canonical_product_id": "easy"},
    ),
)

pipeline = MiningPipeline(
    metadata_filter=MetadataFalseNegativeFilter(),
    similarity_filter=PositiveAwareSimilarityFilter(alpha=0.9),
)
result = pipeline.run(
    MiningRequest(
        queries=queries,
        corpus=corpus,
        positives={"q1": ("p1",)},
        top_k=2,
        retrieve_k=4,
        manifest_values={
            "run_id": "mine-2026-07-14-001",
            "model_path": os.environ["MODEL_CHECKPOINT"],
            "model_checksum": os.environ["MODEL_CHECKSUM"],
            "adapter_checksum": os.environ.get("ADAPTER_CHECKSUM"),
            "instruction_template_checksum": os.environ["TEMPLATE_CHECKSUM"],
            "normalization_rule": "l2",
            "random_seed": 17,
            "teacher_version": None,
        },
    )
)

for row in result.negatives:
    print(row.query_id, row.candidate_id, row.score, row.rank, row.difficulty)
result.manifest.write(Path("mining-manifest.json"))
```

Set the environment values from real artifacts. Never substitute placeholder digests in a production manifest.

## Filter order

The reference pipeline applies these rules in order:

1. Remove known positive corpus IDs.
2. Remove candidates that share a configured non-empty identity value.
3. Remove candidates that fail the positive-aware similarity rule, when enabled.
4. Stop after `top_k` retained candidates.

The Python `MetadataFalseNegativeFilter()` default covers canonical product,
SPU, SKU, product family, image hash, and duplicate listing groups. The CLI
enables only the keys passed with `--metadata-key`. The positive-aware rule
retains a candidate only when its score is below
`alpha * best_positive_score`.

The pipeline uses exact Torch inner product and does not claim approximate-index parity or production-scale throughput. It orders by descending score and preserves corpus input order when scores tie, making row ranks, strata, and checksums deterministic. It labels retained thirds as hard, medium, and easy for downstream analysis; the score and rank remain the primary evidence.

## Audit checks

Before using mined rows for training:

- sample retained rows and false-negative removals by task and category
- compare query, corpus, model, template, and normalization checksums
- verify every known positive exists in the corpus
- verify `top_k <= retrieve_k <= corpus size`
- record the retained count before and after every filter
- keep the manifest immutable beside the mined JSONL

A new model, prompt, corpus, dimension, normalization rule, or filter threshold requires a new mining run and manifest.
