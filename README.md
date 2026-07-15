# MIRA

MIRA is a strict training, inference, mining, and evaluation framework for
multimodal embedding models. Its direct Qwen3-VL-Embedding-2B path has passed
real text and image inference; the 8B contract is implemented but its weights
remain `NOT RUN`. Retrieval data, checkpoints, and vector artifacts use
explicit, checksummed contracts.

MIRA keeps model-specific loading at the integration boundary. Data semantics,
`GroupLayout`, objectives, distributed loss scaling, checkpoints, encoded
artifacts, and exact retrieval stay under MIRA's control.

## Install

Python 3.11 and 3.12 are supported.

```bash
python -m pip install "mira-embed[inference-qwen3-vl-embedding]"
mira init-config \
  --preset qwen3-vl-embedding-2b-single-gpu-lora \
  --output mira.yaml
```

The `inference-qwen3-vl-embedding` extra pins the audited Transformers runtime. It does not
install MS-SWIFT. Training through the MS-SWIFT boundary uses a separate,
source-audited environment because its dependency range conflicts with the
Transformers inference tuple.

Check the local inference environment before downloading weights:

```bash
mira doctor \
  --workflow inference \
  --config mira.yaml
```

Repository contributors can instead run `make sync-inference` and use the
config under `configs/experiments`.

## Python inference

`load_encoder` accepts an audited Hugging Face model ID, a native MIRA
checkpoint, or a local `mira export` directory. Remote models always use the
immutable revision from the config.

```python
from mira.inference import load_encoder

MODEL = "Qwen/Qwen3-VL-Embedding-2B"
CONFIG = "mira.yaml"

encoder = load_encoder(MODEL, CONFIG, device="auto", dtype="auto")

queries = encoder.encode_query(
    ["red waterproof hiking jacket", "compact camera for travel"],
    dimension=512,
)
documents = encoder.encode_document(
    ["A lightweight shell jacket with sealed seams."],
    dimension=512,
)

print(queries.shape)    # torch.Size([2, 512])
print(documents.shape)  # torch.Size([1, 512])
```

Outputs are batch-major, finite floating-point tensors. Full-width and MRL
vectors are L2-normalized when `normalize=True`. The backend uses official
last-token pooling and rejects silent truncation, unsupported audio, ambiguous
message roles, missing media, checksum drift, and remote code.

Multimodal Python inputs use `MultimodalItem`:

```python
from mira.data import MediaRef, Message, MultimodalItem

item = MultimodalItem(
    item_id="product-42",
    messages=(Message(role="user", content="<image> blue running shoe"),),
    images=(
        MediaRef(
            uri="/absolute/verified/shoe.png",
            checksum="sha256:<64 lowercase hex digits>",
        ),
    ),
)

vector = encoder.encode_document([item])
```

The Python API requires absolute media paths with SHA-256 checksums. The CLI
accepts confined relative media paths and resolves them before inference.

## CLI inference

Create `documents.jsonl`:

```json
{"item_id":"doc-1","messages":[{"role":"user","content":"waterproof trail jacket"}]}
{"item_id":"doc-2","messages":[{"role":"user","content":"compact mirrorless camera"}]}
```

Encode a document corpus:

```bash
mira encode \
  --checkpoint Qwen/Qwen3-VL-Embedding-2B \
  --config mira.yaml \
  --input documents.jsonl \
  --output artifacts/documents \
  --role document \
  --dimension 512
```

`mira encode` validates the complete JSONL before model loading, then streams
batches into a `mira.encoded.v2` directory. Each tensor and record shard has a
SHA-256 checksum. The directory is published atomically and is never replaced.
Save the `manifest_sha256` printed by the command. Mining and evaluation accept
that digest as a trusted input.

Use `--role query` for query artifacts. Query and document roles are checked by
downstream commands; a document artifact cannot be passed as a query artifact.

## Native checkpoints and exports

A native checkpoint requires its trusted manifest digest:

```python
encoder = load_encoder(
    "runs/checkpoint-step-1000",
    CONFIG,
    expected_checkpoint_manifest_sha256="<trusted SHA-256>",
)
```

A standalone export requires the checksum printed by `mira export`:

```python
encoder = load_encoder(
    "exports/qwen3-vl-embedding-2b",
    CONFIG,
    expected_export_manifest_checksum="<trusted manifest checksum>",
)
```

MIRA snapshots local exports through descriptor-bound, symlink-free reads. It
verifies the exact file set and every artifact checksum before Transformers can
load the private snapshot.

## Framework surfaces

| Surface | Current contract |
| --- | --- |
| Inference | Direct Transformers backend; real 2B text and image inference verified, 8B weights `NOT RUN` |
| Encoded artifacts | Streaming safetensors shards, aligned JSONL records, role binding, SHA-256 verification, atomic publication |
| Data | Versioned multi-positive retrieval records and explicit media placeholder alignment |
| Objectives | Multi-positive InfoNCE, Matryoshka representation learning, and false-negative masks |
| Distributed | Autograd-aware candidate gathering and globally correct loss scaling |
| Checkpoints | Atomic native checkpoints, trusted manifests, exact resume modes, all-or-nothing RNG restoration |
| Mining | Deterministic exact hard-negative mining with provenance checks |
| Evaluation | Exact retrieval metrics, MRL dimensions, per-query evidence, and slices |
| Training integration | Optional MS-SWIFT boundary pinned to an audited upstream release |

Token and hybrid inference remain `planned`; the released
Qwen3-VL-Embedding backend exposes global embeddings only. Audio is outside the
model capability contract. CUDA, NCCL, DeepSpeed, and FlashAttention claims
remain tied to recorded hardware evidence.

## Documentation

- [Installation](docs/getting-started/installation.md)
- [Inference API](docs/inference/api.md)
- [Encoded artifact format](docs/inference/artifacts.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [Qwen3-VL-Embedding recipe](docs/recipes/qwen3-vl-embedding.md)
- [Multimodal input format](docs/data/multimodal-format.md)
- [Checkpoint and export compatibility](docs/architecture/compatibility-matrix.md)
- [v0.2 migration](docs/migration/v0.2.md)
- [Release evidence](docs/release-status.md)

## Development

```bash
make sync-inference
make check
make audit
```

`make check` runs formatting, lint, strict Linux and macOS type checks, import
contracts, dead-code checks, tests, and branch coverage. `make audit` checks the
base, development, FAISS, and Qwen3-VL-Embedding inference dependency graphs.

## License

MIRA is licensed under Apache License 2.0. Checkpoint provenance records model
and dataset licenses. Export manifests keep the exact model revision and model
license; they do not grant rights to training data or generated artifacts.
