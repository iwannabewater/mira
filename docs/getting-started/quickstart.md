# Quickstart

This walkthrough encodes text with Qwen3-VL-Embedding, inspects the v2
artifact, and computes a similarity matrix. It uses the audited 2B revision
from the reference config.

## 1. Install and check the runtime

```bash
python -m pip install "mira-embed[inference-qwen3-vl-embedding]"
mira init-config \
  --preset qwen3-vl-embedding-2b-single-gpu-lora \
  --output mira.yaml
mira doctor \
  --workflow inference \
  --config mira.yaml
```

## 2. Create query and document JSONL

```bash
mkdir -p examples artifacts
```

Create `examples/queries.jsonl`:

```json
{"item_id":"query-1","messages":[{"role":"user","content":"waterproof red hiking jacket"}]}
{"item_id":"query-2","messages":[{"role":"user","content":"small camera for city travel"}]}
```

Create `examples/documents.jsonl`:

```json
{"item_id":"doc-1","messages":[{"role":"user","content":"Red three-layer shell jacket with sealed seams."}]}
{"item_id":"doc-2","messages":[{"role":"user","content":"Pocket-sized mirrorless camera with a pancake lens."}]}
{"item_id":"doc-3","messages":[{"role":"user","content":"Insulated black winter parka."}]}
```

Every line must be one `MultimodalItem`. IDs must be globally unique inside an
input file. Blank lines and duplicate JSON keys are rejected.

## 3. Encode both axes

```bash
MODEL=Qwen/Qwen3-VL-Embedding-2B
CONFIG=mira.yaml

mira encode \
  --checkpoint "$MODEL" \
  --config "$CONFIG" \
  --input examples/queries.jsonl \
  --output artifacts/queries \
  --role query \
  --dimension 512

mira encode \
  --checkpoint "$MODEL" \
  --config "$CONFIG" \
  --input examples/documents.jsonl \
  --output artifacts/documents \
  --role document \
  --dimension 512
```

Each command prints its item count, throughput, selected device and dtype,
shard count, config checksum, and `manifest_sha256`. Keep both manifest digests
with the artifacts.

The output paths are directories. MIRA refuses to replace an existing path.
Choose a new directory or remove the old one deliberately after inspecting it.

## 4. Load verified artifacts

```python
from mira.inference import load_encoded_artifact

queries = load_encoded_artifact(
    "artifacts/queries",
    expected_manifest_sha256="<query manifest SHA-256>",
    expected_role="query",
)
documents = load_encoded_artifact(
    "artifacts/documents",
    expected_manifest_sha256="<document manifest SHA-256>",
    expected_role="document",
)

scores = queries.embeddings @ documents.embeddings.T
print(scores)
```

The vectors are L2-normalized, so inner product equals cosine similarity.

## 5. Use the Python encoder directly

```python
from mira.inference import load_encoder

MODEL = "Qwen/Qwen3-VL-Embedding-2B"
CONFIG = "mira.yaml"

encoder = load_encoder(MODEL, CONFIG)
vectors = encoder.encode_query(
    ["waterproof red hiking jacket"],
    dimension=512,
)
print(vectors.shape)
```

## Add an image

Place an image below the JSONL directory and compute its digest:

```bash
shasum -a 256 examples/media/jacket.png
```

Then use a relative, confined path:

```json
{"item_id":"doc-image-1","messages":[{"role":"user","content":"<image> red hiking jacket"}],"images":[{"uri":"media/jacket.png","checksum":"sha256:<digest>"}]}
```

The CLI rejects absolute paths, symlinks, path escapes, remote URLs, checksum
drift, and media counts that differ from placeholder counts. It resolves the
path once during validation and verifies it again immediately before vision
preprocessing.

## Next steps

- Read the [inference API](../inference/api.md) for checkpoints, exports,
  multimodal inputs, devices, and MRL.
- Read the [artifact format](../inference/artifacts.md) before moving encoded
  datasets between hosts.
- Use the [hard-negative mining recipe](../recipes/hard-negative-mining.md) for
  retrieval datasets.
- Review [release evidence](../release-status.md) before relying on accelerator
  or training claims.
