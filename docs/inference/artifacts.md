# Encoded artifact format

`mira encode` writes a directory with schema `mira.encoded.v2`. The format is
safe to inspect without Python object deserialization and supports datasets
larger than memory during encoding.

## Directory layout

```text
documents/
├── manifest.json
├── embeddings-000000.safetensors
├── records-000000.jsonl
├── embeddings-000001.safetensors
└── records-000001.jsonl
```

Each safetensors file contains one key named `embeddings`. The aligned JSONL
file contains one row per vector:

```json
{"item_id":"doc-1","metadata":{"source":"documents.jsonl"}}
```

The manifest records:

- schema and semantic role (`query` or `document`)
- item count, embedding dimension, dtype, and normalization state
- model, revision, source identity, config checksum, backend, preprocessing
  profile, and preprocessing fingerprint
- ordered shard names, row counts, and SHA-256 checksums

Shard names are canonical and contiguous. Extra files, missing files,
directories, symlinks, special files, unexpected safetensors keys, duplicate
IDs, row-count drift, dtype drift, non-finite values, and false normalization
claims are rejected.

## Write artifacts from Python

Use `EncodedArtifactWriter` when vectors arrive in batches:

```python
from mira.inference import EncodedArtifactWriter

writer = EncodedArtifactWriter(
    "artifacts/documents",
    provenance={
        "schema": "example.v1",
        "model": "Qwen/Qwen3-VL-Embedding-2B",
    },
    role="document",
    normalized=True,
    shard_size=4096,
)

try:
    writer.append(item_ids, embeddings, metadata)
    summary = writer.close()
except Exception:
    writer.abort()
    raise

print(summary.manifest_sha256)
```

`append` enforces global ID uniqueness across batches and one fixed dimension
and dtype. Metadata is copied into canonical JSON form before buffering.

The writer creates a private sibling directory, fsyncs each shard and the
manifest, then publishes with an atomic no-replace directory rename. If
validation or serialization fails, it removes the unpublished directory. A
destination created by another process is never overwritten.

Use `write_encoded_artifact` for an in-memory `EncodedCollection`.

## Load and verify

```python
from mira.inference import load_encoded_artifact

collection = load_encoded_artifact(
    "artifacts/documents",
    expected_manifest_sha256="<digest printed by mira encode>",
    expected_role="document",
)
```

The trusted manifest digest binds the full artifact. The loader opens the root
directory without following symlinks, verifies the exact file set, and
snapshots one declared shard at a time through a descriptor. Before allocating
the final matrix, it completes a full verification pass over every shard,
including checksums, actual tensor shapes, row counts, records, global ID
uniqueness, dtype, dimension, and normalization. It then enforces an 8 GiB
default materialization budget and performs a second verified pass to fill the
matrix without retaining duplicate copies of all shards. Set
`max_materialized_bytes` explicitly only after establishing a trusted memory
budget:

```python
collection = load_encoded_artifact(
    "artifacts/documents",
    expected_manifest_sha256="<trusted digest>",
    expected_role="document",
    max_materialized_bytes=16 * 1024**3,
)
```

An allocation failure is reported as `ContractError` with guidance to use the
streaming reader.

For datasets that should stay shard-oriented, use the context-managed reader:

```python
from mira.inference import EncodedArtifactReader

with EncodedArtifactReader(
    "artifacts/documents",
    expected_manifest_sha256="<trusted digest>",
    expected_role="document",
) as reader:
    for shard in reader.iter_shards():
        consume(shard.item_ids, shard.embeddings, shard.metadata)
```

The reader permits one active iterator, checks global ID uniqueness across
shards, and rechecks the directory file set after the final shard.

The digest is optional in the low-level API because a caller may have another
trusted transport. It should be supplied whenever the artifact crosses a trust
boundary. `mira mine` and `mira evaluate` expose separate trusted digest options
for query and document artifacts.

## CLI role enforcement

Mining requires:

- `--queries` with role `query`
- `--corpus` with role `document`

Evaluation requires:

- `--queries` with role `query`
- `--candidates` with role `document`

The commands fail before retrieval if roles or provenance differ.

## v1 migration

MIRA can still load a legacy v1 `.pt` collection with
`load_encoded_collection` for offline migration. The v1 format has no trusted
manifest, shard checksums, or query/document role. Strict mining and evaluation
therefore require v2 directories.

Re-run `mira encode` for the original JSONL, or migrate in a trusted
environment:

```bash
mira migrate-encoded \
  --input legacy.pt \
  --output artifacts/queries \
  --role query \
  --normalization normalized
```

The command requires the missing role and normalization semantics explicitly.
It never infers either value from a filename.
