# Retrieval data schema

MIRA stores one `EmbeddingRecord` per JSONL line. Schema version 1 keeps the query, all positives, all explicit negatives, and the metadata that defines their retrieval relationship in one immutable record.

## Record fields

| Field | Type | Requirement |
| --- | --- | --- |
| `schema_version` | integer | Must be `1`; defaults to `1` in the Python model |
| `sample_id` | string | Required and non-empty; the reader enforces per-file uniqueness and `mira validate-data` enforces uniqueness across all configured files |
| `task_id` | string | Required, non-empty |
| `query` | `MultimodalItem` | Required |
| `positives` | array of `CandidateItem` | Required, at least one item |
| `negatives` | array of `CandidateItem` | Optional, defaults to an empty array |
| `metadata` | `RetrievalMetadata` | Required |

Candidate `item_id` values must be unique across `positives + negatives`. The query `item_id` cannot also appear as a candidate.

## Multimodal items

Each query or candidate has this shape:

| Field | Type | Requirement |
| --- | --- | --- |
| `item_id` | string | Required, non-empty |
| `messages` | array of messages | Required, at least one message |
| `images` | array of `MediaRef` | Optional |
| `videos` | array of `MediaRef` | Optional |
| `audios` | array of `MediaRef` | Optional |
| `objects` | annotation object | Optional, processor-independent |
| `instruction` | string or null | Optional item-specific instruction |

A message contains a non-empty `role` and either a non-empty string or a non-empty array of structured content parts. A structured part requires `type`, accepts optional `text`, and retains processor-specific extra keys. MIRA preserves message content, structured text, item instructions, and strings inside opaque processor data exactly as supplied. It does not trim model input.

A media reference can be a URI string or an object:

```json
{
  "uri": "media/images/sku-42.jpg",
  "checksum": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "metadata": {"mime_type": "image/jpeg"}
}
```

The schema requires a non-empty URI. When present, `checksum` must use
`sha256:` followed by 64 lowercase hexadecimal digits. The digest above is a
format example and must be replaced with the digest of the real file.

The dependency-light schema does not open the URI. MIRA 0.1 CLI commands accept
only paths relative to the JSONL or item file. They reject remote and data URIs,
absolute paths, missing files, traversal outside the dataset directory, and
symlink escapes. Media validation computes SHA-256 for every file and compares
it with the declared checksum when one is present.

## Retrieval metadata

`modality_pair` and `dataset_name` are required non-empty strings. The remaining fields are optional unless your sampler, mask policy, slice report, or objective needs them.

| Field | Alignment or meaning |
| --- | --- |
| `query_group_id` | Query identity group |
| `positive_group_ids` | Empty, or one value per positive |
| `negative_group_ids` | Empty, or one value per explicit negative |
| `relevance_labels` | One finite value per item in `positives + negatives` |
| `teacher_scores` | One finite value per item in `positives + negatives` |
| `category_id`, `brand_id`, `spu_id`, `sku_id` | Optional catalog identities and evaluation slices |
| `canonical_product_id` | Canonical product identity used by duplicate and false-negative rules |
| `image_hash_group`, `cluster_id` | Optional duplicate or semantic cluster identities |
| `sampling_weight` | Positive value, defaults to `1.0` |

Positive and negative group ID sets cannot overlap within a record. Missing score arrays use `null`; NaN and infinity are rejected and never mean missing.

## Strictness and immutability

Record models reject unknown fields, except for structured message parts and their processor-specific attributes. Semantic identifiers, message roles, structured part types, and media URIs must not be blank or contain leading or trailing whitespace. Invalid values are rejected instead of trimmed, which prevents normalized identifier collisions and ambiguous paths.

Pydantic freezes validated models. MIRA also detaches and recursively freezes opaque JSON in media metadata, object annotations, and processor-specific message attributes. JSON objects remain read-only mappings in the model, and JSON arrays become tuples. `model_dump(mode="json")`, `model_dump_json()`, and the JSONL writer emit ordinary JSON objects and arrays. A pipeline must create and validate a copied record when it needs a changed value.

Generate the machine-readable schema from the installed version:

```bash
uv run python -c \
  'import json; from mira.data.schema import EmbeddingRecord; print(json.dumps(EmbeddingRecord.model_json_schema(), indent=2))' \
  > embedding-record.schema.json
```

Keep the generated schema with the MIRA version that produced it. Future migrations may change the accepted document while preserving the stored source record for audit.

## Candidate order

Candidate-aligned arrays always follow `positives + negatives`, in that order. `GroupLayout` preserves this relationship explicitly when records become tensors. Sorting or filtering candidates after assigning labels or teacher scores breaks the contract and is rejected when the lengths no longer align, but a length-preserving reorder cannot be detected from values alone. Record the producer and candidate manifest when scores come from another model.
