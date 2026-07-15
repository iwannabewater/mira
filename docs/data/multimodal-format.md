# Multimodal message format

MIRA keeps model-facing messages and media references together in every query and candidate item. The data layer validates structure without importing MS-SWIFT or a model processor.

## Placeholders and media arrays

Use `<image>`, `<video>`, and `<audio>` inside message text. Each placeholder count must equal the length of its matching media array for that item.

```json
{
  "item_id": "query-image-1",
  "messages": [
    {
      "role": "user",
      "content": "Find the product shown in <image>"
    }
  ],
  "images": [
    {
      "uri": "media/query/query-image-1.jpg",
      "checksum": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  ]
}
```

MIRA scans both string content and the `text` fields of structured content parts:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Compare <image> with <image>"},
    {"type": "retrieval_policy", "name": "exact-product"}
  ]
}
```

That message requires exactly two entries in `images`. Placeholder order follows message order and text order. Media references remain in the order supplied by the corresponding array. The schema verifies counts, not the semantic correctness of the pairing, so dataset producers must preserve the intended order.

## Item-local validation

The query, each positive, and each negative validate independently. A valid query cannot hide a broken candidate. For example, this item fails because it has one image reference and no `<image>` placeholder:

```json
{
  "item_id": "broken-positive",
  "messages": [{"role": "user", "content": "red jacket"}],
  "images": ["media/red-jacket.jpg"]
}
```

Errors include the offending item ID and the observed placeholder and reference counts.

## URI and decoding policy

`MediaRef.uri` is a non-empty string at the dependency-light schema boundary.
The MIRA 0.1 CLI supports only paths relative to the JSONL or item file. It
rejects remote and data URIs, absolute paths, missing files, traversal outside
the dataset directory, symbolic links in any path component, and special files
before model loading. It hashes from the same regular-file descriptor it opened.
Immediately before template encoding, MIRA reopens the canonical source without
following links, verifies the recorded digest while copying it once, and passes
only the private temporary copy to MS-SWIFT. Source replacement or in-place
content drift therefore fails before decoder use.

The core reader validates checksum syntax but does not open media. The default
`mira validate-data` path resolves every media file, computes SHA-256, and
compares any declared digest. It does not inspect MIME type, dimensions,
duration, or decoder availability. A production ingestion job should check
those properties before training. Decoder and preprocessing errors must stop
the run with record and path context. MIRA does not replace failed media with
text such as `NULL`.

## Model capability boundary

The general schema includes image, video, and audio so adapters can declare their own capabilities. The audited Qwen3-VL-Embedding adapter declares text, image, video, and mixed-modal global embeddings. It does not declare audio support. Audio records are outside the current Qwen3-VL-Embedding recipe and orchestration must reject them before model loading.

## Instructions

An item may carry an `instruction`, but training and inference must use the same prompt semantics. The Qwen3-VL-Embedding bridge defaults to `Represent the user's input.` and fingerprints every prompt-affecting template setting. A batch whose instruction or template fingerprint differs from the loaded adapter is rejected.

Do not use item instructions as untracked free-form metadata. If instructions vary by task, version the policy and record it with the dataset and export manifests.
