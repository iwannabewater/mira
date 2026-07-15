# Inference API

MIRA exposes one backend-neutral `Encoder` API and one strict direct backend for
Qwen3-VL-Embedding. The backend loads the audited model revision directly with
Transformers and `trust_remote_code=False`.

## Install the runtime

```bash
python -m pip install "mira-embed[inference-qwen3-vl-embedding]"
mira init-config --output mira.yaml
```

The supported dependency window is intentionally narrow:

| Package | Supported range |
| --- | --- |
| Transformers | `>=5.13.1,<5.14` |
| Accelerate | `>=1.14,<1.15` |
| PEFT | `>=0.19.1,<0.20` |
| qwen-vl-utils | `>=0.0.14,<0.0.15` |
| torchvision | `>=0.28,<0.29` |
| safetensors | `>=0.8,<0.9` |

`mira doctor --workflow inference` reports every installed version and fails
before model loading when the tuple is incomplete.

## Load an encoder

```python
from mira.inference import load_encoder

encoder = load_encoder(
    checkpoint="Qwen/Qwen3-VL-Embedding-2B",
    config="mira.yaml",
    backend="qwen3-vl-embedding-transformers",
    device="auto",
    dtype="auto",
    batch_size=8,
)
```

The config fixes the model ID, immutable revision, native embedding dimension,
attention implementation, image budget, and instruction. A remote checkpoint
argument must exactly match `model.name_or_path`.

### Source types

| Source | Required arguments | Preprocessing profile |
| --- | --- | --- |
| Audited Hugging Face model | Exact model ID | `official` |
| Native MIRA checkpoint | Checkpoint directory and optional trusted checkpoint manifest SHA-256 | `mira_v1` |
| Local MIRA export | Export directory and required trusted export manifest checksum | `mira_v1` |

Native checkpoint loading reconstructs the configured PEFT topology before
loading model state. QLoRA fails explicitly because the released inference API
does not define a quantized load contract. It does not degrade to ordinary
LoRA.

Local exports are treated as untrusted input. MIRA opens each path component
without following symlinks, copies the manifest and its exact declared file set
into a private directory, verifies SHA-256 checksums, rejects executable model
formats and remote-code metadata, then gives the snapshot to Transformers.

## Encode text

```python
query_vectors = encoder.encode_query(
    ["red trail shoe", "waterproof jacket"],
    batch_size=2,
    dimension=512,
    normalize=True,
)

document_vectors = encoder.encode_document(
    ["A lightweight red shoe with a rock plate."],
    dimension=512,
)
```

Both methods return a `torch.Tensor` with shape `[N, D]`. A single string still
returns `[1, D]`. Input order is preserved across backend batches.

The default query and document instruction is:

```text
Represent the user's input.
```

You can pass an explicit `instruction=` to a Python encode call. The CLI keeps
the configured instruction fixed so its artifact provenance remains exact.

## Encode multimodal items

```python
from mira.data import MediaRef, Message, MultimodalItem

item = MultimodalItem(
    item_id="catalog-17",
    messages=(
        Message(
            role="user",
            content="Front view: <image> Side view: <image>",
        ),
    ),
    images=(
        MediaRef(
            uri="/data/catalog-17-front.png",
            checksum="sha256:<front digest>",
        ),
        MediaRef(
            uri="/data/catalog-17-side.png",
            checksum="sha256:<side digest>",
        ),
    ),
)

embedding = encoder.encode_document([item])
```

Placeholders and media references are positional and must align exactly.
Python callers provide absolute, regular, symlink-free paths and tagged
SHA-256 values. MIRA snapshots each file and verifies the digest immediately
before qwen-vl-utils processes it.

Supported inputs:

- text strings
- images through `<image>` and `MediaRef`
- videos through `<video>` and `MediaRef`
- mixed text, image, and video items
- alternating `user` and `assistant` history that starts and ends with `user`

Audio, remote media URLs, data URLs, opaque structured message parts, and
implicit media ordering are rejected.

## Pooling and normalization

The processor terminates every input with `<|endoftext|>`. MIRA resolves that
token directly and verifies the last valid token in every row. It does not use
the tokenizer's `eos_token_id`, which is a different token for the audited
Qwen3-VL-Embedding processor.

The adapter consumes `last_hidden_state` from the underlying `Qwen3VLModel`,
selects the last valid token through the attention mask, and returns the raw
vector to `Encoder`. It requests all layer outputs only when an explicit hidden
layer representation asks for them. `Encoder` then applies optional MRL prefix
truncation and L2 normalization.

This sequence is fixed:

1. preprocess without truncation;
2. reject an input longer than the configured maximum;
3. run the model and pool the last valid token;
4. truncate to the requested MRL prefix, when supplied;
5. L2-normalize the returned vector.

## Devices and dtypes

`device="auto"` selects CUDA, then MPS, then CPU. `dtype="auto"` selects BF16
on compatible CUDA, FP16 on other CUDA and MPS devices, and FP32 on CPU.

Explicit values fail if the device or dtype is unavailable. MPS rejects BF16.
CPU and MPS device strings cannot include an index. A CUDA index must exist.
MIRA does not retry a failed model load with a different device, dtype, or
attention backend.

## Capability boundary

The released Qwen3-VL-Embedding backend implements global embeddings. Calls to
`encode_tokens` or hybrid representation requests raise a capability error.
Those representations remain `planned` until their source layer, projection,
normalization, masking, storage, and scoring contracts are implemented and
tested.
