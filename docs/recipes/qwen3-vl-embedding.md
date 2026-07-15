# Qwen3-VL-Embedding

MIRA supports the official 2B and 8B embedding checkpoints at audited immutable
revisions:

| Model | Revision | Global dimension |
| --- | --- | ---: |
| `Qwen/Qwen3-VL-Embedding-2B` | `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` | 2048 |
| `Qwen/Qwen3-VL-Embedding-8B` | `2c4565515e0f265c6511776e7193b22c0968ddc7` | 4096 |

The direct inference backend uses the Transformers implementation class
`Qwen3VLModel`. That class name describes the underlying architecture; every
MIRA model ID, capability, error, config, and public document refers to the
embedding model as Qwen3-VL-Embedding.

## Install inference

```bash
uv sync --frozen --extra inference-qwen3-vl-embedding
uv run mira doctor \
  --workflow inference \
  --config configs/experiments/qwen3_vl_embedding_2b_single_gpu_lora.yaml
```

The direct backend owns prompt construction, media verification, last-token
pooling, optional MRL truncation, and L2 normalization. It does not use
MS-SWIFT for inference.

## Encode the official model

```bash
uv run mira encode \
  --checkpoint Qwen/Qwen3-VL-Embedding-2B \
  --config configs/experiments/qwen3_vl_embedding_2b_single_gpu_lora.yaml \
  --input data/documents.jsonl \
  --output artifacts/qwen3-vl-embedding-2b-documents \
  --role document \
  --batch-size 8 \
  --shard-size 4096
```

Remote model loading uses the official preprocessing profile and the revision
from the config. `--checkpoint` cannot name another repository or a mutable
branch.

Use `--dimension 512` or `--dimension 1024` for a trained Matryoshka prefix.
MIRA truncates the raw global vector first and normalizes the prefix afterward.

## Input contract

Text-only item:

```json
{"item_id":"item-1","messages":[{"role":"user","content":"Represent this product description."}]}
```

Mixed-modal item:

```json
{"item_id":"item-2","messages":[{"role":"user","content":"Product image: <image> Product video: <video>"}],"images":[{"uri":"media/item-2.png","checksum":"sha256:<digest>"}],"videos":[{"uri":"media/item-2.mp4","checksum":"sha256:<digest>"}]}
```

Messages alternate `user` and `assistant`, start with `user`, and end with
`user`. Media references follow placeholder order exactly. Qwen3-VL-Embedding
does not declare audio support.

The CLI confines media paths to the JSONL directory. Python callers use
absolute verified paths. Both paths fail on checksum drift and never replace a
failed image or video with placeholder content.

## Load a trained MIRA checkpoint

```python
from mira.inference import load_encoder

encoder = load_encoder(
    "runs/qwen3-vl-embedding-2b/checkpoint-step-1000",
    "configs/experiments/qwen3_vl_embedding_2b_single_gpu_lora.yaml",
    expected_checkpoint_manifest_sha256="<trusted manifest SHA-256>",
)
```

Native checkpoint inference uses the `mira_v1` preprocessing profile so the
prompt matches training. MIRA verifies model name, revision, and config
checksum, reconstructs the configured LoRA, DoRA, or RS-LoRA topology, then
loads model-only state. QLoRA requires a future quantized inference contract
and fails explicitly today.

## Export and reload

Export a native checkpoint:

```bash
uv run mira export \
  --checkpoint runs/qwen3-vl-embedding-2b/checkpoint-step-1000 \
  --config configs/experiments/qwen3_vl_embedding_2b_single_gpu_lora.yaml \
  --output-dir exports/qwen3-vl-embedding-2b \
  --expected-checkpoint-manifest-sha256 "<trusted checkpoint manifest SHA-256>"
```

The command safely merges a configured PEFT adapter, writes safetensors and
processor assets, records `pooling=last_token`, and prints the export manifest
checksum.

Reload the export:

```python
encoder = load_encoder(
    "exports/qwen3-vl-embedding-2b",
    "configs/experiments/qwen3_vl_embedding_2b_single_gpu_lora.yaml",
    expected_export_manifest_checksum="<checksum printed by mira export>",
)
```

Local exports are copied into a checksum-verified private snapshot before
Transformers sees them. Executable formats, remote-code metadata, symlinks,
unexpected files, and checksum drift are rejected.

## Training boundary

The reference config declares:

- multi-positive InfoNCE and false-negative masking
- MRL dimensions 512, 1024, and 2048
- LoRA target selection and component learning rates
- BF16, SDPA, gradient checkpointing, and one-device runtime intent
- checksummed train and validation datasets

Inspect and validate it without loading model weights:

```bash
uv run mira inspect \
  --config configs/experiments/qwen3_vl_embedding_2b_single_gpu_lora.yaml
uv run mira validate-data \
  --config configs/experiments/qwen3_vl_embedding_2b_single_gpu_lora.yaml
```

Training construction remains behind the pinned MS-SWIFT 4.4.1 adapter. Keep
its environment separate from direct inference because the supported
Transformers ranges do not overlap. Run `mira doctor --workflow training`
before any hardware job, and keep unavailable CUDA, NCCL, FlashAttention,
DeepSpeed, or model-size checks recorded as `NOT RUN`.

## Current capability status

| Capability | Status |
| --- | --- |
| Text, image, video, and mixed-modal global inference | Implemented |
| Official 2B model direct runtime | Verified where listed in the release ledger |
| 8B model construction contract | Implemented; hardware evidence is tracked separately |
| MRL global prefixes | Implemented |
| Audio | Unsupported by the model contract |
| Token and hybrid embeddings | Planned |
| Silent truncation or media fallback | Prohibited |
| Remote custom code | Prohibited |
