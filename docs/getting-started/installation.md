# Installation

MIRA supports Python 3.11 and 3.12 on Linux and macOS. The secure filesystem
boundary requires POSIX descriptor-relative operations; Windows is not a
supported runtime.

## Choose an environment

The base package covers configuration, data validation, objectives,
checkpoints, mining, and exact evaluation:

```bash
python -m pip install mira-embed
```

Install the direct Qwen3-VL-Embedding inference runtime with:

```bash
python -m pip install "mira-embed[inference-qwen3-vl-embedding]"
mira init-config \
  --preset qwen3-vl-embedding-2b-single-gpu-lora \
  --output mira.yaml
```

Repository contributors should use the lock:

```bash
git clone https://github.com/iwannabewater/mira.git
cd mira
uv sync --frozen --extra inference-qwen3-vl-embedding
```

`inference-qwen3-vl-embedding` installs the exact supported ranges for Transformers,
Accelerate, PEFT, qwen-vl-utils, torchvision, and safetensors. MIRA checks those
ranges before importing model code.

## Verify inference dependencies

```bash
mira doctor \
  --workflow inference \
  --device auto \
  --dtype auto \
  --config mira.yaml
```

The command reports package versions, the selected device and dtype,
FlashAttention requirements, model location, and output-directory access. A
missing optional package returns a nonzero status and names the required range.

You can also verify the tuple from Python:

```bash
python -c \
  'from mira.integrations.qwen3_vl_embedding import require_compatible; print(require_compatible())'
```

## Device policy

The default `device=auto` order is CUDA, MPS, then CPU. The default dtype is:

| Device | Automatic dtype |
| --- | --- |
| CUDA with BF16 support | BF16 |
| Other CUDA | FP16 |
| Apple MPS | FP16 |
| CPU | FP32 |

Use explicit values when reproducibility matters:

```bash
mira encode \
  --device cuda:0 \
  --dtype bfloat16 \
  --help
```

MIRA rejects an unavailable device, invalid CUDA index, or BF16 on MPS. It does
not switch precision after a model-load failure.

## Model cache and network access

Remote loading accepts only the audited model IDs and immutable revisions in
the config. Transformers downloads standard safetensors and processor files
with `trust_remote_code=False`.

For an offline host, populate the Hugging Face cache first, then set the normal
Hugging Face offline environment variables. MIRA still verifies the config's
exact revision. A local path is treated as a MIRA checkpoint or export and
requires its corresponding trusted manifest checksum.

## Training environment

The training adapter is pinned to MS-SWIFT 4.4.1 for source and compatibility
auditing. Its Transformers range conflicts with the direct inference runtime,
and its full dependency graph is not part of the released package extra.

Keep training and direct inference in separate environments. From a repository
checkout with the source-audited training dependencies available, run:

```bash
uv run mira doctor \
  --workflow training \
  --config configs/experiments/qwen3_vl_embedding_2b_single_gpu_lora.yaml
```

The training doctor checks the MS-SWIFT tuple, CUDA, distributed variables,
NCCL when the config requests distributed execution, and FlashAttention when
selected. A schema-valid config is not proof that an accelerator path ran.

## Build and inspect a wheel

```bash
make build
python -m zipfile --list dist/mira_embed-*.whl
```

Install the wheel into a clean virtual environment without resolving a second
dependency graph:

```bash
uv export --frozen --no-dev --no-emit-project --output-file /tmp/mira-runtime.txt
uv venv --python 3.12 /tmp/mira-wheel-smoke
uv pip install \
  --python /tmp/mira-wheel-smoke/bin/python \
  --require-hashes \
  --requirements /tmp/mira-runtime.txt
uv pip install \
  --python /tmp/mira-wheel-smoke/bin/python \
  --no-deps \
  dist/mira_embed-*.whl
/tmp/mira-wheel-smoke/bin/mira --version
/tmp/mira-wheel-smoke/bin/mira --help
```

## Development gates

```bash
make sync-inference
make check
make audit
```

`make check` includes branch coverage. `make audit` checks all registry pins and
exports separate base, development, FAISS, and inference requirement sets for
`pip-audit`.
