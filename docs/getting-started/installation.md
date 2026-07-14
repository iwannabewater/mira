# Installation

MIRA 0.1 supports Linux and macOS with Python 3.11 or 3.12. The locked development environment uses Python 3.12.13. Windows is not supported because root-confined configuration, media, and local-export readers require POSIX descriptor-relative traversal with `O_DIRECTORY` and `O_NOFOLLOW`; unsupported hosts fail closed. The base package installs the CPU-capable reference contracts; FAISS is optional.

## Install from this checkout

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
git clone https://github.com/iwannabewater/mira.git
cd mira
make sync
uv run --no-sync mira --help
```

`uv.lock` is the dependency record. `make sync` installs that lock in two steps and builds MIRA without a fresh isolated build-dependency resolution.

For development, run the complete local gate after installation:

```bash
make check
```

## MS-SWIFT security boundary

MIRA contains a lazy adapter boundary for the source-audited MS-SWIFT 4.4.1
release, but MIRA 0.1 does not publish an installable `ms-swift` extra.
MS-SWIFT 4.4.1 requires `gradio<6`. The compatible dependency resolution
retains known advisories in Gradio, Pillow, and Starlette, while the patched
versions fall outside that upstream dependency graph. The release security
gate therefore blocks installation of this backend.

There is no supported MS-SWIFT installation command for this release. Re-audit
the complete resolved graph before adding a future backend extra. Installing
MS-SWIFT manually does not turn the Qwen or accelerator lanes into supported
paths.

`mira doctor` writes its JSON report to standard output. `--output-dir` checks
that a target directory is writable; it does not store the report. In the base
environment, the command is expected to exit with status 1 because required
backend and accelerator checks fail. You can still preserve that failure
evidence:

```bash
mkdir -p runs/doctor-qwen3-vl-2b
uv run mira doctor \
  --config configs/experiments/qwen3_vl_2b_single_gpu_lora.yaml \
  --output-dir runs/doctor-qwen3-vl-2b \
  > runs/doctor-qwen3-vl-2b/report.json
```

Qwen model weights, CUDA, NCCL, FlashAttention, and DeepSpeed were `NOT RUN` on
the release host.

## Optional FAISS evaluator

Install the optional CPU FAISS evaluator with:

```bash
make sync-faiss
```

FAISS availability depends on the Python version and platform. The exact PyTorch evaluator remains the reference when the optional module is absent.

## Verify a built wheel

Packaging changes require an isolated installation, not an import from the source tree:

```bash
ARTIFACT_DIR="$(mktemp -d)"
make build ARTIFACT_DIR="$ARTIFACT_DIR"
WHEEL="$ARTIFACT_DIR/mira_embed-0.1.0-py3-none-any.whl"
test -f "$WHEEL"
TMP_ENV="$(mktemp -d)/venv"
uv venv "$TMP_ENV" --python 3.12
uv export --frozen --no-dev --no-emit-project --output-file "$ARTIFACT_DIR/runtime.txt"
uv pip install --python "$TMP_ENV/bin/python" --require-hashes --requirements "$ARTIFACT_DIR/runtime.txt"
uv pip install --python "$TMP_ENV/bin/python" --no-deps "$WHEEL"
uv pip check --python "$TMP_ENV/bin/python"
"$TMP_ENV/bin/mira" --help
"$TMP_ENV/bin/python" -c "import mira; print(mira.__version__)"
```

Run this smoke test against the wheel selected for release. Building a wheel alone does not prove that its entry point or package data work after installation.

## Current runtime evidence

The CPU reference path and two-rank Gloo collective test have run on the release host. The MS-SWIFT dependency installation is security-blocked. CUDA, NCCL, DeepSpeed, FlashAttention, and Qwen3-VL-Embedding 2B and 8B model weights are `NOT RUN`. See [release status](../release-status.md) before choosing a production environment.
