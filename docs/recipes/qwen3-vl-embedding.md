# Qwen3-VL-Embedding recipe

Status: adapter contracts implemented; backend installation security-blocked;
model-weight execution `NOT RUN`.

MIRA targets `Qwen/Qwen3-VL-Embedding-2B` and `Qwen/Qwen3-VL-Embedding-8B` through the audited MS-SWIFT 4.4.1 boundary. The release host did not run either checkpoint, CUDA, NCCL, FlashAttention, or DeepSpeed.

## Representation contract

The adapter follows these fixed semantics:

- default instruction: `Represent the user's input.`
- one global vector from the last valid EOS token
- L2 normalization after optional MRL truncation
- 2,048 dimensions for the 2B model
- 4,096 dimensions for the 8B model
- text, image, video, and mixed-modal inputs
- no declared audio or stable token-vector output

Every encoded row must end in the configured EOS token. Training and inference batches must carry the same instruction and template fingerprint as the loaded adapter. Mismatches raise an error.

## Backend security boundary

MIRA 0.1 does not publish an `ms-swift` extra. MS-SWIFT 4.4.1 requires
`gradio<6`; the compatible dependency resolution retains known advisories in
Gradio, Pillow, and Starlette. Patched versions fall outside the audited
upstream graph, so the release security gate blocks backend installation.
There is no supported Qwen environment command for this release.

The adapter remains dependency-lazy so the CPU framework can be installed and
audited without that graph. Re-audit the resolved dependency set before adding
a future backend extra. A manual MS-SWIFT installation is outside the supported
release boundary and does not change any `NOT RUN` status.

Inspect the single-GPU 2B experiment before downloading weights:

```bash
uv run mira inspect \
  --config configs/experiments/qwen3_vl_2b_single_gpu_lora.yaml

mkdir -p runs/doctor-qwen3-vl-2b
uv run mira doctor \
  --config configs/experiments/qwen3_vl_2b_single_gpu_lora.yaml \
  --output-dir runs/doctor-qwen3-vl-2b \
  > runs/doctor-qwen3-vl-2b/report.json
```

`mira doctor` emits JSON to standard output and exits with status 1 when a
required check fails. `--output-dir` is a writeability probe, not a report
destination. Failure is expected in the base environment because the backend
dependencies and CUDA runtime are absent. Preserve the JSON as failure evidence
rather than relabeling it as a pass.

Preserve the resolved config, dependency versions, model revision, model-file
checksums, and doctor output. The model and dataset licenses are separate from
MIRA's Apache 2.0 license. The bundled config uses `licenses: [NOASSERTION]` for
example datasets. That value records an unresolved license conclusion; it does
not grant permission to use or redistribute data. Replace it with the
applicable SPDX identifier or documented `LicenseRef-*` before using real data.

## Prepare data

The 2B config expects:

```text
data/train.jsonl
data/validation.jsonl
```

Each line must follow the [retrieval schema](../data/schema.md). Use a media placeholder for every referenced image or video. The optional backend must receive already validated records; it must not select only the first positive or replace a failed media item with fallback text.

Validate the files:

```bash
uv run mira validate-data \
  --config configs/experiments/qwen3_vl_2b_single_gpu_lora.yaml
```

The config uses the `all-linear` LoRA selector, then explicit component groups keep the vision tower frozen while training the language model and aligner. It also uses SDPA attention, BF16 precision, a local candidate pool, four explicit hard negatives, and MRL dimensions 512, 1024, and 2048. These are declared experiment settings, not measured recommendations.

## Experimental training command

After a compatible MS-SWIFT dependency graph passes security review and an
accelerator environment is recorded, the intended smoke command is:

```bash
uv run mira train \
  --config configs/experiments/qwen3_vl_2b_single_gpu_lora.yaml \
  --output-dir runs/qwen3-vl-2b-lora-smoke \
  --max-steps 10
```

This command was `NOT RUN` on the release host. It is also blocked by the
MS-SWIFT dependency audit in this release. A successful schema check, import,
mocked model test, or CPU objective test does not change that status.

## Evidence required from the first hardware run

Keep the following artifacts together:

- resolved config and SHA-256 checksum
- exact model repository revision and model-file checksums
- template fingerprint and instruction
- dependency, driver, GPU, dtype, and attention-backend report
- training log with finite loss, norms, and gradient diagnostics
- checkpoint manifest and an exact resume check
- validation metrics at every configured MRL dimension
- exported inference manifest plus reload and encode parity

Use `mira encode`, `mira evaluate`, and `mira export` only with a verified
native MIRA checkpoint. The evaluator consumes encoded query and candidate
bundles plus a versioned relevance tensor; it does not encode raw data itself:

```bash
uv run mira evaluate \
  --checkpoint runs/qwen3-vl-2b-lora-smoke \
  --config configs/experiments/qwen3_vl_2b_single_gpu_lora.yaml \
  --queries artifacts/qwen3-vl-2b-validation-queries.pt \
  --candidates artifacts/qwen3-vl-2b-validation-candidates.pt \
  --relevance artifacts/qwen3-vl-2b-validation-relevance.pt \
  --output artifacts/qwen3-vl-2b-validation-report.json

uv run mira export \
  --checkpoint runs/qwen3-vl-2b-lora-smoke \
  --config configs/experiments/qwen3_vl_2b_single_gpu_lora.yaml \
  --output-dir exports/qwen3-vl-2b-lora-smoke
```

The export command emits `manifest_checksum`. Preserve that value through a
trusted channel. Any later command that uses the export directory as
`--checkpoint` must also pass it as `--expected-export-manifest-checksum`; MIRA
loads only a private snapshot of the exact verified artifact set.

Do not publish a model result until those commands succeed against the actual trained artifact and the resulting manifests have been inspected.
