# Data validation

Validate configuration, records, and runtime prerequisites before allocating model weights. Each layer catches a different class of failure.

## Command-line validation

Run the config inspection first:

```bash
uv run mira inspect --config configs/experiments/qwen3_vl_2b_single_gpu_lora.yaml
```

Then validate the JSONL paths named by the config:

```bash
uv run mira validate-data --config configs/experiments/qwen3_vl_2b_single_gpu_lora.yaml
```

Keep both outputs with the run metadata. A clean schema result is evidence for the files that were read at that time, not for a later file with the same path.

## What the record reader checks

For each JSONL file, MIRA checks:

- the file is readable UTF-8 JSONL with no blank lines
- every line is one JSON object
- `schema_version` is supported and unknown fields are rejected
- `sample_id` values are unique within that file
- every record has a query and at least one positive
- query and candidate identities do not conflict
- group IDs and candidate-aligned score arrays have the required lengths
- labels and teacher scores are finite
- every media placeholder count matches its item-local reference array

Errors use `path:line: detail`, which makes a failing source record reproducible.

After reading all configured files, `mira validate-data` also requires
`sample_id` values to be unique across training and validation sources. With
the default `--check-media` setting, it accepts only local relative media paths,
rejects path escapes, symbolic links, and special files through a
descriptor-relative walk, computes SHA-256 from the opened descriptor, and
compares any declared checksum. Model-facing encoding rechecks that digest
while copying the content into a private temporary snapshot. `--skip-media`
skips the validation-stage file checks explicitly; it does not weaken the
model-facing snapshot boundary.

The reader alone does not open media or compute checksums. Neither the reader
nor `mira validate-data` inspects model tokens, decodes media, or proves that
relevance labels are correct. Those checks require a dataset-specific ingestion
stage and, for tokenization, the selected model processor.

## Python validation

Use the streaming reader when a dataset may not fit in memory:

```python
from mira.data.readers import iter_jsonl

count = 0
for record in iter_jsonl("data/train.jsonl"):
    count += 1
print(f"validated {count} records")
```

Build the tensor relationship separately:

```python
from mira.data.layout import build_group_layout
from mira.data.readers import read_jsonl

records = read_jsonl("data/train.jsonl")
result = build_group_layout(records, include_in_batch_negatives=True)
print(result.layout.positive_mask.shape)
print(len(result.flattened_items))
```

`GroupLayout` is the only source of positive, explicit-negative, in-batch-negative, and false-negative semantics. The builder never infers groups from labels or flattened positions.

## Batch and sampler checks

The no-duplicate sampler checks declared identity keys across each global batch and saves its epoch, cursor, seed, record fingerprint, and batch settings. Restoring incompatible state raises an error instead of restarting from a new order.

False-negative masks compare candidate identity and selected metadata fields before the objective builds its denominator. If relevance or teacher scores are present for one record in a batch, they must be present for every record in that batch.

## Runtime diagnostics

`mira doctor` emits its JSON report to standard output and uses `--output-dir`
only as a writeability check. Preserve the expected failure report from the
base environment:

```bash
mkdir -p runs/doctor-qwen3-vl-2b
uv run mira doctor \
  --config configs/experiments/qwen3_vl_2b_single_gpu_lora.yaml \
  --output-dir runs/doctor-qwen3-vl-2b \
  > runs/doctor-qwen3-vl-2b/report.json
```

The command exits with status 1 when a required check fails. MIRA 0.1 ships no
MS-SWIFT extra because its compatible dependency graph is security-blocked.
Dependency presence would not be model execution evidence in any case. CUDA,
NCCL, DeepSpeed, FlashAttention, and Qwen model-weight paths remain `NOT RUN`
until their own smoke logs and manifests exist.

## Failure policy

Do not skip an invalid row, select the first positive, duplicate a hard negative, substitute failed media, or shrink an oversized MRL dimension. Correct the source or the configuration, then rerun validation and preserve the new evidence.
