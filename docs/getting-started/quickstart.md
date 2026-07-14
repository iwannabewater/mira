# Quickstart

This quickstart validates a small retrieval dataset and its resolved experiment config. It stops before downloading model weights or starting Qwen training.

## 1. Install the project

From the repository root:

```bash
make sync
```

This installs the CPU reference framework. MIRA 0.1 does not publish an
`ms-swift` extra because the dependency graph required by MS-SWIFT 4.4.1 does
not pass the release security audit. The Qwen and accelerator paths therefore
remain `NOT RUN`.

## 2. Create a valid JSONL record

Create the local data directory first:

```bash
mkdir -p data
```

Then create `data/train.jsonl` and place the following JSON object on one line:

```json
{"schema_version":1,"sample_id":"quickstart-1","task_id":"multimodal_retrieval","query":{"item_id":"query-1","messages":[{"role":"user","content":"red waterproof hiking jacket"}]},"positives":[{"item_id":"product-1","messages":[{"role":"user","content":"red waterproof shell jacket"}]}],"negatives":[{"item_id":"product-2","messages":[{"role":"user","content":"blue cotton shirt"}]},{"item_id":"product-3","messages":[{"role":"user","content":"black running shoe"}]},{"item_id":"product-4","messages":[{"role":"user","content":"green wool hat"}]},{"item_id":"product-5","messages":[{"role":"user","content":"yellow travel bag"}]}],"metadata":{"modality_pair":"text_to_text","dataset_name":"retrieval-train","canonical_product_id":"product-1","relevance_labels":[1.0,0.0,0.0,0.0,0.0]}}
```

The bundled 2B config also names a validation file. Create a second record with
a distinct `sample_id` and dataset name:

```bash
sed \
  -e 's/"quickstart-1"/"quickstart-validation-1"/' \
  -e 's/"retrieval-train"/"retrieval-validation"/' \
  data/train.jsonl > data/validation.jsonl
```

The content is still duplicated, so this pair is suitable only for checking
contracts. It is not a meaningful evaluation split. MIRA requires
`sample_id` values to be unique across every configured training and
validation file.

The example config records `licenses: [NOASSERTION]` for both datasets.
`NOASSERTION` means that the example does not establish the dataset license. It
does not grant permission to use or redistribute the data. Replace it with the
applicable SPDX identifier or a documented `LicenseRef-*` value before using a
real dataset.

## 3. Resolve the config

```bash
uv run mira inspect \
  --config configs/experiments/qwen3_vl_2b_single_gpu_lora.yaml
```

Inspection loads the strict schema and catches unknown fields, invalid objective combinations, dataset weight errors, unsupported model capabilities, and oversized MRL dimensions before training.

## 4. Validate the data

```bash
uv run mira validate-data \
  --config configs/experiments/qwen3_vl_2b_single_gpu_lora.yaml
```

The JSONL reader rejects blank lines, malformed JSON, duplicate `sample_id` values within a file, missing positives, duplicate candidates, candidate alignment errors, and mismatched media placeholders.

You can exercise the same record contract from Python:

```python
from mira.data.readers import read_jsonl

records = read_jsonl("data/train.jsonl")
record = records[0]
print(record.sample_id, len(record.positives), len(record.negatives))
```

## 5. Treat Qwen execution as experimental

The intended single-GPU command is:

```bash
uv run mira train \
  --config configs/experiments/qwen3_vl_2b_single_gpu_lora.yaml \
  --output-dir runs/qwen3-vl-2b-quickstart \
  --max-steps 10
```

This command requires MS-SWIFT, Qwen model weights, and a compatible
accelerator environment. There is no supported MS-SWIFT installation path in
MIRA 0.1 because the MS-SWIFT 4.4.1 dependency graph is security-blocked. The
command was also `NOT RUN` on the current release host, which has no CUDA
device. Do not treat successful config or data validation as training evidence.

Read the [Qwen recipe](../recipes/qwen3-vl-embedding.md) before a hardware run. Use [release status](../release-status.md) to distinguish verified CPU behavior from unexecuted accelerator paths.
