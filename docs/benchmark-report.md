# Benchmark report

Snapshot date: 2026-07-14.

This report separates correctness verification from performance and
retrieval-quality benchmarks. The release snapshot contains CPU correctness
evidence and one single-shot diagnostic measurement, but no publishable
latency, throughput, memory, or retrieval-quality result.

## Host and dependency tuple

| Surface | Observed value |
| --- | --- |
| Host | Apple M5, arm64 |
| Operating system | macOS 26.5.1, build 25F80 |
| Python | 3.12.13 |
| PyTorch | 2.13.0 |
| NumPy | 2.4.6 |
| Pydantic | 2.12.3 |
| PyYAML | 6.0.3 |
| CUDA available | false |

These values describe the snapshot host only. They are not a supported accelerator tuple.

## Correctness lane

| Check | Command | Status | What it proves |
| --- | --- | --- | --- |
| Two-rank distributed reference | `uv run pytest --no-cov -m distributed` | `PASS` | CPU/Gloo variable-row gather, gradient modes, global-query loss scaling, and mismatch rejection |

The distributed test launches two local workers. It does not measure throughput and does not exercise a Qwen3-VL-Embedding model, CUDA, NCCL, multi-node networking, DeepSpeed, or FlashAttention.

## Performance lanes

| Lane | Status | Published result |
| --- | --- | --- |
| Exact Torch retrieval latency and throughput | `NOT RUN` | none |
| CPU hard-negative mining throughput | `NOT RUN` | none |
| FAISS flat-index parity and speed | `NOT RUN` | none |
| Qwen3-VL-Embedding-2B encode or train | `NOT RUN` | none |
| Qwen3-VL-Embedding-8B encode or train | `NOT RUN` | none |
| CUDA and NCCL scaling | `NOT RUN` | none |
| DeepSpeed ZeRO-2 or ZeRO-3 | `NOT RUN` | none |
| FlashAttention 2 | `NOT RUN` | none |

No throughput, latency, peak-memory, or speedup claim should cite this document.

## Retrieval-quality lanes

| Lane | Status | Missing evidence |
| --- | --- | --- |
| Phase-1 training-time validation | `NOT RUN` | The CLI records validation provenance but does not execute a validation loop |
| Trained MIRA checkpoint retrieval quality | `NOT RUN` | No trained model artifact, versioned dataset, relevance policy, or query-level report |
| Qwen3-VL-Embedding-2B model quality | `NOT RUN` | The backend installation and model-weight lanes did not run |
| Qwen3-VL-Embedding-8B model quality | `NOT RUN` | The backend installation and model-weight lanes did not run |

No Recall, Precision, Hit Rate, MRR, MAP, or NDCG value has been published for
a trained MIRA checkpoint. The evaluator implementations and mathematical
tests do not substitute for the missing evidence above.

The e-commerce config declares desired metrics and MRL dimensions. Those declarations are evaluation inputs, not results.

## Diagnostic microbenchmark

The current `mira benchmark` command creates seeded random CPU tensors and times
one exact-search call. It does not load a dataset or model, perform warmup,
repeat measurements, synchronize an accelerator, or estimate uncertainty. Use
it only as a local diagnostic:

```bash
uv run mira benchmark \
  --config configs/experiments/qwen3_vl_embedding_2b_single_gpu_lora.yaml \
  --queries 64 \
  --candidates 512 \
  --output artifacts/benchmark-diagnostic.json
```

A single invocation ran on the snapshot host with 64 seeded random queries, 512
seeded random candidates, dimension 2,048, and the process default of four
intra-op Torch threads. It reported 0.00258341699372977 seconds and 24,773.3913
queries per second. The result checksum was 33.647491455078125, and the resolved
configuration checksum was
`686051b51932cb8d21b875a01fb60b2a3ea233a69e8f66f777bae54775bd3c84`.
The saved JSON had SHA-256
`32f3bb692533b46670c3a99bc35dcb6d2b2151ae06bc5d5df91837df12e8083f`.

That value is a diagnostic observation, not the exact Torch retrieval latency
and throughput comparison lane above. It has one iteration, no warmup, no
uncertainty estimate, and no committed raw artifact. Its elapsed time and
queries-per-second value are not a publishable benchmark.
A comparison runner must first define the dataset, warmup, measured iterations,
batch size, precision, representation dimension, media preprocessing,
synchronization policy, and target hardware. Preserve its raw output with the
resolved config and artifact checksums.

A report intended for comparison must include:

- immutable model and adapter revisions plus checksums
- dataset manifest, query count, corpus count, and modality distribution
- instruction and template fingerprint
- embedding dimension and normalization rule
- exact or approximate index type and parameters
- hardware, driver, dependency, thread, and process settings
- warmup and measurement protocol
- median and tail latency, throughput, and peak memory
- retrieval metrics with query-level confidence intervals and slice support

Change one of those fields and the result becomes a new benchmark record. Do not merge measurements from different tuples into one headline number.
