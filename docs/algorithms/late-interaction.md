# Late interaction

Status: planned. Multi-vector adapters and MaxSim are not in the stable registry.

## Motivation

A single vector can miss local OCR, part, and attribute matches. Late interaction keeps token or region vectors and scores each query vector against its best document vector.

## Score

For query vectors `q_i` and document vectors `d_j`:

\[
S(q, d) = \sum_i \max_j q_i^T d_j.
\]

Mean-MaxSim divides by the number of valid query vectors. A hybrid pipeline can combine a global ANN score with a local MaxSim rerank score, but the calibration rule must be explicit.

## Inputs and outputs

Query tensors have shape `[B_q, L_q, D]`, document tensors `[B_d, L_d, D]`, and masks `[B, L]`. Padding positions never enter a maximum and receive no gradient. The score matrix has shape `[B_q, B_d]`.

## Suitable uses

Use late interaction for visually dense documents, OCR-heavy product pages, or fine-grained local evidence after global recall. Avoid it when index storage, memory bandwidth, or tail latency cannot support several vectors per document.

## Configuration and complexity

The future schema will record source hidden layer, projection dimension, vector budget, pruning method, score reduction, chunk size, and global/local calibration. Exact scoring costs `O(L_q L_d D)` per pair and stores `O(L_d D)` per document before compression.

## Distributed semantics

Variable token lengths require explicit masks during gather and scoring. Chunking must preserve the maximum across all document chunks. A partial maximum cannot be averaged across ranks.

## Common errors

- Letting padding participate in MaxSim.
- Normalizing a padded zero vector.
- Summing over a variable query length without documenting length bias.
- Taking a maximum independently per chunk and then averaging maxima.
- Claiming global-vector ANN latency for a multi-vector index.

## Required tests before registration

Tests must match a hand-computed MaxSim score, prove padding has zero gradient, compare chunked and unchunked results, cover empty-token errors, and measure storage and latency against the global baseline.

## Reference

- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction](https://arxiv.org/abs/2004.12832)
