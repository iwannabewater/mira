# Multi-positive InfoNCE

Status: stable after the mathematical, gradient, distributed, and integration gates pass.

## Motivation

Retrieval records can have several valid documents. Selecting the first positive discards supervision and can turn another valid positive into a negative. MIRA assigns probability mass to the complete positive set.

## Objective

For normalized query `q_i`, candidate `c_j`, temperature `tau`, positive set `P_i`, and valid denominator `C_i`:

\[
z_{ij} = \frac{q_i^T c_j}{\tau},
\qquad
L_i =
\log \sum_{j \in C_i} e^{z_{ij}}
-
\log \sum_{p \in P_i} w_{ip}e^{z_{ip}}.
\]

Default positive weights are 1. Graded weights must satisfy `0 < w <= 1` and are not normalized by positive count. This keeps the weighted numerator within the unweighted positive mass, so the loss remains non-negative while adding a positive cannot remove existing mass.

The implementation uses masked `logsumexp` in float32. It validates every positive and denominator row before inserting `-inf` masks.

## Inputs and outputs

- Global embeddings have shape `[N, D]`.
- Query and candidate selectors produce `[Q, D]` and `[C, D]`.
- Pair masks and weights have shape `[Q, C]`.
- The result contains a scalar loss, scalar metrics, and diagnostic counts.

Diagnostics include positive mean, negative mean, hardest negative, positive-to-hardest margin, valid and masked pair counts, skipped query count, temperature, norm statistics, and finite-state flags.

## Configuration

```yaml
name: retrieval
type: multi_positive_infonce
weight: 1.0
temperature:
  type: fixed
  value: 0.07
negatives:
  in_batch: true
  cross_device: true
  cross_device_gradient: detached_remote
  explicit_hard_negatives: 4
false_negative_mask:
  enabled: true
  metadata_rules: true
  similarity_margin: 0.05
```

Fixed and learnable temperatures must stay finite and positive. Learnable
temperature configuration also requires `minimum`, `maximum`, and
`learning_rate`. Schema version 1 does not expose symmetric InfoNCE or logit
clipping as configuration fields.

## Suitable uses

Use this objective for global-vector query-to-document retrieval with one or more positives, fixed explicit negatives, compatible in-batch negatives, and optional cross-device candidates.

Do not use it for token-level MaxSim, listwise teacher-score matching, identity classification, or a batch whose tasks disagree on relevance semantics.

## Complexity

The similarity matrix costs `O(QCD)` multiply-adds and `O(QC)` memory. Chunking can reduce peak memory but must preserve the exact log-sum-exp reduction.

## Distributed semantics

Each rank computes local queries against a global candidate pool. `none`, `detached_remote`, and `autograd` specify candidate gradient flow. A global-query mean corrects unequal local query counts. Candidate order and metadata order share one gathered offset plan.

## Symmetric mode

Symmetric InfoNCE averages query-to-document and document-to-query objectives:

\[
L_{sym} = \frac{L_{q \to d} + L_{d \to q}}{2}.
\]

The reverse direction needs an explicit reverse layout. Transposing a mask is insufficient when query and candidate sets, weights, or validity rules differ.

## Common errors

- Reconstructing groups from label positions.
- Selecting only one positive.
- Normalizing positive weights by the positive count.
- Leaving a false negative in the denominator.
- Applying local-rank mean reduction to unequal query counts.
- Treating a detached gather as full autograd.
- Continuing after a zero norm or an all-masked row.

## Verification

The release gate requires hand-computed values, lower loss when positive similarity rises, higher loss when negative similarity rises, negative permutation invariance, non-worsening loss after adding a positive, an explicit all-masked error, a symmetric manual oracle, finite diagnostics, and nonzero gradients.

## References

- [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
- [Qwen3-VL-Embedding technical report](https://arxiv.org/abs/2601.04720)
- [Audited MS-SWIFT loss](https://github.com/modelscope/ms-swift/blob/98a09c18cdf95ff07051324b9b8cc90f5184b24b/swift/loss/embedding.py)
