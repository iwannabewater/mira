# Embedding batch semantics

An embedding batch is a retrieval graph, not a rectangular list of labels. MIRA keeps that graph in `GroupLayout` and sends model inputs beside it.

## Tensor roles

Let one model forward produce `E` with shape `[N, D]`. A layout selects query rows `Q` and candidate rows `C`, then defines pairwise tensors with shape `[|Q|, |C|]`:

| Tensor | Meaning |
| --- | --- |
| `positive_mask` | Candidate pairs that belong in the positive probability mass |
| `denominator_mask` | Pairs allowed in the objective denominator |
| `explicit_negative_mask` | Negatives stored with the query record |
| `in_batch_negative_mask` | Candidates contributed by compatible records in the batch |
| `false_negative_mask` | Related pairs removed from the denominator |
| `positive_weights` | Positive coefficients in `(0, 1]`; non-positive positions are zero |
| `relevance` | Optional graded evaluation labels, with a separate presence mask |
| `teacher_scores` | Optional distillation scores, with a separate presence mask |

Every query has at least one positive. A positive is part of the denominator. Explicit and in-batch negative masks do not overlap positive pairs. A false negative cannot remain in the denominator. Query and candidate indices are unique and in range.

The stable multi-positive objective uses weight 1 for each positive. Graded weights are allowed only when every positive coefficient is in `(0, 1]`; MIRA does not normalize them by the number of positives. Normalization would change positive-set probability mass and could make a newly added positive increase the loss.

## Flattening

The collator may flatten variable-size records into one model batch:

```text
record 0: q0, p00, p01, n00
record 1: q1, p10, n10, n11

model rows: [q0, p00, p01, n00, q1, p10, n10, n11]
query rows: [0, 4]
candidate rows: [1, 2, 3, 5, 6, 7]
```

The loss does not infer these boundaries. Packing, padding-free execution, or a backend-specific token layout may change token tensors without changing `GroupLayout`.

## Compatibility

In-batch negatives require compatible negative semantics. By default, a global batch is homogeneous in dataset, task ID, modality pair, and negative policy. Mixing tasks requires an explicit compatibility declaration during configuration validation.

## Failure behavior

MIRA rejects missing positives, duplicate rows, shape mismatches, a positive removed from the denominator, a false negative left in the denominator, non-finite weights, or a query with no valid denominator. These are user-facing layout errors, not assertions.

## Verification

Unit tests cover every invariant and exact mask construction. Property tests vary the number of queries, positives, negatives, masks, and dimensions. Distributed tests reconstruct global candidate offsets from unequal local candidate counts.
