# Matryoshka representation learning

Matryoshka representation learning trains nested prefixes of one embedding. A deployment can choose a smaller prefix to reduce storage and search cost without training another encoder.

For dimensions `d_1 < ... < d_K`, MIRA computes

\[
\hat e^{(d_k)} = \operatorname{normalize}(e_{:d_k}),
\qquad
L_{MRL} = \sum_{k=1}^{K} \alpha_k L^{(d_k)}.
\]

Each prefix is sliced and L2-normalized independently. Reusing the full-vector norm changes cosine geometry and is incorrect. Inference and export call the same truncation function used during training.

## Validation

Dimensions must be positive, unique, strictly increasing, and no larger than the adapter's hidden size. Weights must be finite, non-negative, and not all zero. Invalid values fail during configuration loading. MIRA never skips an oversized dimension.

## Inputs and outputs

The input is a normalized or unnormalized global tensor `[N, D]`. Every prefix produces `[N, d_k]`. The objective returns one scalar and per-dimension losses and norm diagnostics.

## Suitable uses

MRL fits global-vector retrieval with several index budgets, progressive rollout, or a coarse-to-fine search pipeline. It does not provide token-level late interaction, compression guarantees for dimensions never trained, or evidence that the smallest prefix meets a business target.

## Distributed semantics

Every rank uses the same dimension and weight map. The candidate gather occurs per dimension or on the full embedding followed by deterministic local slicing, provided gradient semantics remain identical. MIRA tests both the loss and shared-parameter gradient.

## Complexity

Model-forward cost is unchanged. Similarity and loss work scales with `sum(d_k)` and the candidate matrix size. Index storage at inference scales with the selected dimension.

## Common errors

- Truncating without re-normalizing.
- Letting an oversized dimension disappear with a warning.
- Exporting a prefix without recording its dimension and normalization.
- Comparing dimensions with different candidate masks.
- Claiming an untested dimension inherits full-vector quality.

## Verification

Tests check independent unit norms, strict validation, weighted aggregation, nonzero shared gradients, training/export parity, and manual loss values at more than one dimension.

## References

- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- [Official MRL implementation](https://github.com/RAIVNLab/MRL/tree/7ccb42df6be05f3d21d0648aa03099bba46386bf)
