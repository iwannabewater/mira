# False negatives

A false negative is a candidate that the current objective would repel even though product or semantic evidence says it is related to the query. Similarity alone cannot identify every case.

MIRA applies filters in this order:

1. Remove exact positives.
2. Apply metadata rules such as canonical product, SKU, SPU, product family, duplicate listing, and image-hash group.
3. Apply an optional positive-aware similarity rule.
4. Apply an optional reranker or judge decision.
5. Preserve graded relevance for evaluation or a graded objective.

Metadata filters run before similarity filters. A candidate masked by metadata stays out of the denominator even if an embedding model gives it a low score.

## Positive-aware threshold

One optional rule accepts a negative only when

\[
s(q, n) < \alpha s(q, p),
\]

for a configured `alpha`. The rule depends on the model and positive used for mining, so both belong in the mining manifest. It does not replace metadata rules.

## Batch representation

`false_negative_mask[q, c]` records the exclusion. The same pair must be absent from `denominator_mask`. MIRA reports the masked count and ratio per batch. A sudden increase can indicate a duplicated dataset, a weak sampler, or a mismatched mining index.

## Failure cases

- Treating all same-category products as negatives for a substitute-product task.
- Filtering by similarity without recording the model revision.
- Applying one query's positive threshold to another query.
- Removing a positive from the denominator.
- Hiding an empty denominator after filtering.

Tests cover metadata precedence, positive preservation, threshold boundaries, empty-denominator errors, and deterministic mask construction.
