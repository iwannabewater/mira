# In-batch negatives

In-batch negatives reuse candidates from other records in the same batch. They increase the candidate pool without another model forward, but only when the records share compatible retrieval semantics.

For query `q_i`, MIRA builds a denominator from its positives, explicit negatives, and compatible candidates contributed by other records:

\[
C_i = P_i \cup N_i^{\text{explicit}} \cup N_i^{\text{batch}}.
\]

The sampler prevents duplicate sample IDs, canonical product IDs, positive groups, and configured image-hash groups within a global batch. The layout builder then removes metadata-known false negatives before the objective sees the matrix.

## When to use them

Use in-batch negatives when records share dataset semantics, task ID, modality pair, and relevance policy. They work well for text-to-text, image-to-image, or a homogeneous image-to-multimodal task.

Do not mix unrelated tasks only to enlarge the matrix. A text-retrieval record and an exact-SKU image record may treat the same candidate differently. Their candidate sets require separate denominators unless configuration declares them compatible.

## Fixed hard negatives

Explicit hard negatives stay attached to their query in `explicit_negative_mask`. MIRA does not duplicate a short list at random to reach a configured count. The sampler either supplies the required number deterministically or rejects the batch with record context.

## Distributed behavior

Cross-device candidates extend `N_i^{batch}`. The gather policy determines gradient flow; it does not change pair semantics. Each rank computes local-query loss against the global candidate pool, then applies global-query reduction.

## Common failures

- A duplicate product appears as another record's negative.
- A task with category-restricted retrieval shares candidates with an unrestricted task.
- Local means are averaged across ranks with unequal query counts.
- Candidate metadata is gathered in a different order from embeddings.

MIRA tests candidate permutation invariance, duplicate rejection, task compatibility, unequal local batch sizes, and global target offsets.
