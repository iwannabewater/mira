# Identity learning

Status: planned. Identity objectives are not in the stable registry.

## Motivation

Instance retrieval may need exact SKU or canonical-product separation beyond semantic similarity. An identity head adds supervised angular or prototype structure while the retrieval objective keeps cross-modal neighborhoods useful.

## Candidate objective

For a normalized feature `x`, normalized class weight `W_y`, scale `s`, and angular margin `m`, ArcFace uses

\[
L_{arc} = -\log
\frac{e^{s\cos(\theta_y + m)}}
{e^{s\cos(\theta_y + m)} + \sum_{j \ne y} e^{s\cos\theta_j}}.
\]

CosFace replaces the target logit with `s(cos(theta_y) - m)`. A joint run would use `L = L_retrieval + lambda_id L_id`.

## Proposed inputs and outputs

Inputs are `[B, D]` embeddings, trusted identity IDs `[B]`, an identity-confidence mask `[B]`, and prototype state `[K, D]` or a sampled subset. The result is one scalar plus active-sample, class-coverage, margin, and prototype diagnostics.

Unknown or low-confidence products must not be forced into an existing prototype. They contribute to retrieval loss only.

## Suitable uses

Use identity learning for exact-instance or canonical-product retrieval with stable IDs and enough examples per identity. Avoid it for substitute-product retrieval, noisy listing IDs, open-set categories without an unknown policy, or datasets where the class table cannot be versioned.

## Configuration and complexity

The future schema will identify ArcFace, CosFace, proxy classification, sampled softmax, or PartialFC; scale and margin; confidence policy; class-table manifest; and objective weight. Full classification costs `O(BKD)` and stores `O(KD)`. PartialFC reduces the active class set but adds distributed sampling and checkpoint state.

## Distributed semantics

All ranks need consistent class IDs, prototype ownership, sampling seeds, and reductions. PartialFC must document which rank owns each prototype and how sampled logits contribute to the global mean.

## Common errors

- Using a business listing ID as a clean product identity.
- Updating prototypes for unknown identities.
- Saving model weights without the class map.
- Comparing an angular-margin implementation only against decreasing training loss.
- Combining per-rank sampled classes without a global target map.

## Required tests before registration

Tests must match ArcFace and CosFace formulas by hand, verify gradients for features and prototypes, exclude unknown IDs, resume prototype and sampler state, and cover multi-rank class ownership.

## Reference

- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
