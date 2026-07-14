# Reranker distillation

Status: planned. Distillation objectives are not in the stable registry.

## Motivation

A cross-encoder or multimodal reranker can express candidate interactions that a global embedding cannot. Distillation transfers its relative preferences into the embedding score while preserving a cheaper retrieval path.

## Listwise objective

For teacher scores `r_j`, student similarities `s_j`, and temperatures `T_t` and `T_s`:

\[
p_t(j|q) = \operatorname{softmax}(r_j/T_t),
\qquad
p_s(j|q) = \operatorname{softmax}(s_j/T_s),
\]

\[
L_{KD} = D_{KL}(p_t || p_s).
\]

The candidate order in teacher scores must match the layout exactly. MIRA will carry a teacher-score presence mask; NaN is not a missing-value marker.

## Inputs and outputs

Inputs are student scores `[Q, C]`, teacher scores `[Q, C]`, validity masks, and optional grades. The objective returns one scalar and per-query entropy, KL, rank agreement, missing-score, and calibration diagnostics.

## Suitable uses

Use listwise distillation when teacher scores are produced for the same versioned candidate set and the teacher has been evaluated on the target relevance policy. Avoid it when scores come from different candidate orders, prompts, model revisions, or incomparable scales without calibration.

## Configuration and complexity

The future schema will record teacher type and revision, prompt checksum, score transform, temperatures, objective weight, missing-score policy, and candidate manifest. Computation is `O(QC)` after embeddings; producing teacher scores is usually the dominant cost.

## Distributed semantics

Teacher scores follow candidate gather order. If scores are computed offline, every rank verifies the same manifest checksum. Loss reduction uses global valid-query count, not rank count.

## Common errors

- Sorting candidates after teacher scoring.
- Mixing logits from teachers with different scales.
- Filling missing scores with zero.
- Distilling a teacher that treats a known positive as irrelevant.
- Saving weights without teacher and candidate manifests.

## Required tests before registration

Tests must compare KL with a manual softmax, reject reordered candidates, verify temperature behavior, mask missing scores, preserve gradients only through the student, and resume the teacher manifest unchanged.
