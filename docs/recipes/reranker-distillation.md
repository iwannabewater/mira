# Reranker distillation preparation

Status: planned. MIRA 0.1 does not register or train a distillation objective.

The current data schema can preserve candidate-aligned teacher scores so a dataset can be audited before the objective exists. Do not add a distillation objective type to a config: the strict schema rejects it, and `mira train` does not consume `teacher_scores` as a distillation loss.

## Prepare aligned scores

For each record, score candidates in the exact order `positives + negatives`. Store one finite number per candidate:

```json
{
  "sample_id": "distill-1",
  "task_id": "text_to_product",
  "query": {
    "item_id": "q1",
    "messages": [{"role": "user", "content": "red waterproof jacket"}]
  },
  "positives": [
    {
      "item_id": "p1",
      "messages": [{"role": "user", "content": "red shell jacket"}]
    },
    {
      "item_id": "p2",
      "messages": [{"role": "user", "content": "crimson rain jacket"}]
    }
  ],
  "negatives": [
    {
      "item_id": "n1",
      "messages": [{"role": "user", "content": "blue cotton shirt"}]
    }
  ],
  "metadata": {
    "modality_pair": "text_to_text",
    "dataset_name": "teacher-v1",
    "relevance_labels": [1.0, 1.0, 0.0],
    "teacher_scores": [8.7, 7.9, -1.4]
  }
}
```

The schema rejects missing entries, extra entries, NaN, and infinity. It cannot detect a length-preserving reorder, so the score producer must record candidate IDs and order in its own manifest.

## Teacher manifest

Record at least:

- teacher model identifier, immutable revision, and artifact checksum
- tokenizer or processor revision
- instruction and prompt-template checksum
- score meaning, calibration, and temperature
- candidate-set checksum and exact candidate order
- preprocessing code revision and dependency tuple
- generation date and random seed, when sampling is involved

Do not combine scores from teachers with different scales unless the calibration transform is explicit and versioned. Missing scores should use a presence mask in a future objective, not zero or NaN.

## Validate the prepared dataset

Use an experiment config that names the scored JSONL, then run:

```bash
uv run mira validate-data --config path/to/experiment.yaml
```

This proves structural alignment only. It does not evaluate teacher quality, calibration, prompt consistency, or suitability for the retrieval relevance policy.

## Activation gate

Distillation can move from `planned` only after MIRA has:

- a typed objective config and registered implementation
- a presence mask with an explicit missing-score policy
- a manual KL and temperature oracle
- tests that preserve gradients through the student only
- reordered-candidate rejection through a manifest checksum
- global valid-query reduction in distributed runs
- checkpoint and resume coverage for teacher provenance
- retrieval and calibration comparisons against a non-distilled baseline

The proposed listwise formula and test contract are documented in [Reranker distillation](../algorithms/distillation.md). Until those gates pass, use teacher scores for offline analysis only.
