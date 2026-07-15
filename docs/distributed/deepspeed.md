# DeepSpeed

Status: experimental and `NOT RUN`.

MIRA schema version 1 recognizes `deepspeed_zero2` and `deepspeed_zero3` so experiment intent can be validated and recorded. The current release does not declare a DeepSpeed dependency, publish a supported launcher command, or claim that sharded Qwen3-VL-Embedding training and resume work.

## Configuration surface

The e-commerce reference config declares ZeRO-2:

```yaml
runtime:
  precision: bf16
  attention: flash_attention_2
  distributed:
    strategy: deepspeed_zero2
```

This is a target experiment definition. It is not evidence that DeepSpeed, BF16, FlashAttention, or the Qwen3-VL-Embedding-8B weights ran on the release host.

The retrieval objective controls candidate gathering independently:

```yaml
negatives:
  cross_device: true
  cross_device_gradient: detached_remote
```

Changing the sharding strategy must not change positive masks, candidate order, gather gradient semantics, or the global-query reduction.

## Integration boundary

MIRA owns the records, `GroupLayout`, objectives, loss scaling, and checkpoint contract. The dependency-lazy MS-SWIFT bridge may construct an upstream trainer in a future audited environment while retaining MIRA's collator and loss. It does not make untested DeepSpeed behavior stable.

The reference `CheckpointManager` saves an unsharded model plus optimizer, scheduler, scaler, RNG, sampler, dataset, gradient-cache, objective, and curriculum state when supplied. That implementation is not proof that a ZeRO-sharded checkpoint can be saved, consolidated, or resumed. A DeepSpeed run must use an explicit sharded lifecycle and verify every required component before `resume_full` is allowed.

## Required environment evidence

A DeepSpeed lane needs all of the following before it can move out of `NOT RUN`:

- a Linux host and compatible CUDA, driver, PyTorch, DeepSpeed, and compiler tuple
- an MS-SWIFT integration environment whose complete dependency graph passes security audit
- rank-local device and process-group initialization logs
- ZeRO-2 or ZeRO-3 forward, backward, optimizer, save, and exact-resume evidence
- unequal local-query scaling parity with the CPU oracle
- an export or consolidation check that reloads the produced artifact
- explicit peak-memory, throughput, and numerical-comparison records

MIRA 0.1 cannot satisfy the backend requirement because the compatible
MS-SWIFT 4.4.1 dependency graph is security-blocked. Do not add DeepSpeed or
MS-SWIFT ad hoc to the locked release environment and report the result as
reproducible. Record a re-audited dependency tuple in an environment lock or
container manifest, then attach the runtime log and smoke manifest.

## Current limitations

DeepSpeed ZeRO-2 and ZeRO-3, CUDA, NCCL, FlashAttention, Qwen3-VL-Embedding-2B, and Qwen3-VL-Embedding-8B are `NOT RUN`. The verified two-rank evidence uses CPU/Gloo only. See [DDP](ddp.md) for the tested collective semantics and [release status](../release-status.md) for the evidence ledger.
