# DistributedDataParallel

MIRA's verified distributed lane is a two-rank CPU test using PyTorch DistributedDataParallel and the Gloo backend. CUDA and NCCL are `NOT RUN`, and this release does not publish a production `torchrun` training command.

## Gather modes

Cross-device candidate gathering has three explicit gradient modes:

| Mode | Collective behavior | Gradient behavior |
| --- | --- | --- |
| `none` | Returns the local tensor without a collective | Local graph only |
| `detached_remote` | Gathers every rank after row padding | Local rows stay live; remote rows are detached |
| `autograd` | Gathers every rank after row padding | Gradients for each source slice are summed back to its source rank |

The objective config selects the mode:

```yaml
negatives:
  in_batch: true
  cross_device: true
  cross_device_gradient: detached_remote
  explicit_hard_negatives: 4
```

Set `cross_device: false` together with `cross_device_gradient: none` for a local candidate pool. Any other pairing fails schema validation.

## Unequal local batch sizes

`gather_variable` supports different row counts on different ranks. It performs tensor-only collectives in this order:

1. Exchange rank and shape metadata.
2. Verify matching mode, dtype, device type, rank, and trailing dimensions.
3. Pad each leading dimension to the largest local row count.
4. Gather equal-shaped tensors.
5. Remove padding and concatenate rank slices.

The returned `sizes` and prefix `offsets` define every rank's exact slice. Metadata that describes candidates must follow the same order. Python-object collectives are not used.

At least one rank must contribute a row. Trailing dimensions must match. A mode, dtype, device, or operation-order mismatch raises `DistributedError`.

## Global-query loss scaling

DDP averages parameter gradients across ranks. A plain mean of local means is wrong when ranks own different numbers of valid queries. For local loss sum `S_r`, world size `W`, and global valid-query count `N`, MIRA backpropagates:

\[
L_r^{backward} = \frac{W S_r}{N}.
\]

After DDP's gradient average, this equals the gradient of the global query mean. Reporting uses the detached value:

\[
L^{report} = \frac{\sum_r S_r}{N}.
\]

`global_query_mean` accepts either a rank-1 tensor of per-query losses with an optional boolean validity mask, or a scalar local sum with an explicit local count. It rejects non-finite loss on any rank.

## Collective ordering

All ranks must enter collectives in the same order. Reuse one `CollectiveGuard` for the ordered boundaries in a step. The guard exchanges a sequence number and operation digest before the guarded operation. It detects ranks that arrive with different names or order. A rank that never reaches the guard still depends on the process-group timeout.

## Verified command

Run the distributed verifier from the repository root:

```bash
uv run pytest --no-cov -m distributed
```

The test launches two local Gloo workers and covers unequal query and candidate rows, detached-remote gradients, full-autograd gather gradients, global-query DDP scaling, metadata mismatch rejection, and collective-order mismatch rejection.

This test ran successfully on CPU/Gloo for the release snapshot. It does not exercise CUDA kernels, NCCL collectives, multi-node networking, or a Qwen3-VL-Embedding model.

## Production boundary

The config schema accepts `runtime.distributed.strategy: ddp`, but schema acceptance is not launcher evidence. A supported production command requires a recorded CUDA/NCCL or target-backend smoke test, shutdown behavior, checkpoint resume evidence, and parity with the CPU mathematical oracle. Those lanes are `NOT RUN` in the current release status.
