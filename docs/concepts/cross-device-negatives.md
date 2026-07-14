# Cross-device negatives

MIRA exposes three gradient policies for cross-device candidates.

| Mode | Forward candidate pool | Backward behavior |
| --- | --- | --- |
| `none` | Local candidates | Local gradients only |
| `detached_remote` | All ranks | Local slice retains gradients; remote slices are constants |
| `autograd` | All ranks | Gather backward accumulates gradients into the source rank |

The mode is a typed configuration field and appears in the run manifest. Changing it changes the objective, so resume validation treats a mismatch as incompatible.

## Unequal candidate counts

Tensor collectives need equal shapes. MIRA gathers each local count, pads candidates to the largest count, gathers padded tensors, unpads each rank, and computes prefix-sum offsets. Embeddings use tensor collectives. Small metadata may use an object collective when no tensor encoding exists, but it must preserve the same rank and row order.

## Autograd semantics

MIRA uses a small autograd function around tensor `all_gather`. The forward gathers equal-shaped padded tensors and removes padding after the collective. The backward stacks rank contributions, performs a tensor `all_reduce`, and returns the slice owned by the source rank. A real two-rank gradient oracle compares this path with a single-process reference across the supported PyTorch range.

PyTorch documents different backend implementations for the backward collective. Passing on CPU/Gloo proves only CPU/Gloo behavior. CUDA/NCCL remains `NOT RUN` until a separate smoke test records it.

## Global-query mean

With unequal local query counts, each rank computes a local loss sum. DistributedDataParallel later averages parameter gradients, so the differentiable local scalar is:

\[
L_r^{backward} = \frac{W \sum_{i \in r} \ell_i}{\sum_{k=1}^{W} Q_k},
\]

where `W` is world size and `Q_k` is the query count on rank `k`. A detached all-reduce produces the reported global mean. Averaging local means assigns the wrong weight when `Q_k` differs.

## Collective safety

All ranks validate dtype, feature dimension, mode, and collective plan before the first embedding gather. A rank may contribute zero local rows when at least one other rank contributes data; a globally empty gather is rejected. A mismatch raises a distributed error with rank and expected shape rather than waiting for a timeout.

## References

- [PyTorch distributed documentation](https://docs.pytorch.org/docs/2.9/distributed.html)
- [PyTorch autograd collective implementation](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/distributed/nn/functional.py)
