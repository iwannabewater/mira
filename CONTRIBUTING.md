# Contributing to MIRA

MIRA accepts focused changes with reproducible correctness evidence. Open an issue before changing a public contract, configuration schema, checkpoint format, or distributed collective order.

## Setup

```bash
git clone https://github.com/iwannabewater/mira.git
cd mira
make sync
uv run --no-sync pre-commit install
make check
```

## Change rules

- Put model-specific behavior in an adapter.
- Carry retrieval relationships in `GroupLayout`; do not reconstruct them from labels or row positions.
- Reject invalid input with a MIRA exception that includes record or configuration context.
- Register only implemented capabilities with tests and documentation.
- Preserve `NOT RUN` for checks that need unavailable weights, hardware, credentials, or services.
- Add a schema migration when a persisted configuration or manifest changes.

Every objective change needs a hand-computed oracle, monotonicity checks, a nonzero-gradient check, and failure tests for invalid masks or non-finite tensors. Every distributed change needs a real two-rank CPU test before review. CUDA and NCCL need their own smoke evidence.

## Commit and review

Keep commits scoped and use imperative commit subjects. Pull requests must list the exact verification commands that ran. A green job that skipped the changed path is not verification.

Contributions are licensed under Apache License 2.0. Identify copied or adapted third-party code in the file header and update `NOTICE` when required.
