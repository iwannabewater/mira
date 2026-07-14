# ADR 0001: Use a thin, versioned MS-SWIFT adapter

Status: Accepted

Date: 2026-07-14

## Context

MIRA needs Qwen3-VL model loading, official prompt construction, multimodal preprocessing, PEFT, distributed launch support, checkpoint integration, and export. MS-SWIFT supplies the model, template, and trainer entry points; the PEFT package supplies adapter construction. MIRA supplies the checkpoint and export contracts. The Swift embedding path also encodes assumptions that conflict with MIRA's domain contract: one positive per record, positional group inference, environment-driven objective settings, detached remote negatives, and warning-based MRL dimension handling.

Forking MS-SWIFT would make MIRA responsible for a large and fast-moving training stack. Importing its embedding trainer directly would make core mathematical behavior depend on those assumptions.

## Decision

MIRA uses an independent domain core and an optional adapter under `mira.integrations.ms_swift`.

The dependency direction is:

```text
MIRA core and domain modules
            ^
MIRA training interfaces
            ^
MS-SWIFT adapter
```

`mira.core`, `mira.objectives`, `mira.data`, `mira.mining`, and `mira.evaluation` do not import `swift`. A static import-boundary check enforces this rule.

The first release accepts MS-SWIFT 4.4.1 only. The adapter checks the installed distribution version before importing backend symbols. A mismatch raises a compatibility error that names the installed version, required version, and audited upstream revision.

MIRA owns these semantics:

- typed configuration and resolved configuration checksums
- records, batch metadata, and `GroupLayout`
- objectives, false-negative masks, and loss scaling
- samplers and data-resume cursors
- evaluation metrics and statistical reports
- MIRA checkpoint serialization, validation, and resume-mode semantics
- MIRA inference export contract and manifests

The adapter facade orchestrates these operations:

- build the model and processor from audited public constructors
- build the official template and preserve its terminal token
- apply one explicit LoRA, RS-LoRA, or DoRA topology through PEFT
- connect a MIRA collator and objective to the backend trainer
- bind MIRA checkpoint save and resume to its single cached model instance
- export that model through the MIRA adapter after any explicit PEFT merge

No monkey patch is allowed in the initial release. A future compatibility patch must live under `mira/integrations/ms_swift/patches`, state an exact upstream revision range, link an upstream issue or explain why no issue applies, include a focused test, and name a removal condition.

## Stability labels

Core contracts become stable only after unit, property, mathematical, distributed, and tiny integration tests pass. The Qwen3-VL-Embedding adapter remains experimental until the required model-weight smoke tests run on suitable CUDA hardware. Registry status is evidence-backed and cannot be upgraded by configuration alone.

## Rejected alternatives

### Maintain an MS-SWIFT fork

A fork gives immediate access to internal behavior but turns every upstream release into a merge and security-maintenance burden. It also makes it easy to scatter MIRA-specific conditions through generic trainer code. The maintenance cost outweighs the short-term convenience.

### Use the Swift CLI as the only boundary

A subprocess boundary is stable for launch automation but cannot pass a structured `GroupLayout`, objective diagnostics, or an autograd-safe candidate gather into the training step. MIRA may use subprocesses for isolated smoke tests, but not as its only training extension point.

### Replace Swift internals with runtime monkey patches

Monkey patches make import order part of correctness and can fail after training has started. MIRA uses explicit adapters and fails during compatibility validation.

## Premise failure and rollback

This decision assumes the audited model, processor, template, and trainer entry points remain callable throughout MS-SWIFT 4.4.1 and that the compatible PEFT API remains available. If an installed build violates that assumption, the adapter fails its startup probe and reports the missing symbol. MIRA's checkpoint manager and export contract do not depend on Swift checkpoint or export constructors. Core training and tests remain usable without backend dependencies.

MIRA 0.1 does not publish the optional dependency because the compatible
MS-SWIFT graph does not pass the security audit. Removing the source adapter
would require deleting its registration and integration package. It would not
change stored MIRA records, objective definitions, evaluation data, or core
checkpoint format. Loading a MIRA checkpoint for a Qwen model would still
require a compatible backend installation to reconstruct that model. The
facade does not claim support for reading Swift-native checkpoint formats.

## Verification

The architecture gate checks that forbidden packages contain no `swift`
imports. Adapter contract tests inject an isolated fake Swift module to verify
argument mapping and failure messages. A future installed-runtime smoke test
must use a clean environment whose complete backend graph passes security audit
before it imports every audited symbol. The installed backend, model-weight,
and CUDA smoke tests retain `NOT RUN` status until their runtime manifests
exist.
