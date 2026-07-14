# Security policy

## Reporting

Do not open a public issue for a vulnerability or leaked credential. Use GitHub's private vulnerability reporting for `iwannabewater/mira`. Include the affected version, reproduction, impact, and any known mitigation. Do not include real model tokens, customer data, or proprietary training samples.

## Supported versions

Until MIRA reaches 1.0, only the latest tagged release receives security fixes. Experimental integrations can change between minor releases and are not a substitute for a production threat assessment.

## Security boundaries

MIRA treats configuration files, checkpoints, datasets, media, model repositories, and backend plugins as untrusted inputs.

- Checkpoint members must be regular, non-symlink files whose identity remains
  stable while open. MIRA copies state into a private temporary file while
  hashing it, validates the manifest and complete file set, and deserializes
  only that verified snapshot with PyTorch's restricted `weights_only` loader.
  Internal checksums detect corruption, not authenticity. Supply a manifest
  SHA-256 from a trusted channel when artifact authenticity matters.
- MIRA 0.1 accepts only local, relative media paths. Every directory component
  and final file is opened through a descriptor-relative, no-symlink walk. MIRA
  computes SHA-256 from that descriptor, binds the resolved reference to the
  digest, and copies the same verified content into a private temporary file
  before MS-SWIFT encoding. A changed source never reaches the upstream decoder.
- Environment interpolation exposes only named variables present in the
  configuration. `extends` and `compose` targets resolve inside the selected
  root configuration's directory tree. Each source is read once from an
  identity-bound regular descriptor; the parser consumes those immutable bytes
  without reopening the path. Traversal, absolute-path escape, symbolic links,
  special files, and mutation during the read are rejected. Validation errors
  do not echo interpolated values. Configuration loading is bounded to 1 MiB per source or override,
  64 container levels, 32 files in a composition chain, and 128 composed-file
  loads per resolution. Manifests must not contain secret values.
- Optional backends run with the user's process privileges. The public
  MS-SWIFT load boundary accepts only the two audited Qwen model IDs, their
  exact immutable revisions and dimensions, and the Hugging Face source. MIRA
  0.1 publishes no MS-SWIFT extra because the compatible 4.4.1 dependency graph
  is security-blocked.
- Export requires model identity and revision, model license, artifact checksums,
  template, representation, normalization, tokenizer, processor, and visual
  preprocessing metadata. Loading a local export also requires its manifest
  checksum from a trusted channel. MIRA opens the declared set through a bound
  directory descriptor, rejects executable, pickle, remote-code, symlink, and
  special-file inputs, copies and hashes each artifact once into a private
  snapshot, and gives only that snapshot to the backend. Adapter identity and
  checksum travel together when an adapter is merged.

These descriptor-relative boundaries require POSIX `dir_fd`, `O_DIRECTORY`,
`O_NOFOLLOW`, and `O_NONBLOCK` support. MIRA 0.1 supports Linux and macOS and
fails closed on Windows and other hosts without those primitives.

Dependency vulnerabilities are checked with `pip-audit`. A backend with an
unresolved advisory remains outside the installable release even when its
source adapter can be audited independently.
