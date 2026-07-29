# Stage 8A independent cross-platform replay status

## Execution record

- Source commit: `42202b13a7f54a25e67d50d62649ff2b19600a67`
- GitHub Actions run: `30359998043`
- Independent host: Ubuntu 24.04 GitHub-hosted runner, Linux x86-64
- Portable verification image base: `python:3.12.7-slim-bookworm`
- Base manifest-list digest:
  `sha256:60d9996b6a8a3689d36db740b49f4327be3be09a21122bd02fb8895abb38b50d`
- Built image ID:
  `sha256:483dc2e82072826e651262cde68f1ee0ff6d9de302803ceba624b856bf11fbc2`
- Built image archive SHA-256:
  `d0aec8f908119592bfe285c7fa078534bdf28ed1ff65a313a528c5077f22f1b7`
- Evidence artifact ID: `8688496689`
- Evidence artifact SHA-256:
  `2ad8c13b7382c6789f009f7bee595d96b71209feac122e45f3a80011c8116587`

The complete downloaded artifact and its extracted contents are preserved under
`results/independent_exact_lock_seed11/`.

## Outcome

The container build and all 36 tests passed. The seed-11 regenerated-data gate
also passed:

- the frozen dataset hash was preserved;
- schema, row order, labels, and all 49 model-visible features matched exactly;
- only the two registered audit-only floating columns differed, by at most
  `7.216449660063518e-16`, below the fixed absolute tolerance of `2e-15`.

The numerical replay failed its unchanged `1e-12` result gate. QFC, generic
kernel, sample-index, and tensor-train results reproduced exactly (tensor train
to `5.55e-17`). The learned MLP pathway differed by up to
`0.011011011011010985` balanced accuracy.

## First-divergence diagnosis

The MLP initialization hash and the complete 56-epoch sample-order hash match
the registered values exactly. The first detected difference is the epoch-2
training loss:

- reference: `0.603543821014004`
- independent Linux replay: `0.6035437943022928`
- absolute difference: `2.6711711176297115e-08`

The reference and replay both use Python 3.12.7, PyTorch 2.12.0, identical
package versions, the same input arrays, the same initial state, and the same
minibatch order. The preserved Linux replay recorded PyTorch intra-op/inter-op
thread counts of `1/2`. The public replay script was subsequently hardened to
`1/1`; this later hardening was not part of the preserved replay. The observed
difference is therefore classified as unresolved cross-platform/runtime
numerical sensitivity in the CPU MLP arithmetic or backward/optimizer path, not
as a difference uniquely attributable to processor architecture.

The OCI process started with `PYTHONHASHSEED=0`. The value `32011` recorded
later by the replay script was assigned at runtime by the seed utility and did
not reset Python's interpreter-start hash randomization.

The later differences are downstream:

- best epoch: reference 44, replay 45;
- history length: reference 56, replay 57;
- trained-checkpoint hashes differ;
- selected learned kernel remains MLP-hidden linear, but selected SVM `C`
  changes from 1 to 10.

No tolerance was changed and no additional multi-seed experiment was run.

## Claim and release consequence

This record completes the requested diagnostic response to a failed exact-lock
replay; it does not turn the replay into a pass. The scientific negative-result
interpretation is unchanged because the QFC and generic-kernel paths reproduce
exactly and the Linux MLP drift does not create a QFC advantage.

The release formally discloses this platform sensitivity while treating the
frozen seed-level outputs as the numerical evidence. An independent macOS ARM64
replay would be useful additional same-platform evidence but is not required by
the final hostile audit.

Replacing the frozen seed-11 learned-comparator scores with the Linux replay
scores changes none of the six ten-seed QFC-minus-learned mean signs and shifts
an aggregate learned mean by at most `0.001101`. The central
comparator-escalation conclusion is therefore robust to the observed seed-11
platform drift.

The immutable nested replay ZIP is authoritative for the original
`command_transcript.log` and `failure.log` records. These ignored log files are
not duplicated as loose files in the public source snapshot.

`results/independent_exact_lock_seed11/EVIDENCE_SHA256SUMS_RELATIVE.txt`
provides the same 53 evidence hashes using paths relative to the actual
extracted ZIP root. The original workflow-generated checksum file is retained
unchanged for provenance.
