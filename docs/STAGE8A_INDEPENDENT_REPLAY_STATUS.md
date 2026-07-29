# Stage 8A independent cross-platform replay status

## Evidence levels

Stage 8A contains one internal clean seed-11 reconstruction and two
independently recorded Linux seed-11 replays. The frozen ten-seed outputs
remain the authoritative numerical evidence.

### Internal clean reconstruction

- Environment: macOS ARM64, Python 3.13.1, PyTorch 2.12.0.
- Maximum absolute discrepancy from the registered table values:
  `5.551115123125783e-17`.
- Interpretation: same-platform-family reconstruction, not independent
  cross-platform replication.

### First preserved Linux replay

- Source commit: `42202b13a7f54a25e67d50d62649ff2b19600a67`.
- GitHub Actions run: `30359998043`.
- PyTorch intra-op/inter-op threads: `1/2`.
- Artifact ID: `8688496689`.
- Artifact SHA-256:
  `2ad8c13b7382c6789f009f7bee595d96b71209feac122e45f3a80011c8116587`.
- Maximum learned balanced-accuracy discrepancy:
  `0.011011011011010985`.

The build, all 36 tests, regenerated-data identity gate, QFC pathways, and
generic-kernel pathway passed; the tensor-train result agreed within
`5.551115123125783e-17`. The learned MLP pathway failed the unchanged `1e-12`
gate. Initialization and minibatch order matched. The first detected
training-loss divergence occurred in epoch 2 with absolute difference
`2.6711711176297115e-08`.

### Final public-commit Linux replay

- Repository:
  <https://github.com/Subhojith90/QNLP_Bench_R_Public>.
- Source commit: `1072dbb46a97e59cd736a93c724cd355b67b24b9`.
- GitHub tags resolving to that commit: `v1.0.0` and `v1.0.1`.
- Workflow run:
  <https://github.com/Subhojith90/QNLP_Bench_R_Public/actions/runs/30433905960>.
- Workflow run ID: `30433905960`.
- Job ID: `90517168721`.
- Artifact ID: `8716699380`.
- Artifact name:
  `stage8a-independent-replay-1072dbb46a97e59cd736a93c724cd355b67b24b9`.
- GitHub artifact SHA-256:
  `7bf3b72efca74ac2268e5f09481f399109535afb910bc473d537d8da88579b1e`.
- Artifact size: `942191` bytes.
- PyTorch intra-op/inter-op threads: `1/1`.
- Built image ID:
  `sha256:cb393cb41b9a62f6eec718deda797f18c75f4324bf7c596f988e11617d386f3bf`.
- Built image archive SHA-256:
  `8d24a48c5583636b885f2fc93ffe03d7d8b65636d7bc5166a614bc9e365ce11`.

The container build, all 36 tests, regenerated-data identity gate, QFC
pathways, and generic-kernel pathway passed. The tensor-train maximum absolute
error was `5.551115123125783e-17`. The learned MLP pathway again failed the
unchanged `1e-12` gate despite identical initialization and complete minibatch
order. The first training-loss divergence was
`3.0906112113981976e-09` in epoch 1. The selected early-stopping epoch changed
from 44 to 67, the selected kernel changed from linear with `C=1` to
`rbf_gamma_x1` with `C=0.1`, and the maximum balanced-accuracy discrepancy was
`0.027027027027026973`.

No tolerance was relaxed. Replacing only the frozen seed-11 learned-comparator
scores with the final replay values changes none of the six ten-seed
QFC-minus-learned mean signs; the maximum absolute shift in a ten-seed mean is
`0.002702702702702696`.

## Interpretation

Both independent replays verify execution, data identity, exact QFC and
generic-kernel reconstruction, tensor-train agreement to round-off, and
robustness of the central negative-result conclusion. Neither establishes
tolerance-exact cross-platform MLP retraining. The MLP divergence is classified
as unresolved cross-platform/runtime numerical sensitivity and is not
attributed uniquely to processor architecture.

The final downloaded workflow artifact is stored under
`results/final_public_commit_replay_1_1/`. Its original ZIP digest matches the
GitHub artifact digest, and its extraction-relative 53-file
`EVIDENCE_SHA256SUMS.txt` manifest verifies without path rewriting.
