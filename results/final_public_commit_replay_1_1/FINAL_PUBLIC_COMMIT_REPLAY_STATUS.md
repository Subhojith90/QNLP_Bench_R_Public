# Final public-commit replay status

The release-verification replay was dispatched manually from the public
repository at commit
`1072dbb46a97e59cd736a93c724cd355b67b24b9`.

- Repository: <https://github.com/Subhojith90/QNLP_Bench_R_Public>
- Workflow run: <https://github.com/Subhojith90/QNLP_Bench_R_Public/actions/runs/30433905960>
- Workflow job ID: `90517168721`
- Artifact ID: `8716699380`
- Downloaded artifact SHA-256:
  `7bf3b72efca74ac2268e5f09481f399109535afb910bc473d537d8da88579b1e`
- Artifact size: `942191` bytes
- PyTorch intra-op/inter-op threads: `1/1`
- Digest-pinned container build: pass
- Built image ID:
  `sha256:cb393cb41b9a62f6eec718deda797f18c75f4324bf7c596f988e11617d386f3bf`
- Built image archive SHA-256:
  `8d24a48c5583636b885f2fc93ffe03d7d8b65636d7bc5166a614bc9e365ce11`
- Complete automated test suite: 36 passed
- Regenerated-data identity gate: pass
- QFC pathways: exact
- Generic-kernel pathway: exact
- Tensor-train maximum absolute error: `5.551115123125783e-17`
- Learned-MLP selection match: false
- Maximum learned balanced-accuracy discrepancy: `0.027027027027026973`
- Strict replay gate: fail at the unchanged `1e-12` tolerance

The replay used the exact initialization and minibatch order recorded by the
release. MLP training nevertheless diverged in the first epoch: the first
recorded loss difference above `1e-12` was
`3.0906112113981976e-9`. The selected early-stopping epoch changed from 44 to
67, and the selected learned kernel changed from the registered linear kernel
with `C=1` to `rbf_gamma_x1` with `C=0.1`. This is unresolved
cross-platform/runtime numerical sensitivity in the MLP arithmetic or
backward/optimizer path. It is not attributed uniquely to processor
architecture, and no tolerance was relaxed.

Replacing the frozen seed-11 learned-comparator values with the final replay
values changes none of the six ten-seed QFC-minus-learned mean signs. All six
remain negative. The maximum absolute shift in a ten-seed mean is
`0.002702702702702696`. The scientific comparator-escalation conclusion is
therefore robust to this replay, although tolerance-exact cross-platform MLP
retraining is not established.

The downloaded workflow artifact is preserved in the adjacent
`stage8a-independent-replay-1072dbb46a97e59cd736a93c724cd355b67b24b9`
directory. Its 53-file `EVIDENCE_SHA256SUMS.txt` manifest verifies without
error.
