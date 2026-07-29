# Stage 8A environment provenance

## Original frozen model-reference environment

The frozen Stage 6D reference MLP checkpoints and registered outputs were
produced on macOS ARM64 with Anaconda Python 3.12.7 and PyTorch 2.12.0. These
author-produced artifacts are the original numerical reference.

## Internal clean-reconstruction environment

The complete internal seed-11 reconstruction was run on macOS ARM64 with Python
3.13.1 and PyTorch 2.12.0. It matched the registered table values within a
maximum absolute discrepancy of `5.55e-17`. This is an internal
same-platform-family reconstruction, not an independent cross-platform
replication.

## Portable independent verification environment

The digest-pinned OCI definition in `Dockerfile` is the portable independent
Linux x86-64 verification environment.

- Base tag: `python:3.12.7-slim-bookworm`
- Base manifest-list digest:
  `sha256:60d9996b6a8a3689d36db740b49f4327be3be09a21122bd02fb8895abb38b50d`
- Python: CPython 3.12.7
- Python packages: exact versions in `requirements-lock.txt`
- Thread limits: one thread for OpenMP, OpenBLAS, MKL, NumExpr, and vecLib
- Final public replay script: PyTorch intra-op and inter-op thread counts are
  explicitly pinned to `1/1`
- Preserved independent replay: PyTorch intra-op/inter-op counts were `1/2`;
  the explicit `1/1` hardening was introduced afterward
- Python hash randomization at interpreter startup: `PYTHONHASHSEED=0` from the
  OCI environment

The image build does not upgrade package-management tools and installs the local
package without dependency resolution. Consequently, `Dockerfile`,
`requirements-lock.txt`, and `pyproject.toml` describe one consistent package
set. The container build, 36-test suite, dataset-identity gate, QFC pathway and
generic-kernel pathway passed; tensor-train results agreed within `5.55e-17`.
The learned MLP pathway exhibited disclosed cross-platform/runtime numerical
sensitivity, with a maximum balanced-accuracy difference of
`0.011011011011010985`. The preserved replay cannot attribute this difference
uniquely to processor architecture because it predates the final `1/1` PyTorch
thread hardening.

## Developer environment

`environment.yml` is provided for local development. It fixes Python 3.12.7 and
delegates Python packages to `requirements-lock.txt`, but Conda platform
packages and solver metadata can change. Results obtained only from that
environment are not accepted as the independent replay.

## Required replay evidence

An independent replay record must contain:

1. source commit and working-tree status;
2. host operating system and architecture;
3. container-engine version;
4. pinned base reference, built image ID, image inspection output, and an image
   archive SHA-256 digest;
5. installed Python package inventory and deterministic thread settings;
6. complete test and replay stdout/stderr with exit statuses;
7. the regenerated-data identity-gate report;
8. model-state hashes, epoch-history comparison, selected-comparator comparison,
   and the first numerical divergence when one exists; and
9. SHA-256 hashes for every evidence file.

The GitHub Actions workflow
`.github/workflows/stage8a-exact-lock-replay.yml` creates this record on an
independent Linux runner.

The immutable ZIP
`results/independent_exact_lock_seed11/stage8a-independent-replay-42202b13.zip`
is the authoritative container for the original replay log files. The extracted
repository view intentionally omits ignored `.log` files; all other extracted
evidence and the ZIP checksum are preserved.
