# Auditing Apparent Quantum Feature-Circuit Gains

This folder contains the registered Stage 8A scientific-strengthening analysis,
including the source code, configurations, inputs, checkpoints, data products,
seed-level outputs, reports, and verification material for:

> *Auditing Apparent Quantum Feature-Circuit Gains on Synthetic Compositional
> Language Tasks*

The study evaluates two trained six-qubit quantum feature circuits on a synthetic
latent-compositional classification task. Its principal comparison uses identical
saved sample indices for the QFC, generic-kernel, and learned-representation
comparators. The positive QFC comparison against generic kernels does not survive
comparison with validation-selected learned classical representation kernels. Exact
initialization-matched controls support a narrower trainability result, not a
quantum-advantage claim.

## Registered analyses

- index-matched generic and learned-kernel comparisons across ten paired seeds;
- exact initialization-matched and label-shuffled QFC controls;
- modifier-depth-stratified held-out-pair evaluation;
- validation-selected tensor-train baselines;
- kernel alignment, effective-rank, class-separation, and support-vector
  diagnostics; and
- a clean end-to-end rerun of one complete seed.

The protocol was frozen before Stage 8A results were inspected. Its seal is in
`docs/STAGE8A_PROTOCOL_SEAL.json`. Historical Stage 6D, Stage 6E, and Stage 7F
names occur only where immutable input filenames or provenance records require
them. The sealed Stage 7F supervisor package is not modified.

`docs/STAGE8A_PROTOCOL_CLARIFICATION_002.md` records that the registered
MLP-hidden `linear` and `cosine` labels are implementation aliases. The labels
remain in the frozen ordering; no performance result was changed.

For provenance, the earlier Stage 6G hardening release performed no new model training
and no quantum scale-up. Stage 8A is a separate registered scientific
release and does include the initialization controls, tensor-train fits, and
clean seed-level rerun described above.

## Reproducibility environments

The release distinguishes three provenance layers. The original frozen
model-reference artifacts were produced on macOS ARM64 with Python 3.12.7. An
internal clean seed-11 reconstruction ran on macOS ARM64 with Python 3.13.1 and
matched the registered table values within `5.55e-17`. The digest-pinned Linux
x86-64 OCI image defined by `Dockerfile` is the portable independent
verification environment. It uses CPython 3.12.7 and the exact package versions
in `requirements-lock.txt`. Build and run it with:

```bash
docker build --pull=false -t qnlpbench-r:stage8a .
docker run --rm qnlpbench-r:stage8a python -m pytest tests -q
```

The fully instrumented seed-11 replay writes its evidence outside the container:

```bash
mkdir -p independent-replay-evidence
docker run --rm \
  -e QNLPBENCH_HOST_LABEL=independent-replay-host \
  -v "$PWD/independent-replay-evidence:/evidence" \
  qnlpbench-r:stage8a \
  python scripts/run_clean_end_to_end_seed11.py --output /evidence/results
```

`environment.yml` is a developer convenience. It resolves the
same direct pins but does not replace the OCI definition because Conda solver
metadata and platform packages can vary. See
`docs/STAGE8A_NORMATIVE_ENVIRONMENT.md` for the provenance layers and
evidence requirements.

The completed independent Linux replay and its first-divergence diagnosis are
preserved under `results/independent_exact_lock_seed11/` and summarized in
`docs/STAGE8A_INDEPENDENT_REPLAY_STATUS.md`. The image build, test suite, and
strict dataset gate pass. The numerical replay remains a recorded failure
because platform-sensitive MLP training changes the selected SVM regularization
value; no tolerance was relaxed.

The short and complete registered workflows remain available inside the image:

```bash
docker run --rm qnlpbench-r:stage8a bash scripts/run_stage8a_smoke.sh
docker run --rm qnlpbench-r:stage8a bash scripts/run_stage8a_full.sh
```

Verify the repository-level file manifest with:

```bash
python scripts/verify_folder_manifest.py
```

The public repository contains code, configurations, frozen inputs, and
machine-readable outputs. It intentionally excludes the unpublished manuscript.
Historical Stage 6D, Stage 6E, and Stage 7F archives are deposited separately as
immutable archival assets; their registered hashes and role are documented in
`docs/STAGE8A_HISTORICAL_ARTIFACT_DEPOSITION.md`.

## Citation and archive

The citable Zenodo concept DOI for all versions of this software release is
[`10.5281/zenodo.21670284`](https://doi.org/10.5281/zenodo.21670284). Creator
metadata is defined in `.zenodo.json` in the manuscript author order:
Subhojit Halder, Srinjoy Ganguly, and Shalini Devendrababu. Version-specific
DOIs and verified release assets are linked from the corresponding GitHub
release pages.

## Claim boundary

The results support a descriptive trainability diagnostic for the simulated QFCs
and a controlled negative result against the selected learned classical
representations. They do not establish quantum advantage, QFC superiority,
hardware relevance, resource advantage, or natural-language generalisation.
