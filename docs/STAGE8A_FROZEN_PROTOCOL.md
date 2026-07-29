# Stage 8A frozen scientific-strengthening protocol

Protocol frozen before inspection of any Stage 8A performance result.

## Scope and provenance

Stage 8A is restricted to the single latent semantic programme used in the sealed
Stage 7F evidence (`semantic_seed=314159`, latent dimension 4, margin 0.10, no
label noise). It does not modify the Stage 7F package. The Stage 7F supervisor
package has SHA-256
`63138564e490a410f30e78de8552b14d2384baadd40f9b55a2664714e509184d`.

Stage 8A imports two frozen inputs:

- Stage 6D replication archive:
  `fc71c84c5a3080033b30cb1f0e43fcee4aa4a5a63e6eaec185e9c10e071c92e9`
- Stage 6E learned-kernel archive:
  `fd60a81310645260ff127540aac409859167c4e5ccb47d42f9bd55064d9234e5`

The retained QFC topologies are `qfc_all_to_all` and `qfc_random_01`. The
experimental seeds are 11, 23, 37, 41, 53, 67, 79, 83, 97, and 109.

## Primary analysis: index-matched comparator escalation

For each seed, one stratified sample of at most 128 records is selected from
train, validation, ID test, held-out-pair OOD, and OOD-depth. The seed offsets
are fixed in `configs/stage8a.yaml`. Every model and comparator uses the same
saved integer indices. Each index file is serialized once as NumPy `int64`,
hashed with SHA-256, and reopened before use. The audit fails if a hash changes
or if any model records a different index hash.

The trained QFC checkpoints are the frozen Stage 6D checkpoints. QFC SVM
regularisation is selected from `C={0.1,1,10,100}` using validation balanced
accuracy. Generic primitive-input kernels and learned-representation kernels use
the identical capped data. All kernel hyperparameters and SVM regularisation are
selected using validation balanced accuracy only. Ties are resolved by the
candidate order frozen in the configuration.

Generic candidates are normalized linear/cosine, RBF with median-distance
gamma times `{0.5,1,2}`, Laplacian with the same multipliers, and normalized
degree-2 polynomial kernels. Learned candidates reproduce the registered Stage
6E families: trained MLP hidden states and logits, degree-2 interactions, random
Fourier features, and ExtraTrees proximity. No test or OOD score enters
selection.

For each seed, topology, and evaluation split, the release records QFC,
selected-generic, and selected-learned balanced accuracy; all three paired
differences; positive-seed counts; and descriptive 95% paired bootstrap
intervals using 10,000 resamples. The primary inferential posture is descriptive.

## Exact-initialization controls

New QFC training runs use deterministic Torch algorithms on CPU. Before model
construction, Python, NumPy, and Torch seeds are set. The post-construction RNG
states and the exact initial state dictionary are saved. Training begins from
that object without re-instantiation. Initial and selected trained checkpoints
are hashed and linked one-to-one in a manifest. Evaluation uses the primary
analysis index files.

For each topology and seed, a second model starts from the same initial state and
is trained after a fixed, seeded permutation of training labels. Its initial
checkpoint hash must equal the corresponding ordinary-training initial hash.
The primary training diagnostic is paired balanced accuracy of the trained
state-overlap kernel minus the exact-initial state-overlap kernel. The
label-shuffled trained state is a negative control.

Training budget: at most 80 epochs, Adam learning rate 0.003, weight decay
0.0001, batch size 128, early-stopping patience 12, gradient clipping at 1.0.
Validation balanced accuracy selects the checkpoint. No OOD score is used.

## Depth-stratified held-out-pair analysis

Within each seed's selected held-out-pair sample, records are separated into
fewer than three modifiers and at least three modifiers. The analysis reports
sample and class counts, selected QFC and selected learned-classical balanced
accuracy, and paired differences by topology. Strata containing only one class
are reported but balanced accuracy is marked unavailable. Aggregate bootstrap
intervals are descriptive because some strata are small.

## Tensor-train (matrix-product-state) baseline

The classical compositional comparator is a tensor-train classifier. The 49
primitive features are treated as seven ordered semantic sites: subject, verb,
object, negation, and four modifier positions; a scalar bias channel is appended
at each site. The model contracts trainable rank-3 cores sequentially and maps
the final bond state to two logits. Bond dimension is selected from
`{2,4,8,16,32}` using validation balanced accuracy only.

Every candidate uses the same capped training and validation indices, optimizer
budget, and early-stopping rule: Adam, learning rate 0.003, weight decay 0.0001,
at most 120 epochs, batch size 128, patience 20, gradient clipping at 1.0.
The release records selected bond dimension, parameter count, runtime,
best epoch, convergence status, and balanced accuracy on all evaluation splits.
The final bond representation defines an RBF kernel whose gamma multiplier and
SVM C are validation-selected using the frozen grids. This representation kernel
is used for paired QFC comparisons and kernel diagnostics.

## Kernel diagnostics

On the capped training sample, diagnostics are calculated for trained QFC,
exact-initial QFC, selected learned classical, and selected tensor-train kernels:

- centered kernel-target alignment;
- eigenvalues of the symmetrized centered kernel;
- entropy effective rank and stable rank;
- dominant-eigenvalue fraction;
- numerical condition number over positive eigenvalues;
- mean within-class and between-class similarity;
- selected-SVM support-vector fraction and signed margin quantiles;
- normalized Frobenius displacement from exact initial to trained QFC kernel;
- centered alignment between each QFC and learned-classical kernel.

Numerical tolerances, centering, and eigenvalue clipping are fixed in code and
reported in the release metadata.

## End-to-end rerun and release conditions

Seed 11 is rerun from dataset generation through both QFC trainings, learned and
generic comparators, tensor-train selection, kernel evaluation, and table
construction. The rerun must reproduce the dataset content hash and frozen
Stage 6D trained-QFC checkpoint evaluation within a documented tolerance; newly
trained initialization-matched results are expected to be deterministic only
within the locked Stage 8A environment.

The release includes a one-command smoke run, locked Python requirements,
container recipe, command transcript, hardware/software capture, output hashes,
failure log, and numerical comparison report. A Stage 8A archive is sealed only
after all tests and manifest verification pass. The manuscript will distinguish
historical Stage 7F evidence from new Stage 8A results.

## Claim rules

Stage 8A does not test quantum advantage, hardware performance, natural-language
semantics, multiple semantic programmes, asymptotic scaling, or resource
advantage. Results will be reported whether favourable, null, or unfavourable.
No new topology, comparator family, hyperparameter grid, or outcome-dependent
exclusion may be introduced after this protocol freeze without a dated,
hash-recorded amendment that is labelled exploratory.
