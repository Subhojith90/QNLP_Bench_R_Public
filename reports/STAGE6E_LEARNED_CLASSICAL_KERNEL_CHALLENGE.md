# Stage 6E: Learned Classical Representation Kernel Challenge

Stage 6E-A is the focused screening diagnostic following the Stage 6D ten-seed replication. It does not redesign the latent-compositional benchmark and does not rerun the direct model experiment. It evaluates whether the replicated trained QFC state-overlap kernel signal on held-out-composition OOD survives comparison against stronger **learned classical representation kernels** derived from the same Stage 6D inputs and training evidence.

## Primary question

Does `qfc_all_to_all` or `qfc_random_01` retain a positive paired OOD-composition kernel-SVM delta after replacing generic raw-input classical kernels with learned or interaction-aware classical representation kernels?

## Classical representation-kernel challenge family

The validation-selected classical candidate pool contains:

- Raw primitive input kernels: linear/cosine/RBF/Laplacian/polynomial.
- Penultimate hidden-layer embeddings from the trained `sequence_mlp_256x3` model, with linear/cosine/RBF/Laplacian/polynomial kernels.
- Trained MLP logit-space kernels.
- Second-order primitive interaction feature kernels.
- Random Fourier feature-map kernels.
- Supervised Random-Forest and Extra-Trees proximity kernels trained on the same capped training sample.

Selection is performed only on the validation split. Test, OOD-composition and OOD-depth results are evaluated after selection. Identical split-local sampled indices are used for QFC and classical representation candidates within each seed/cap condition.

## QFC models retained

- `qfc_all_to_all`
- `qfc_random_01`
- `qfc_random_01_random_label` (negative control)
- `qfc_random_01_random_label` as negative control

## Claim boundary

A positive Stage 6E result would support a narrow learned-representation-kernel diagnostic under the latent-compositional benchmark. It would not establish quantum advantage, broad QNLP superiority, hardware relevance, or a manuscript-level claim without supervisor review.


## Execution tiers

`configs/stage6e_learned_kernel.yaml` runs the ten-seed Stage 6E-A screen at a matched cap of 128 to keep the learned-representation challenge computationally manageable. It must be run first. `configs/stage6e_confirmation_template.yaml` is a larger-cap Stage 6E-B confirmation template and must be run only if Stage 6E-A leaves at least one QFC topology with a positive paired OOD-composition delta against the best learned classical representation kernel.


## Scope correction frozen in Stage 6F

The Stage 6E output evaluates `qfc_all_to_all`, `qfc_random_01`, and `qfc_random_01_random_label` only. It does not evaluate `qfc_linear_chain`. The challenge outcome is a negative comparator-escalation result: learned classical representation kernels remove the earlier QFC advantage over generic kernels.
