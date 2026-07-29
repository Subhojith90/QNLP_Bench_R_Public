# Stage 8A regenerated-data identity policy

The frozen seed-11 CSV remains immutable and must retain SHA-256
`9a81a615a82537d3fa492c3abe001b0723ed2dd696ada0fdda9e9ca45b5ff80b`.

The replay compares the regenerated CSV with that frozen file as follows:

- the column schema and row count must match exactly;
- row order must match exactly using `id`, `text`, and `split`;
- `label` and `clean_label` must match exactly;
- all 49 model-visible feature columns must match exactly;
- every remaining non-floating metadata field must match exactly; and
- only `audit_latent_score` and `audit_latent_margin`, which are unavailable to
  all models, may differ, with absolute tolerance `2e-15` and zero relative
  tolerance.

The tolerance is therefore not applied to labels, samples, ordering, splits, or
model inputs. `scripts/stage8a_dataset_identity.py` implements the policy and
the replay writes the result to `generated_data_identity_gate.json`.
