# Stage 6G: Artifact Metadata Correction, Public Reproducibility Hardening, and Negative-Results Outline

Stage 6G is a final artifact-release preparation stage approved after supervisory audit of Stage 6F. It adds no new performance experiment. The stage corrects metadata workflow design by splitting payload integrity and final-delivery integrity into two non-self-referential manifests, adopts safe ZIP extraction in all validators, removes generated cache artifacts from the source release, and generates reviewer-facing framing documents.

## Manifest design

A ZIP file cannot contain an authoritative hash of itself without changing that hash. Stage 6G therefore uses:

1. `internal_artifact_manifest.json`, stored inside the Stage 6G ZIP, covering its payload files but excluding itself and the outer ZIP; and
2. `QNLPBench_R_Stage6G_Delivery_Manifest.json`, stored beside the sealed ZIP, covering the final ZIP, source ZIP and any subsequently delivered report PDFs.

## Scientific boundary

Stage 6G freezes the Stage 6E negative comparator-escalation result: trained QFC kernels learn measurable predictive geometry relative to untrained QFC kernels, but do not outperform learned classical representation kernels under the submitted latent-compositional benchmark. No quantum-scale experiment or positive benefit claim is introduced.
