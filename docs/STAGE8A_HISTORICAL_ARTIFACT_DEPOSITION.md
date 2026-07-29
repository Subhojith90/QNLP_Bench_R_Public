# Historical artifact deposition

The public source tree retains the extracted inputs required by Stage 8A and
the registered hashes that identify their source archives. Large immutable
historical archives are distributed as assets of the tagged GitHub Release:

<https://github.com/Subhojith90/QNLP_Bench_R_Public/releases/tag/v1.0.0>

- `QNLPBench_R_Stage6D_Replication_Output.zip`
  - SHA-256:
    `fc71c84c5a3080033b30cb1f0e43fcee4aa4a5a63e6eaec185e9c10e071c92e9`
- `QNLPBench_R_Stage6E_Learned_Kernel_Challenge_Output.zip`
  - SHA-256:
    `fd60a81310645260ff127540aac409859167c4e5ccb47d42f9bd55064d9234e5`

The same release also carries the final public-commit `1/1` replay artifact,
the release-wide checksum manifest, the tagged source archive, and the clean
one-commit Git-history bundle. Zenodo DOI
<https://doi.org/10.5281/zenodo.21670285> archives the audited public source
commit; the additional large historical and replay assets remain available
from the GitHub Release.

Historical filenames are preserved as provenance identifiers and are not
current workflow-stage labels.

`docs/STAGE8A_PUBLIC_PROVENANCE_REDACTION.json` records:

- each redacted public provenance file;
- its pre-redaction and public SHA-256 digests;
- the replacement policy for private hostnames and paths; and
- the immutable hashes of the historical ZIP artifacts.

The redactions affect only machine-specific provenance strings. Scientific
arrays, labels, checkpoints, metrics, and results are unchanged.
