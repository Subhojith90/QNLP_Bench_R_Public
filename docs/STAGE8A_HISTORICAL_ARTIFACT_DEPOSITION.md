# Historical artifact deposition

The public source repository does not duplicate the large immutable development
archives. The original Stage 6D, Stage 6E, Stage 6G/7F, and supervisor-package
ZIP files are archival assets and must be deposited separately with permanent
identifiers when the release DOI is created.

The repository retains the extracted inputs needed by Stage 8A and the
registered hashes that identify their source archives. Historical filenames are
identifiers only; they are not current workflow stages.

`docs/STAGE8A_PUBLIC_PROVENANCE_REDACTION.json` records:

- each redacted public provenance file;
- its pre-redaction and public SHA-256 digests;
- the replacement policy for private hostnames and paths; and
- the immutable hashes of the historical ZIP artifacts.

The redactions affect only machine-specific provenance strings. Scientific
arrays, labels, checkpoints, metrics, and results are unchanged.
