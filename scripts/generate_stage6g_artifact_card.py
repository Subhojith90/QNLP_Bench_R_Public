from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--output-dir', default='results/stage6g/framing'); parser.add_argument('--verification-manifest', default='results/stage6g/verification/stage6e_verification_manifest.json'); args=parser.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    vm=json.loads(Path(args.verification_manifest).read_text(encoding='utf-8'))
    text=f'''# QNLPBench-R Stage 6G Artifact Card

## Scope
Stage 6G is an artifact-metadata correction and public-reproducibility hardening release. It performs **no new quantum-performance experiment**. It freezes and packages the Stage 5B–6E evidence trajectory, with the decisive scientific conclusion based on Stage 6E.

## Frozen conclusion
Stage 6D reports positive trained-QFC kernel diagnostics against generic classical kernels. Stage 6E escalates the comparison to learned classical representation kernels and finds that the apparent QFC advantage does not survive. QFC training learns non-trivial state-space geometry relative to untrained QFC kernels; this is not evidence of quantum superiority.

## Verification checkpoint
- Stage 6E summary rows recomputed: {vm.get('n_summary_rows')}
- Seed-level paired rows inspected: {vm.get('n_seed_rows')}
- Maximum absolute recomputation error: {vm.get('max_abs_recomputed_mean_error')}
- Reproduction pass: {vm.get('reproduction_pass')}

## Claims allowed
- Stage 6E compares trained QFC state-overlap kernels with learned classical representation kernels.
- QFC training changes its own predictive kernel geometry on the submitted latent benchmark.
- The Stage 6D generic-kernel signal is closed by the stronger Stage 6E comparison.
- QNLPBench-R can be framed as an audit-first controlled benchmark methodology.

## Claims forbidden
- Quantum advantage or QFC superiority over learned classical representations.
- Stable general OOD-composition quantum advantage.
- Hardware relevance or real-language QNLP generalisation.
- Positive quantum-benefit manuscript framing.

## Reproduction commands
```bash
make test
make validate-stage6d
make validate-stage6e
make audit-stage6e-summary
make freeze-stage6g
```

## Manifest policy
`internal_artifact_manifest.json` hashes the contents packaged inside the Stage 6G ZIP, excluding the manifest files themselves. The final outer ZIP hash is recorded only after packaging in `QNLPBench_R_Stage6G_Public_Artifact_Submission.zip.sha256` and `QNLPBench_R_Stage6G_Delivery_Manifest.json`; this avoids a self-referential ZIP hash.

## Limitations
The benchmark is synthetic. Stage 6E uses a cap-128 learned-kernel screening challenge adequate to close a positive superiority claim, not to establish universal classical dominance. No hardware or natural-language claim is supported.
'''
    (out/'ARTIFACT_CARD.md').write_text(text, encoding='utf-8')
    print({'artifact_card': str(out/'ARTIFACT_CARD.md')})
if __name__=='__main__': main()
