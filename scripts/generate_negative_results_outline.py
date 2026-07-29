from __future__ import annotations
import argparse
from pathlib import Path

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--output-dir', default='results/stage6g/framing'); args=parser.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    text='''# QNLPBench-R: Audit-First Evaluation of Apparent Quantum Representation Gains in Compositional Language Circuits

## Proposed framing
A methodology / controlled negative-results paper or thesis chapter. The contribution is a reproducible control ladder for identifying when apparently positive QFC representation findings disappear under rigorous comparator escalation.

## Central research question
When quantum feature circuits appear to improve compositional generalisation diagnostics, do the observed gains survive split repair, baseline escalation, topology controls, replication, and learned-classical representation comparison?

## Proposed structure

### 1. Introduction and motivation
- QNLP experiments require unusually careful controls because positive signals can be produced by exposed label structure, weak comparator families, or post-hoc topology selection.
- Position QNLPBench-R as an audit-first framework, not a quantum-advantage benchmark.

### 2. Audit ladder design
- Stage 5B: repaired split/baseline/topology audit demonstrates classical saturation of the deterministic primitive-rule task.
- Stage 6A/6B: latent-compositional redesign and calibration avoids immediate classical saturation while maintaining clean OOD splits.
- Stage 6C: prospective topology pilot establishes candidate QFC behaviour.
- Stage 6D: ten-seed replication identifies a positive QFC-kernel diagnostic against generic kernels.
- Stage 6E: learned-classical representation kernels close the apparent positive QFC claim.

### 3. Experimental protocol
- Primitive observable inputs; latent semantics hidden from trainable models.
- Held-out composition isolation; no exact text overlap.
- Ten-seed paired evaluation.
- Direct baselines, QFC topology/control family, generic kernels, then learned MLP representation kernels.
- Report runtime/parameter information and negative controls.

### 4. Results
- State the Stage 6D generic-kernel signal only as motivation for Stage 6E.
- Headline the Stage 6E comparator-escalation closure: all principal QFC-minus-learned-classical means are non-positive.
- Retain the within-QFC trained-versus-untrained geometry finding as a narrow diagnostic result.

### 5. Discussion
- Why negative findings are valuable for QNLP methodology.
- What the benchmark does and does not establish.
- Limitations: synthetic semantics, cap-128 Stage 6E screening, no hardware or real-language extension.

### 6. Reproducibility and artifact release
- Safe ZIP validation, internal payload manifest, external delivery manifest, artifact card, claim matrix, full archived trajectory.

## Frozen claim boundary
Allowed: QFC training changes its own kernel geometry; apparent generic-kernel gains do not survive learned classical representations; QNLPBench-R is an audit-first control framework.

Forbidden: quantum advantage, QFC superiority, hardware relevance, real-language generalisation, stable OOD-composition quantum benefit.

## Supervisor decisions requested
1. Approve audit-first / controlled negative-results framing.
2. Approve whether this is pursued as a manuscript outline or a thesis-chapter section first.
3. Decide whether a separate future hypothesis should consider resource constraints, noise-aware behaviour, or semi-synthetic transfer; none belongs to the current claim chain.
'''
    (out/'NEGATIVE_RESULTS_OUTLINE.md').write_text(text, encoding='utf-8')
    print({'outline': str(out/'NEGATIVE_RESULTS_OUTLINE.md')})
if __name__=='__main__': main()
