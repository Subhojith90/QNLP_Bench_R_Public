from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROWS = [
    ("Stage 6E evaluates trained QFC state-overlap kernels against learned classical representation kernels.", "allowed", "Verified directly from the Stage 6E raw output and reproduction audit."),
    ("QFC training improves over untrained QFC kernels on the submitted latent-compositional benchmark.", "allowed", "A within-QFC geometry diagnostic; not superiority."),
    ("The Stage 6D QFC advantage over generic kernels does not survive learned classical representation-kernel comparison in Stage 6E.", "allowed", "Central comparator-escalation conclusion."),
    ("QNLPBench-R is useful as an audit-first control framework for detecting false-positive quantum-representation claims.", "allowed", "Methods/controlled negative-results framing."),
    ("Quantum advantage.", "forbidden", "Not supported by the learned-classical comparator."),
    ("QFC superiority over learned classical representations.", "forbidden", "Contradicted by Stage 6E."),
    ("Stable general OOD-composition quantum advantage.", "forbidden", "OOD-composition deltas are mixed and non-positive in mean."),
    ("Hardware relevance or real-language QNLP generalisation.", "forbidden", "Outside the submitted synthetic benchmark evidence."),
    ("A positive quantum-benefit manuscript claim.", "forbidden", "Inconsistent with the frozen comparator-escalation result."),
]

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--output-dir', default='results/stage6g/framing'); args=parser.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df=pd.DataFrame(ROWS, columns=['claim','status','basis'])
    df.to_csv(out/'final_claim_matrix.csv', index=False)
    (out/'final_claim_matrix.md').write_text('# Frozen Stage 6G Claim Matrix\n\n' + df.to_markdown(index=False) + '\n', encoding='utf-8')
    print({'claim_rows': len(df), 'output_dir': str(out)})
if __name__=='__main__': main()
