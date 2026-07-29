from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path[:0] = [str(SRC), str(ROOT)]
from qnlpbench_r.utils.io import read_json, write_json, ensure_dir


def key_for(run_dir: Path) -> tuple[str, int]:
    meta = read_json(run_dir / 'run_metadata.json')
    return str(meta.get('model_name')), int(meta.get('seed'))


def audit(root: Path, source: str, expected: int) -> tuple[pd.DataFrame, dict]:
    rows, keys = [], {}
    for metrics_path in sorted(root.glob('**/metrics.json')):
        run = metrics_path.parent
        meta = read_json(run / 'run_metadata.json')
        key = key_for(run)
        keys.setdefault(key, []).append(str(run))
        rows.append({'source': source, 'run_dir': run.name, 'model_name': key[0], 'seed': key[1], 'status': meta.get('status'), 'elapsed_seconds': meta.get('elapsed_seconds')})
    duplicates = [{'source': source, 'model_name': k[0], 'seed': k[1], 'n_runs': len(v), 'runs': '|'.join(v)} for k,v in keys.items() if len(v)>1]
    missing_count = max(0, expected - len(keys))
    report = {'source': source, 'input_dir': str(root), 'expected_unique_runs': expected, 'observed_metric_dirs': len(rows), 'observed_unique_runs': len(keys), 'n_duplicates': len(duplicates), 'missing_count': missing_count, 'pass': len(keys) == expected and not duplicates}
    return pd.DataFrame(rows), {'report': report, 'duplicates': duplicates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--classical_dir', default='results/stage6d/runs/classical')
    ap.add_argument('--topology_dir', default='results/stage6d/runs/topology')
    ap.add_argument('--expected_classical', type=int, default=80)
    ap.add_argument('--expected_topology', type=int, default=70)
    ap.add_argument('--output_dir', default='results/stage6d/manifests')
    args = ap.parse_args()
    out = ensure_dir(args.output_dir)
    frames=[]; reports=[]; duplicates=[]
    for root, source, expected in [(Path(args.classical_dir),'classical',args.expected_classical),(Path(args.topology_dir),'topology',args.expected_topology)]:
        frame, result = audit(root, source, expected); frames.append(frame); reports.append(result['report']); duplicates += result['duplicates']
    pd.concat(frames, ignore_index=True).to_csv(out/'run_inventory.csv', index=False)
    pd.DataFrame(duplicates).to_csv(out/'duplicate_run_audit.csv', index=False)
    manifest = {'stage': 'Stage 6D - Focused Ten-Seed Replication', 'reports': reports, 'pass': all(x['pass'] for x in reports), 'claim_boundary': 'Run-integrity gate only; not scientific evidence.'}
    write_json(manifest, out/'run_integrity_manifest.json')
    print(f"Stage 6D run integrity pass: {manifest['pass']}; reports: {reports}")
    if not manifest['pass']:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
