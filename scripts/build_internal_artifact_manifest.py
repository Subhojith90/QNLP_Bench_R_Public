from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from safe_zip_utils import sha256_file


def archive_path_for_evidence(path: Path) -> Path:
    if path.name.startswith('qnlpbench_r_code_refined_'):
        return Path('Source') / path.name
    if path.suffix.lower() == '.pdf':
        return Path('Reports') / path.name
    return Path('Output') / path.name


def add_file(rows: list[dict], source: Path, archive_path: Path, category: str) -> None:
    if not source.exists() or not source.is_file():
        return
    rows.append({'archive_path': archive_path.as_posix(), 'source_path': source.as_posix(), 'sha256': sha256_file(source), 'bytes': source.stat().st_size, 'category': category})


def main() -> None:
    parser = argparse.ArgumentParser(description='Create the Stage 6G internal payload manifest. The manifest does not hash itself or the outer ZIP.')
    parser.add_argument('--evidence-dir', default='evidence')
    parser.add_argument('--source-zip', required=True)
    parser.add_argument('--output-dir', default='results/stage6g/manifests')
    parser.add_argument('--report-file', action='append', default=[])
    args = parser.parse_args()
    rows: list[dict] = []
    evidence = Path(args.evidence_dir)
    if evidence.exists():
        for path in sorted(evidence.glob('*')):
            if path.is_file():
                add_file(rows, path, archive_path_for_evidence(path), 'submitted_evidence')
    source_zip = Path(args.source_zip)
    add_file(rows, source_zip, Path('Source') / source_zip.name, 'stage6g_source_release')
    for item in args.report_file:
        path = Path(item)
        add_file(rows, path, Path('Reports') / path.name, 'report')
    for path in sorted(Path('results/stage6g').rglob('*')):
        if not path.is_file() or 'manifests/internal_artifact_manifest' in path.as_posix() or path.name in {'external_delivery_manifest.json'}:
            continue
        parts = path.parts
        if 'framing' in parts:
            archive_path = Path('Framing') / path.name
        else:
            archive_path = Path('Verification') / Path(*parts[2:])
        add_file(rows, path, archive_path, 'stage6g_generated')
    for path in sorted(Path('figures').glob('stage6g_*')):
        add_file(rows, path, Path('Figures') / path.name, 'stage6g_figure')
    # remove duplicate archive entries deterministically; last instance wins only if hashes agree.
    by_archive: dict[str, dict] = {}
    for row in rows:
        key = row['archive_path']
        if key in by_archive and by_archive[key]['sha256'] != row['sha256']:
            raise ValueError(f'Conflicting payload files mapped to {key}')
        by_archive[key] = row
    rows = [by_archive[key] for key in sorted(by_archive)]
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'internal_artifact_manifest.csv', index=False)
    manifest = {'stage': 'Stage 6G - Public Reproducibility Hardening', 'manifest_policy': 'This internal manifest hashes the payload included inside the Stage 6G ZIP. It intentionally excludes itself and the outer ZIP. The final outer ZIP is hashed in an external delivery manifest after sealing.', 'n_payload_files': len(rows), 'files': rows}
    (output / 'internal_artifact_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print({'internal_manifest_payload_files': len(rows), 'manifest': str(output / 'internal_artifact_manifest.json')})


if __name__ == '__main__':
    main()
