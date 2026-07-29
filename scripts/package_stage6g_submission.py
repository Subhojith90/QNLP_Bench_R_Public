from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Package the Stage 6G public artifact payload using its internal artifact manifest.')
    parser.add_argument('--manifest', default='results/stage6g/manifests/internal_artifact_manifest.json')
    parser.add_argument('--output-zip', default='QNLPBench_R_Stage6G_Public_Artifact_Submission.zip')
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    output = Path(args.output_zip)
    with zipfile.ZipFile(output, 'w') as archive:
        for row in manifest['files']:
            source = Path(row['source_path'])
            compression = zipfile.ZIP_STORED if source.suffix.lower() in {'.zip', '.pdf', '.png'} else zipfile.ZIP_DEFLATED
            archive.write(source, row['archive_path'], compress_type=compression)
        archive.write(manifest_path, 'Verification/internal_artifact_manifest.json', compress_type=zipfile.ZIP_DEFLATED)
        csv_path = manifest_path.with_suffix('.csv')
        archive.write(csv_path, 'Verification/internal_artifact_manifest.csv', compress_type=zipfile.ZIP_DEFLATED)
    print({'output_zip': str(output.resolve()), 'payload_files': len(manifest['files']), 'zip_entries': len(manifest['files']) + 2, 'bytes': output.stat().st_size})


if __name__ == '__main__':
    main()
