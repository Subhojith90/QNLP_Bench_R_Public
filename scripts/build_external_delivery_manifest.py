from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from safe_zip_utils import sha256_file


def main() -> None:
    parser=argparse.ArgumentParser(description='Hash sealed Stage 6G delivery files after packaging. This manifest remains external to the outer ZIP.')
    parser.add_argument('--submission-zip', required=True)
    parser.add_argument('--source-zip', required=True)
    parser.add_argument('--include', action='append', default=[], help='Additional delivered file, e.g. final PDF report or supervisory audit PDF.')
    parser.add_argument('--output-manifest', default='QNLPBench_R_Stage6G_Delivery_Manifest.json')
    parser.add_argument('--sha256-file', default='QNLPBench_R_Stage6G_Public_Artifact_Submission.zip.sha256')
    args=parser.parse_args()
    paths=[Path(args.submission_zip), Path(args.source_zip), *[Path(item) for item in args.include]]
    missing=[str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing delivered files: {missing}')
    rows=[{'file': path.name, 'sha256': sha256_file(path), 'bytes': path.stat().st_size} for path in paths]
    manifest={'stage':'Stage 6G - External Delivery Manifest', 'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'policy':'Generated after the outer ZIP is sealed. This is the authoritative hash record for delivered outer archives and reports.', 'files': rows}
    Path(args.output_manifest).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    Path(args.sha256_file).write_text(f"{rows[0]['sha256']}  {paths[0].name}\n", encoding='utf-8')
    print({'delivery_manifest': args.output_manifest, 'submission_zip_sha256': rows[0]['sha256'], 'submission_zip_bytes': rows[0]['bytes'], 'delivered_files': len(rows)})


if __name__=='__main__': main()
