from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from safe_zip_utils import safe_extract_zip, sha256_file


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--zip', required=True, dest='zip_path'); args=parser.parse_args()
    archive=Path(args.zip_path)
    with tempfile.TemporaryDirectory(prefix='stage6g_validate_') as directory:
        root=Path(directory)
        safe_extract_zip(archive, root)
        manifest_path=root/'Verification/internal_artifact_manifest.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        missing=[]; mismatches=[]; size_mismatches=[]
        for row in manifest['files']:
            path=root/row['archive_path']
            if not path.exists():
                missing.append(row['archive_path'])
            elif sha256_file(path) != row['sha256']:
                mismatches.append(row['archive_path'])
            elif path.stat().st_size != row['bytes']:
                size_mismatches.append(row['archive_path'])
    result={'package': str(archive), 'verified': not missing and not mismatches and not size_mismatches, 'n_payload_files': len(manifest['files']), 'missing': missing, 'hash_mismatch': mismatches, 'size_mismatch': size_mismatches}
    print(result)
    if not result['verified']:
        raise SystemExit(1)


if __name__=='__main__': main()
