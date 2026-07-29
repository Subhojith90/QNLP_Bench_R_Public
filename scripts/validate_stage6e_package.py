from __future__ import annotations
import argparse, json, zipfile
from pathlib import Path
from safe_zip_utils import sha256_zip_member, validate_zip_members

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--zip', required=True, dest='zip_path'); args=parser.parse_args(); archive=Path(args.zip_path)
    names=set(validate_zip_members(archive)); manifest_member='results/stage6e/stage6e_artifact_sha256_manifest.json'
    if manifest_member not in names: raise FileNotFoundError(f'Missing {manifest_member}')
    with zipfile.ZipFile(archive) as zf:
        manifest=json.loads(zf.read(manifest_member).decode('utf-8')); missing=[]; mismatch=[]
        for row in manifest['files']:
            if row['path'] not in names: missing.append(row['path'])
            elif sha256_zip_member(zf, row['path']) != row['sha256']: mismatch.append(row['path'])
    result={'package':str(archive),'verified':not missing and not mismatch,'n_manifest_files':len(manifest['files']),'missing':missing,'hash_mismatch':mismatch,'safe_zip':True}; print(result)
    if not result['verified']: raise SystemExit(2)
if __name__=='__main__': main()
