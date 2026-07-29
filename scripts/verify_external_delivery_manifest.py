from __future__ import annotations
import argparse, json
from pathlib import Path
from safe_zip_utils import sha256_file

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--manifest', required=True); parser.add_argument('--root', default='.'); args=parser.parse_args()
    data=json.loads(Path(args.manifest).read_text(encoding='utf-8')); root=Path(args.root); missing=[]; mismatch=[]; size_mismatch=[]
    for row in data['files']:
        path=root/row['file']
        if not path.exists(): missing.append(row['file'])
        elif sha256_file(path) != row['sha256']: mismatch.append(row['file'])
        elif path.stat().st_size != row['bytes']: size_mismatch.append(row['file'])
    result={'verified': not missing and not mismatch and not size_mismatch, 'n_files': len(data['files']), 'missing': missing, 'hash_mismatch': mismatch, 'size_mismatch': size_mismatch}; print(result)
    if not result['verified']: raise SystemExit(1)
if __name__=='__main__': main()
