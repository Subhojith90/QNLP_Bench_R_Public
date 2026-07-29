from __future__ import annotations
import argparse
from pathlib import PurePosixPath, Path
from safe_zip_utils import validate_zip_members

FORBIDDEN_PARTS = {'.pytest_cache', '__pycache__'}

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--zip', required=True, dest='zip_path'); args=parser.parse_args()
    names=validate_zip_members(Path(args.zip_path))
    forbidden=[name for name in names if FORBIDDEN_PARTS.intersection(PurePosixPath(name).parts) or name.endswith(('.pyc','.pyo'))]
    result={'source_zip': args.zip_path, 'clean': not forbidden, 'n_entries': len(names), 'forbidden_entries': forbidden}
    print(result)
    if forbidden:
        raise SystemExit(1)
if __name__=='__main__': main()
