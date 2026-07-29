from __future__ import annotations
import argparse
from pathlib import Path
from safe_zip_utils import safe_extract_zip

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--zip', required=True, dest='zip_path'); parser.add_argument('--output-dir', default='.'); args=parser.parse_args()
    safe_extract_zip(Path(args.zip_path), Path(args.output_dir))
    print({'imported_from': args.zip_path, 'output_dir': args.output_dir, 'safe_extraction': True})
if __name__=='__main__': main()
