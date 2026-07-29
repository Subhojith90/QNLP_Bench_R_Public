from __future__ import annotations
import argparse, datetime, platform, subprocess, sys
from pathlib import Path

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--output-dir', default='results/stage6g/environment'); args=parser.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    snapshot=(
        f'captured_utc: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n'
        f'python: {sys.version}\n'
        f'platform: {platform.platform()}\n'
        f'executable: {sys.executable}\n'
        'scope: Stage 6G artifact verification environment only; raw experiment archives preserve their original run metadata.\n'
    )
    (out/'environment_snapshot.txt').write_text(snapshot, encoding='utf-8')
    freeze=subprocess.run([sys.executable,'-m','pip','freeze','--all'],capture_output=True,text=True,check=True).stdout
    (out/'requirements-lock.txt').write_text(freeze, encoding='utf-8')
    print({'output_dir': str(out), 'lock_lines': len(freeze.splitlines())})
if __name__=='__main__': main()
