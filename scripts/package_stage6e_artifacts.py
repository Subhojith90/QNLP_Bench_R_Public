from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--output_zip', default='QNLPBench_R_Stage6E_Learned_Kernel_Challenge_Output.zip'); args=ap.parse_args()
    root=Path('.').resolve(); out=root/args.output_zip
    include=[]
    r=root/'results'/'stage6e'
    if not r.exists(): raise FileNotFoundError('results/stage6e not found; run Stage 6E first.')
    include.extend([p for p in r.rglob('*') if p.is_file()])
    f=root/'figures'; include.extend([p for p in f.glob('stage6e_*') if p.is_file()])
    manifest=[]
    for p in sorted(set(include)):
        manifest.append({'path':str(p.relative_to(root)),'sha256':sha256(p),'bytes':p.stat().st_size})
    manifest_path=r/'stage6e_artifact_sha256_manifest.json'; manifest_path.write_text(json.dumps({'stage':'Stage 6E','files':manifest},indent=2),encoding='utf-8')
    include.append(manifest_path)
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(set(include)): zf.write(p,p.relative_to(root))
    print({'output_zip':str(out),'n_files':len(set(include)),'bytes':out.stat().st_size})
if __name__=='__main__': main()
