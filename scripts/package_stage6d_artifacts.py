from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--output_zip',default='QNLPBench_R_Stage6D_Replication_Output.zip'); args=ap.parse_args()
    roots=[ROOT/'results'/'stage6d', ROOT/'figures']; files=[]
    for root in roots:
        if root.exists(): files += [p for p in root.rglob('*') if p.is_file() and p.suffix not in {'.pyc'} and '__pycache__' not in p.parts and '.DS_Store' not in p.name]
    manifest_path=ROOT/'results'/'stage6d'/'artifact_sha256_manifest.json'
    manifest_files=[p for p in sorted(set(files)) if p != manifest_path]
    manifest={'stage':'Stage 6D - Focused Ten-Seed Latent-Compositional Replication','layout':{'results':'All run artifacts, audits, kernel outputs and summaries live below results/stage6d/.','figures':'All generated plots live directly below figures/.'},'files':[{'path':str(p.relative_to(ROOT)),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in manifest_files]}
    manifest_path.parent.mkdir(parents=True,exist_ok=True); manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    files=sorted(set(manifest_files+[manifest_path])); out=ROOT/args.output_zip
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files: z.write(p,p.relative_to(ROOT))
    print(f'Wrote {out} with {len(files)} files')
if __name__=='__main__': main()
