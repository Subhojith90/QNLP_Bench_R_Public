from __future__ import annotations
import argparse, importlib.util
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'src'
for p in [str(SRC),str(ROOT)]:
    if p not in sys.path: sys.path.insert(0,p)
from qnlpbench_r.evaluation.evaluate import select_kernel_sample_indices
from qnlpbench_r.utils.io import ensure_dir, read_json, write_json

def load_stage4b():
    sp=importlib.util.spec_from_file_location('stage4b', ROOT/'scripts'/'run_stage4b_kernel_classifier.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

def read_yaml(path):
    with Path(path).open('r',encoding='utf-8') as f: return yaml.safe_load(f) or {}

def resolve_cap(v, n): return n if str(v).lower()=='full' else min(int(v),n)

def audit_indices(mod, run_dir: Path, train_cap:int, eval_cap:int, strategy:str, splits:list[str])->list[dict[str,Any]]:
    meta=read_json(run_dir/'run_metadata.json'); seed=int(meta['seed']); rows=[]
    for split, purpose, off, cap in [('train','train_kernel_reference',7001,train_cap),('val','validation_selection',7003,eval_cap)]:
        X,y=mod._load_preprocessed(run_dir,split); idx=select_kernel_sample_indices(y,sample_cap=min(cap,len(y)),seed=seed+off,strategy=strategy)
        for i in idx: rows.append({'run_dir':run_dir.name,'model_name':meta.get('model_name'),'seed':seed,'split':split,'sampling_purpose':purpose,'train_sample_cap':train_cap,'eval_sample_cap':eval_cap,'local_index':int(i),'y_true':int(y[i])})
    for split in splits:
        X,y=mod._load_preprocessed(run_dir,split)
        for source, off in [('shared_quantum_eval',7100+len(split)),('shared_classical_eval',7100+len(split))]:
            idx=select_kernel_sample_indices(y,sample_cap=min(eval_cap,len(y)),seed=seed+off,strategy=strategy)
            for i in idx: rows.append({'run_dir':run_dir.name,'model_name':meta.get('model_name'),'seed':seed,'split':split,'sampling_purpose':source,'train_sample_cap':train_cap,'eval_sample_cap':eval_cap,'local_index':int(i),'y_true':int(y[i])})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/stage4d_kernel_classifier_replication.yaml'); args=ap.parse_args()
    cfg=read_yaml(args.config); mod=load_stage4b(); out=ensure_dir(cfg.get('output_dir','paper_assets/diagnostics/stage4d_kernel_classifier')); figs=ensure_dir(cfg.get('figures_dir','figures_stage4d'))
    roots=[Path(x) for x in cfg['results_dirs']]; runs=mod._find_qfc_runs(roots)[:int(cfg.get('max_runs',1000))]
    rows=[]; sels=[]; indices=[]; failures=[]
    for run in runs:
        for tr_raw in cfg.get('train_sample_caps',[384]):
            for ev_raw in cfg.get('eval_sample_caps',[256]):
                try:
                    Xtr,ytr=mod._load_preprocessed(run,'train'); Xev,yev=mod._load_preprocessed(run,'test')
                    tr=resolve_cap(tr_raw,len(ytr)); ev=resolve_cap(ev_raw,len(yev))
                    local=dict(cfg); local['train_sample_cap']=tr; local['eval_sample_cap']=ev
                    r,s=mod._analyze_run(run,local); rows.extend(r); sels.extend(s); indices.extend(audit_indices(mod,run,tr,ev,str(cfg.get('kernel_sample_strategy','stratified')),list(cfg['splits'])))
                except Exception as exc: failures.append({'run_dir':str(run),'train_cap':str(tr_raw),'eval_cap':str(ev_raw),'error':repr(exc)})
    df=pd.DataFrame(rows); sel=pd.DataFrame(sels); idx=pd.DataFrame(indices)
    df.to_csv(out/'stage4d_kernel_classifier_long.csv',index=False); sel.to_csv(out/'stage4d_kernel_selection_candidates.csv',index=False); idx.to_csv(out/'stage4d_sampled_indices.csv',index=False)
    summary, comp=mod._summaries(df); summary.to_csv(out/'stage4d_kernel_classifier_summary.csv',index=False); comp.to_csv(out/'stage4d_quantum_vs_classical.csv',index=False)
    if failures: pd.DataFrame(failures).to_csv(out/'stage4d_failures.csv',index=False)
    mod._make_figures(comp,figs)
    write_json({'stage':str(cfg.get('stage_label', 'Stage 4D - Strict-control predictive kernel replication')),'config':args.config,'results_dirs':[str(x) for x in roots],'n_qfc_runs':len(runs),'n_cap_combinations':len(cfg.get('train_sample_caps',[384]))*len(cfg.get('eval_sample_caps',[256])),'n_rows':len(df),'n_sample_index_rows':len(idx),'n_failures':len(failures),'claim_boundary':'Exploratory strict-control replication only; no quantum advantage claim.'}, out/'stage4d_kernel_replication_manifest.json')
    print(f'Analyzed {len(runs)} QFC run directories over cap grid; failures={len(failures)}; output={out}')
if __name__=='__main__': main()
