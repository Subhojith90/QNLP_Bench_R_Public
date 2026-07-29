from __future__ import annotations
import argparse, copy
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'src'
for p in (str(SRC), str(ROOT)):
    if p not in sys.path: sys.path.insert(0,p)
from qnlpbench_r.config import load_config
from qnlpbench_r.data.datasets import load_dataset
from qnlpbench_r.data.preprocessing import fit_preprocessor, make_arrays
from qnlpbench_r.evaluation.evaluate import predict_labels
from qnlpbench_r.models import build_model
from qnlpbench_r.training.train import train_model
from qnlpbench_r.utils.io import ensure_dir, write_json
from qnlpbench_r.utils.logging_utils import setup_logger
from sklearn.metrics import balanced_accuracy_score

def read_yaml(path: str|Path) -> dict[str,Any]:
    with Path(path).open('r',encoding='utf-8') as f: return yaml.safe_load(f) or {}

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--config', default='configs/stage6a_latent_validity.yaml'); ap.add_argument('--output_dir', default=None); args=ap.parse_args()
    spec=read_yaml(args.config); base_path=ROOT/spec.get('base_config','configs/stage6_latent_base.yaml'); cfg=load_config(base_path)
    out=ensure_dir(args.output_dir or spec.get('output_dir','paper_assets/diagnostics/stage6a_latent_validity'))
    split_rows=[]; probe_rows=[]; gate_fail=[]
    prohibited={'audit_latent_score','audit_latent_margin','audit_semantic_seed','clean_label'}
    for seed in cfg['seed_list']:
        bundle=load_dataset(cfg, int(seed)); audit=bundle.manifest['split_audit']; visible=set(bundle.feature_columns)
        visible_failure=sorted(prohibited & visible)
        if visible_failure: gate_fail.append({'seed':seed,'reason':'audit_columns_visible','columns':visible_failure})
        for key in ['heldout_compositions_in_train','heldout_compositions_in_val','heldout_compositions_in_test','ood_composition_pair_overlap_with_train','ood_composition_pair_overlap_with_val','ood_composition_pair_overlap_with_test']:
            if int(audit.get(key,0)) != 0: gate_fail.append({'seed':seed,'reason':key,'value':int(audit.get(key,0))})
        if max(audit.get('exact_text_duplicate_overlap',{}).values() or [0]) != 0: gate_fail.append({'seed':seed,'reason':'exact_text_overlap'})
        for split in ['train','val','test','ood_composition','ood_depth']:
            frame=bundle.split(split)
            split_rows.append({'seed':seed,'split':split,'n':len(frame),'positive_rate':float(frame['label'].mean()),'clean_oracle_accuracy':float((frame['clean_label']==frame['label']).mean()),'mean_abs_latent_margin':float(frame['audit_latent_margin'].abs().mean()),'heldout_n':int(frame['is_heldout_composition'].sum())})
        prep=fit_preprocessor(bundle); arrays=make_arrays(bundle,prep)
        for model_cfg in spec.get('probe_models',[]):
            model=build_model({**model_cfg,'seed':int(seed)}, len(bundle.feature_columns), 2, bundle.feature_columns)
            if hasattr(model,'fit_labels'): model.fit_labels(torch.from_numpy(arrays['train'][1]))
            if hasattr(model,'fit_features'): model.fit_features(*arrays['train'])
            for split in ['test','ood_composition','ood_depth']:
                X,y=arrays[split]; pred=predict_labels(model,X,torch.device('cpu'))
                probe_rows.append({'seed':seed,'model_name':model_cfg['name'],'model_type':model_cfg['type'],'split':split,'balanced_accuracy':float(balanced_accuracy_score(y,pred))})
    split_df=pd.DataFrame(split_rows); probe_df=pd.DataFrame(probe_rows)
    split_df.to_csv(out/'stage6a_split_validity.csv',index=False); probe_df.to_csv(out/'stage6a_probe_results.csv',index=False)
    if not probe_df.empty: probe_df.groupby(['model_name','model_type','split'],as_index=False)['balanced_accuracy'].agg(['mean','std','count']).reset_index().to_csv(out/'stage6a_probe_summary.csv',index=False)
    gate={'stage':'Stage 6A - Latent Semantic Generator Validity Gate','gate_pass':not gate_fail,'failures':gate_fail,'model_visible_policy':'primitive_observable_inputs_only','hidden_semantics_not_model_visible':not bool(gate_fail),'n_seeds':len(cfg['seed_list']),'config':str(args.config),'outputs':['stage6a_split_validity.csv','stage6a_probe_results.csv','stage6a_probe_summary.csv']}
    write_json(gate,out/'stage6a_validity_manifest.json'); print(f"Stage 6A latent validity gate pass: {gate['gate_pass']}; failures: {gate_fail}")
if __name__=='__main__': main()
