from __future__ import annotations
import argparse, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from safe_zip_utils import sha256_file, sha256_zip_member, validate_zip_members

KEYS=['model_name','split']
SUMMARY_METRICS={
 'qfc_balanced_accuracy_mean':'qfc_balanced_accuracy',
 'learned_classical_balanced_accuracy_mean':'classical_balanced_accuracy',
 'untrained_qfc_balanced_accuracy_mean':'untrained_balanced_accuracy',
 'delta_qfc_minus_learned_classical_mean':'delta_qfc_minus_learned_classical',
 'delta_trained_minus_untrained_mean':'delta_trained_minus_untrained',
}

def reproduce_summary(seed_path_or_df, submitted_path_or_df):
    seeds = seed_path_or_df if isinstance(seed_path_or_df, pd.DataFrame) else pd.read_csv(seed_path_or_df)
    submitted = submitted_path_or_df if isinstance(submitted_path_or_df, pd.DataFrame) else pd.read_csv(submitted_path_or_df)
    calc=seeds.groupby(KEYS,as_index=False).agg(n_seeds=('seed','nunique'),qfc_balanced_accuracy_mean=('qfc_balanced_accuracy','mean'),learned_classical_balanced_accuracy_mean=('classical_balanced_accuracy','mean'),untrained_qfc_balanced_accuracy_mean=('untrained_balanced_accuracy','mean'),delta_qfc_minus_learned_classical_mean=('delta_qfc_minus_learned_classical','mean'),positive_seed_count=('delta_qfc_minus_learned_classical',lambda x:int((x>0).sum())),delta_trained_minus_untrained_mean=('delta_trained_minus_untrained','mean'))
    comparison=submitted.merge(calc,on=KEYS,how='outer',suffixes=('_submitted','_recomputed'),indicator=True); errors=[]
    for metric in ['n_seeds','positive_seed_count',*SUMMARY_METRICS.keys()]:
        comparison[f'abs_error_{metric}']=(comparison[f'{metric}_submitted'].astype(float)-comparison[f'{metric}_recomputed'].astype(float)).abs(); errors.append(comparison[f'abs_error_{metric}'].max())
    return calc, comparison, float(np.nanmax(errors)) if errors else float('nan')

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--zip', required=True, dest='zip_path'); parser.add_argument('--output-dir', default='results/stage6g/verification'); parser.add_argument('--tolerance', type=float, default=1e-10); args=parser.parse_args()
    archive=Path(args.zip_path).resolve(); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    names=set(validate_zip_members(archive)); mf='results/stage6e/stage6e_artifact_sha256_manifest.json'
    with zipfile.ZipFile(archive) as zf:
        internal_manifest=json.loads(zf.read(mf).decode('utf-8')); missing=[]; mismatches=[]
        for rec in internal_manifest['files']:
            if rec['path'] not in names: missing.append(rec['path'])
            elif sha256_zip_member(zf,rec['path']) != rec['sha256']: mismatches.append(rec['path'])
        seed_df=pd.read_csv(io.BytesIO(zf.read('results/stage6e/stage6e_seed_averaged_paired_deltas.csv')))
        summary_df=pd.read_csv(io.BytesIO(zf.read('results/stage6e/stage6e_learned_kernel_challenge_summary.csv')))
        selection=pd.read_csv(io.BytesIO(zf.read('results/stage6e/stage6e_kernel_selection_candidates.csv')))
    internal={'verified':not missing and not mismatches,'n_manifest_files':len(internal_manifest['files']),'missing':missing,'hash_mismatch':mismatches,'safe_zip':True}
    calc,comp,error=reproduce_summary(seed_df,summary_df)
    selected=selection[selection['selected'].astype(str).str.lower().isin(['true','1'])]
    audit=selected[['seed','train_sample_cap','eval_sample_cap','source','kernel_family','kernel_name','C','val_balanced_accuracy','params_json']].sort_values(['seed'])
    calc.to_csv(output/'stage6e_recomputed_summary.csv',index=False); comp.to_csv(output/'stage6e_reproduction_comparison.csv',index=False); audit.to_csv(output/'stage6e_kernel_selection_audit.csv',index=False)
    result={'stage':'Stage 6G independent Stage 6E verification','source_zip':str(archive),'source_zip_sha256':sha256_file(archive),'internal_artifact_verification':internal,'n_seed_rows':int(seed_df.shape[0]),'n_summary_rows':int(calc.shape[0]),'n_selected_kernel_rows':int(audit.shape[0]),'max_abs_recomputed_mean_error':error,'tolerance':args.tolerance,'reproduction_pass':bool(internal['verified'] and error<=args.tolerance),'claim_boundary':'Stage 6E supports a controlled negative result: trained QFC kernels do not outperform learned classical representation kernels.'}
    (output/'stage6e_verification_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result)
    if not result['reproduction_pass']: raise SystemExit(1)
if __name__=='__main__': main()
