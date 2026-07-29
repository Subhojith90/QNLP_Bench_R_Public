from __future__ import annotations
import argparse, io, zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from safe_zip_utils import validate_zip_members

def read_csv_member(archive: Path, member: str) -> pd.DataFrame:
    names=set(validate_zip_members(archive))
    if member not in names: raise FileNotFoundError(member)
    with zipfile.ZipFile(archive) as zf: return pd.read_csv(io.BytesIO(zf.read(member)))

def save(fig, base: Path) -> None:
    fig.tight_layout(); fig.savefig(base.with_suffix('.pdf')); fig.savefig(base.with_suffix('.png'),dpi=180); plt.close(fig)

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--stage6d-zip',default='evidence/QNLPBench_R_Stage6D_Replication_Output.zip'); parser.add_argument('--stage6e-zip',default='evidence/QNLPBench_R_Stage6E_Learned_Kernel_Challenge_Output.zip'); parser.add_argument('--figures-dir',default='figures'); args=parser.parse_args()
    out=Path(args.figures_dir); out.mkdir(parents=True,exist_ok=True)
    for p in out.glob('stage6g_*'): p.unlink()
    d=read_csv_member(Path(args.stage6d_zip),'results/stage6d/summary/stage6d_kernel_comparison_summary.csv'); e=read_csv_member(Path(args.stage6e_zip),'results/stage6e/stage6e_learned_kernel_challenge_summary.csv')
    models=['qfc_all_to_all','qfc_random_01']; labels=['QFC all-to-all','QFC random-01']
    gd=[float(d[(d.model_name==m)&(d.split=='ood_composition')].iloc[0]['delta_quantum_minus_best_classical']) for m in models]; ge=[float(e[(e.model_name==m)&(e.split=='ood_composition')].iloc[0]['delta_qfc_minus_learned_classical_mean']) for m in models]
    x=np.arange(len(models)); w=.35; fig,ax=plt.subplots(figsize=(7,4)); ax.bar(x-w/2,gd,w,label='Stage 6D: vs generic kernels'); ax.bar(x+w/2,ge,w,label='Stage 6E: vs learned classical kernels'); ax.axhline(0,color='black',linewidth=.8); ax.set_xticks(x,labels); ax.set_ylabel('OOD-composition balanced-accuracy delta'); ax.set_title('Comparator escalation removes apparent QFC kernel advantage'); ax.legend(); save(fig,out/'stage6g_comparator_escalation')
    order=['qfc_all_to_all','qfc_random_01','qfc_random_01_random_label']; splits=['test','ood_composition','ood_depth']; matrix=e[e.model_name.isin(order)].pivot(index='model_name',columns='split',values='delta_qfc_minus_learned_classical_mean').reindex(index=order,columns=splits).values
    fig,ax=plt.subplots(figsize=(7,3.7)); extent=max(.05,float(np.nanmax(np.abs(matrix)))); im=ax.imshow(matrix,cmap='coolwarm',vmin=-extent,vmax=extent,aspect='auto'); ax.set_xticks(range(3),['ID test','OOD comp.','OOD depth']); ax.set_yticks(range(3),['QFC all-to-all','QFC random-01','QFC random-label']);
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]): ax.text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center',fontsize=9)
    ax.set_title('Frozen claim boundary: QFC minus learned classical kernel'); fig.colorbar(im,ax=ax,label='Balanced-accuracy delta'); save(fig,out/'stage6g_claim_boundary'); print({'figures':['stage6g_comparator_escalation','stage6g_claim_boundary'],'output_dir':str(out)})
if __name__=='__main__': main()
