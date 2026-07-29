from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path[:0] = [str(SRC), str(ROOT)]
from qnlpbench_r.utils.io import ensure_dir, read_json, write_json

SPLITS = ['test', 'ood_composition', 'ood_depth']
METRICS = [f'{s}_balanced_accuracy' for s in SPLITS]


def collect(root: Path, source: str) -> pd.DataFrame:
    rows=[]
    for mp in sorted(root.glob('**/metrics.json')):
        run=mp.parent; meta=read_json(run/'run_metadata.json'); metrics=read_json(mp)
        model_summary=meta.get('model_summary', {}) or {}
        rows.append({
            'source': source, 'run_dir': run.name, 'model_name': meta.get('model_name'), 'model_type': meta.get('model_type'),
            'seed': int(meta.get('seed')), 'topology': model_summary.get('structured_entanglement', 'classical'),
            'parameter_count': model_summary.get('parameter_count', np.nan), 'elapsed_seconds': meta.get('elapsed_seconds', np.nan),
            **{m: metrics.get(m, np.nan) for m in ['train_balanced_accuracy','val_balanced_accuracy', *METRICS]}
        })
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, seed: int = 1307, n_boot: int = 10000) -> tuple[float,float,float]:
    values=np.asarray(values,dtype=float); values=values[np.isfinite(values)]
    if len(values)==0: return np.nan, np.nan, np.nan
    rng=np.random.default_rng(seed)
    means=rng.choice(values, size=(n_boot,len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(means,2.5)), float(np.percentile(means,97.5))


def direct_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (source, model, mtype, topology), grp in df.groupby(['source','model_name','model_type','topology'], dropna=False):
        base={'source':source,'model_name':model,'model_type':mtype,'topology':topology,'n_seeds':grp.seed.nunique(),
              'runtime_seconds_mean':grp.elapsed_seconds.mean(),'parameter_count_mean':grp.parameter_count.mean()}
        for metric in ['train_balanced_accuracy','val_balanced_accuracy',*METRICS]:
            mean,lo,hi=bootstrap_mean_ci(grp[metric].to_numpy())
            base[f'{metric}_mean']=mean; base[f'{metric}_ci_low']=lo; base[f'{metric}_ci_high']=hi; base[f'{metric}_std']=grp[metric].std(ddof=1)
        rows.append(base)
    return pd.DataFrame(rows)


def paired_topology_deltas(classical: pd.DataFrame, topology: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    registry=[]; rows=[]
    for split in SPLITS:
        metric=f'{split}_balanced_accuracy'
        class_means=classical.groupby('model_name')[metric].mean().sort_values(ascending=False)
        comparator=str(class_means.index[0])
        registry.append({'split':split,'primary_classical_comparator':comparator,'comparator_mean':float(class_means.iloc[0]),'selection_rule':'highest pre-registered classical mean over ten seeds'})
        c = classical[classical.model_name==comparator][['seed', metric]].rename(columns={metric:'classical_balanced_accuracy'})
        for model, grp in topology.groupby('model_name'):
            merged=grp[['seed',metric]].rename(columns={metric:'qfc_balanced_accuracy'}).merge(c,on='seed',how='inner')
            merged['model_name']=model; merged['split']=split; merged['delta']=merged.qfc_balanced_accuracy-merged.classical_balanced_accuracy
            rows.append(merged)
    long=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    summ=[]
    for (model,split),grp in long.groupby(['model_name','split']):
        mean,lo,hi=bootstrap_mean_ci(grp.delta.to_numpy())
        summ.append({'model_name':model,'split':split,'n_paired_seeds':len(grp),'delta_mean':mean,'delta_ci_low':lo,'delta_ci_high':hi,'delta_std':grp.delta.std(ddof=1),'positive_seed_count':int((grp.delta>0).sum())})
    return pd.DataFrame(registry), long, pd.DataFrame(summ)


def plot_direct(summary: pd.DataFrame, figures: Path) -> None:
    display=summary.copy(); order=display.sort_values('ood_composition_balanced_accuracy_mean',ascending=False).model_name.tolist()
    x=np.arange(len(order)); width=.24
    fig, ax=plt.subplots(figsize=(max(9,len(order)*.65),5.5))
    for i, split in enumerate(SPLITS):
        vals=[float(display.loc[display.model_name==m,f'{split}_balanced_accuracy_mean'].iloc[0]) for m in order]
        ax.bar(x+(i-1)*width, vals, width, label=split.replace('_',' '))
    ax.set_xticks(x, order, rotation=45, ha='right'); ax.set_ylabel('Balanced accuracy'); ax.set_ylim(0.35,1.0); ax.set_title('Stage 6D direct replication: classical and QFC models'); ax.legend(); fig.tight_layout()
    fig.savefig(figures/'stage6d_direct_model_comparison.pdf'); fig.savefig(figures/'stage6d_direct_model_comparison.png',dpi=180); plt.close(fig)


def plot_delta(delta_summary: pd.DataFrame, figures: Path) -> None:
    if delta_summary.empty: return
    piv=delta_summary.pivot(index='model_name',columns='split',values='delta_mean').reindex(columns=SPLITS)
    fig,ax=plt.subplots(figsize=(8,max(4,.45*len(piv)))); mx=max(.02,float(np.nanmax(np.abs(piv.to_numpy())))); im=ax.imshow(piv.fillna(0).to_numpy(), cmap='coolwarm', vmin=-mx, vmax=mx, aspect='auto')
    ax.set_xticks(range(len(SPLITS)),[s.replace('_',' ') for s in SPLITS]); ax.set_yticks(range(len(piv.index)),piv.index)
    for r in range(len(piv.index)):
        for c in range(len(SPLITS)): ax.text(c,r,f'{piv.iloc[r,c]:+.3f}',ha='center',va='center',fontsize=9)
    ax.set_title('Stage 6D QFC delta versus pre-registered best classical comparator'); fig.colorbar(im,ax=ax,label='Paired balanced-accuracy delta'); fig.tight_layout(); fig.savefig(figures/'stage6d_topology_paired_delta_heatmap.pdf'); fig.savefig(figures/'stage6d_topology_paired_delta_heatmap.png',dpi=180); plt.close(fig)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--classical_dir',default='results/stage6d/runs/classical'); ap.add_argument('--topology_dir',default='results/stage6d/runs/topology'); ap.add_argument('--kernel_dir',default='results/stage6d/kernel'); ap.add_argument('--output_dir',default='results/stage6d/summary'); ap.add_argument('--figures_dir',default='figures'); args=ap.parse_args()
    out=ensure_dir(args.output_dir); figures=ensure_dir(args.figures_dir)
    classical=collect(Path(args.classical_dir),'classical'); topology=collect(Path(args.topology_dir),'topology'); direct=pd.concat([classical,topology],ignore_index=True)
    direct.to_csv(out/'stage6d_direct_seed_results.csv',index=False)
    summary=direct_summary(direct); summary.to_csv(out/'stage6d_direct_summary.csv',index=False)
    registry, paired, delta_summary=paired_topology_deltas(classical,topology)
    registry.to_csv(out/'stage6d_primary_comparator_registry.csv',index=False); paired.to_csv(out/'stage6d_paired_deltas.csv',index=False); delta_summary.to_csv(out/'stage6d_paired_delta_summary.csv',index=False)
    plot_direct(summary, figures); plot_delta(delta_summary, figures)
    kernel_src=Path(args.kernel_dir)/'stage4d_quantum_vs_classical.csv'
    kernel_present=kernel_src.exists()
    if kernel_present:
        kernel=pd.read_csv(kernel_src); kernel.to_csv(out/'stage6d_kernel_comparison_by_cap.csv',index=False)
        cols=['quantum_trained_balanced_accuracy_mean','quantum_untrained_balanced_accuracy_mean','best_classical_balanced_accuracy_mean','delta_quantum_minus_best_classical','delta_trained_minus_untrained_quantum']
        ks=kernel.groupby(['model_name','split'],as_index=False)[cols].mean(); ks.to_csv(out/'stage6d_kernel_comparison_summary.csv',index=False)
        piv=ks.pivot(index='model_name',columns='split',values='delta_quantum_minus_best_classical').reindex(columns=SPLITS)
        fig,ax=plt.subplots(figsize=(8,max(4,.45*len(piv)))); mx=max(.02,float(np.nanmax(np.abs(piv.to_numpy())))); im=ax.imshow(piv.fillna(0).to_numpy(),cmap='coolwarm',vmin=-mx,vmax=mx,aspect='auto'); ax.set_xticks(range(3),[s.replace('_',' ') for s in SPLITS]); ax.set_yticks(range(len(piv.index)),piv.index); ax.set_title('Stage 6D trained QFC kernel minus selected classical kernel'); fig.colorbar(im,ax=ax,label='Balanced-accuracy delta'); fig.tight_layout(); fig.savefig(figures/'stage6d_kernel_delta_heatmap.pdf'); fig.savefig(figures/'stage6d_kernel_delta_heatmap.png',dpi=180); plt.close(fig)
    manifest={'stage':'Stage 6D - Focused Ten-Seed Latent-Compositional Replication','selected_regime':{'latent_dim':4,'min_abs_margin':0.10,'n_samples':1800},'n_classical_seed_rows':len(classical),'n_topology_seed_rows':len(topology),'n_total_direct_rows':len(direct),'kernel_present':kernel_present,'bootstrap_resamples':10000,'claim_boundary':'Replication evidence only. Any quantum interpretation must be conditional on paired uncertainty and comparison against strong classical baselines.'}
    write_json(manifest,out/'stage6d_summary_manifest.json'); print(f'Wrote Stage 6D summaries to {out}; kernel_present={kernel_present}')

if __name__=='__main__': main()
