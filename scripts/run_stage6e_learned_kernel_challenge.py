from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path[:0] = [str(SRC), str(ROOT)]
from qnlpbench_r.evaluation.evaluate import select_kernel_sample_indices
from qnlpbench_r.models import build_model
from qnlpbench_r.utils.io import ensure_dir, read_json, write_json


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_arrays(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    pre = read_json(run_dir / 'data' / 'preprocessor.json')
    df = pd.read_csv(run_dir / 'data' / 'dataset.csv')
    cols = pre['feature_columns']
    mean = np.asarray(pre['mean'], dtype=np.float32)
    std = np.asarray(pre['std'], dtype=np.float32)
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ['train', 'val', 'test', 'ood_composition', 'ood_depth']:
        sub = df[df['split'].astype(str) == split]
        X = sub[cols].to_numpy(dtype=np.float32)
        y = sub[pre['label_column']].to_numpy(dtype=np.int64)
        arrays[split] = (((X - mean) / std).astype(np.float32), y)
    return arrays


def index_runs(root: Path) -> dict[tuple[str, int], Path]:
    out: dict[tuple[str, int], Path] = {}
    for meta_path in root.glob('**/run_metadata.json'):
        meta = read_json(meta_path)
        if meta.get('status') != 'completed':
            continue
        out[(str(meta.get('model_name')), int(meta.get('seed')))] = meta_path.parent
    return out


def model_from_run(run_dir: Path, trained: bool = True):
    cfg = read_yaml(run_dir / 'config.yaml')
    pre = read_json(run_dir / 'data' / 'preprocessor.json')
    model_cfg = cfg['models'][0]
    model = build_model(model_cfg, input_dim=len(pre['feature_columns']), n_classes=2, feature_columns=pre['feature_columns'])
    if trained and (run_dir / 'model_best.pt').exists():
        model.load_state_dict(torch.load(run_dir / 'model_best.pt', map_location='cpu'))
    model.eval()
    return model


def mlp_features(model: Any, X: np.ndarray, mode: str) -> np.ndarray:
    xt = torch.from_numpy(X.astype(np.float32))
    with torch.no_grad():
        if mode == 'hidden':
            out = model.net[:-1](xt)
        elif mode == 'logits':
            out = model(xt)
        else:
            raise ValueError(mode)
    return out.detach().cpu().numpy().astype(np.float64)


def qfc_states(model: Any, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model.quantum_states(torch.from_numpy(X.astype(np.float32))).detach().cpu().numpy()


def qfc_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.asarray(np.abs(A @ B.conj().T) ** 2, dtype=np.float64)


def sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)


def l1dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)


def median_gamma(X: np.ndarray) -> float:
    d = sqdist(X, X)
    nz = d[d > 1e-12]
    return 1.0 / max(float(np.median(nz)), 1e-8) if nz.size else 1.0


def normalized_linear(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    K = A @ B.T
    den = np.outer(np.linalg.norm(A, axis=1).clip(min=1e-12), np.linalg.norm(B, axis=1).clip(min=1e-12))
    return K / den


def kernel_matrix(A: np.ndarray, B: np.ndarray, family: str, params: dict[str, Any]) -> np.ndarray:
    if family in {'linear', 'cosine'}:
        return normalized_linear(A, B)
    if family == 'rbf':
        return np.exp(-float(params['gamma']) * sqdist(A, B))
    if family == 'laplacian':
        return np.exp(-float(params['gamma']) * l1dist(A, B))
    if family == 'poly':
        deg = int(params['degree']); gamma = float(params.get('gamma', 1.0 / max(1, A.shape[1])))
        KA = (gamma * (A @ B.T) + 1.0) ** deg
        da = (gamma * (A * A).sum(axis=1) + 1.0) ** deg
        db = (gamma * (B * B).sum(axis=1) + 1.0) ** deg
        return KA / np.sqrt(np.outer(da, db)).clip(min=1e-12)
    raise ValueError(family)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
    }


def fit_eval(Ktr: np.ndarray, ytr: np.ndarray, Kev: np.ndarray, yev: np.ndarray, C: float) -> dict[str, float]:
    clf = SVC(kernel='precomputed', C=float(C), probability=False)
    clf.fit(Ktr, ytr)
    return metrics(yev, clf.predict(Kev))


def select_indices(y: np.ndarray, cap: int | str, seed: int, strategy: str) -> np.ndarray:
    actual = len(y) if str(cap) == 'full' else min(int(cap), len(y))
    return select_kernel_sample_indices(y, sample_cap=actual, seed=seed, strategy=strategy)


def tree_proximity(model: Any, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    la = model.apply(A); lb = model.apply(B)
    if la.ndim == 3:
        la = la[:, :, 0]; lb = lb[:, :, 0]
    return np.mean(la[:, None, :] == lb[None, :, :], axis=2).astype(np.float64)


def candidate_embeddings(Xtr: np.ndarray, Xva: np.ndarray, Xev: dict[str, np.ndarray], mlp: Any, ytr: np.ndarray, cfg: dict[str, Any], seed: int):
    gammas = [float(x) for x in cfg.get('gamma_multipliers', [0.5, 1.0, 2.0])]
    degrees = [int(x) for x in cfg.get('poly_degrees', [2, 3])]
    base_sets = {
        'raw': (Xtr.astype(float), Xva.astype(float), {k: v.astype(float) for k, v in Xev.items()}),
        'mlp_hidden': (mlp_features(mlp, Xtr, 'hidden'), mlp_features(mlp, Xva, 'hidden'), {k: mlp_features(mlp, v, 'hidden') for k, v in Xev.items()}),
        'mlp_logits': (mlp_features(mlp, Xtr, 'logits'), mlp_features(mlp, Xva, 'logits'), {k: mlp_features(mlp, v, 'logits') for k, v in Xev.items()}),
    }
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    base_sets['polynomial_interactions'] = (poly.fit_transform(Xtr), poly.transform(Xva), {k: poly.transform(v) for k, v in Xev.items()})
    for source, (Atr, Ava, Aev) in base_sets.items():
        allowed = cfg.get('classical_kernel_families', {}).get(source, [])
        g0 = median_gamma(Atr)
        for fam in allowed:
            if fam in {'linear', 'cosine'}:
                yield source, fam, fam, {}, kernel_matrix(Atr, Atr, fam, {}), kernel_matrix(Ava, Atr, fam, {}), {k: kernel_matrix(v, Atr, fam, {}) for k, v in Aev.items()}
            elif fam in {'rbf', 'laplacian'}:
                for mult in gammas:
                    params={'gamma': g0*mult, 'gamma_multiplier': mult}
                    yield source, fam, f'{fam}_gamma_x{mult:g}', params, kernel_matrix(Atr, Atr, fam, params), kernel_matrix(Ava, Atr, fam, params), {k: kernel_matrix(v, Atr, fam, params) for k, v in Aev.items()}
            elif fam == 'poly':
                for degree in degrees:
                    params={'degree': degree, 'gamma': 1.0/max(1,Atr.shape[1])}
                    yield source, fam, f'poly_degree_{degree}', params, kernel_matrix(Atr, Atr, fam, params), kernel_matrix(Ava, Atr, fam, params), {k: kernel_matrix(v, Atr, fam, params) for k, v in Aev.items()}
    # Random Fourier learned nonlinear feature maps on primitive features.
    for mult in gammas:
        gamma = median_gamma(Xtr) * mult
        for ncomp in [int(x) for x in cfg.get('rff_components', [128, 256])]:
            rff = RBFSampler(gamma=gamma, n_components=ncomp, random_state=seed + ncomp + int(mult * 100))
            Rtr = rff.fit_transform(Xtr); Rva = rff.transform(Xva); Rev = {k: rff.transform(v) for k,v in Xev.items()}
            for fam in cfg.get('classical_kernel_families', {}).get('random_fourier', []):
                yield 'random_fourier', fam, f'rff_{ncomp}_gamma_x{mult:g}_{fam}', {'gamma_multiplier': mult, 'n_components': ncomp}, kernel_matrix(Rtr,Rtr,fam,{}), kernel_matrix(Rva,Rtr,fam,{}), {k: kernel_matrix(v,Rtr,fam,{}) for k,v in Rev.items()}
    # Supervised tree-proximity kernels; trees are trained on the same capped training data as kernel-SVM.
    ntrees = int(cfg.get('tree_estimators', 300))
    for kind in cfg.get('classical_kernel_families', {}).get('tree_proximity', []):
        if kind == 'extra_trees':
            est = ExtraTreesClassifier(n_estimators=ntrees, random_state=seed, n_jobs=1)
        elif kind == 'random_forest':
            est = RandomForestClassifier(n_estimators=ntrees, random_state=seed, n_jobs=1)
        else:
            continue
        est.fit(Xtr, ytr)
        yield 'tree_proximity', kind, f'{kind}_proximity_{ntrees}', {'n_estimators': ntrees}, tree_proximity(est,Xtr,Xtr), tree_proximity(est,Xva,Xtr), {k: tree_proximity(est,v,Xtr) for k,v in Xev.items()}


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float); vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/stage6e_learned_kernel.yaml')
    ap.add_argument('--max-seeds', type=int, default=None)
    ap.add_argument('--model', action='append', default=None)
    args = ap.parse_args()
    cfg = read_yaml(args.config)
    base = Path(cfg['stage6d_results_dir'])
    classical = index_runs(base / 'classical'); topology = index_runs(base / 'topology')
    models = args.model or list(cfg['qfc_models'])
    seeds = sorted({s for (m,s) in classical if m == cfg['mlp_model']})
    if args.max_seeds is not None: seeds = seeds[:args.max_seeds]
    missing = [(m,s) for m in models for s in seeds if (m,s) not in topology]
    if missing: raise FileNotFoundError(f'Missing QFC Stage 6D runs: {missing[:5]}')
    out = ensure_dir(cfg['output_dir']); figures = ensure_dir(cfg['figures_dir'])
    rows=[]; selection=[]; sample_rows=[]
    strategy=str(cfg.get('kernel_sample_strategy','stratified')); Cs=[float(c) for c in cfg['c_values']]
    for seed in seeds:
        mlp_dir = classical[(cfg['mlp_model'], seed)]
        arrays=load_arrays(mlp_dir); mlp=model_from_run(mlp_dir, trained=True)
        for tcap in cfg['train_sample_caps']:
            tr_idx=select_indices(arrays['train'][1], tcap, seed+8101, strategy); va_idx=select_indices(arrays['val'][1], cfg['eval_sample_caps'][-1], seed+8103, strategy)
            Xtr,ytr=arrays['train'][0][tr_idx], arrays['train'][1][tr_idx]; Xva,yva=arrays['val'][0][va_idx], arrays['val'][1][va_idx]
            for ecap in cfg['eval_sample_caps']:
                eval_arrays={}; eval_indices={}
                for split in cfg['splits']:
                    idx=select_indices(arrays[split][1], ecap, seed+8200+len(split), strategy)
                    eval_indices[split]=idx; eval_arrays[split]=arrays[split][0][idx]
                    for i in idx: sample_rows.append({'seed':seed,'train_sample_cap':str(tcap),'eval_sample_cap':str(ecap),'split':split,'sample_index':int(i)})
                for i in tr_idx: sample_rows.append({'seed':seed,'train_sample_cap':str(tcap),'eval_sample_cap':str(ecap),'split':'train','sample_index':int(i)})
                candidates=list(candidate_embeddings(Xtr,Xva,eval_arrays,mlp,ytr,cfg,seed))
                best=None; best_score=-1.0
                for source,fam,name,params,Ktr,Kva,Kev in candidates:
                    for C in Cs:
                        val=fit_eval(Ktr,ytr,Kva,yva,C)
                        selection.append({'seed':seed,'train_sample_cap':str(tcap),'eval_sample_cap':str(ecap),'source':source,'kernel_family':fam,'kernel_name':name,'C':C,'selected':False,'val_balanced_accuracy':val['balanced_accuracy'],'params_json':json.dumps(params,sort_keys=True)})
                        if val['balanced_accuracy'] > best_score:
                            best_score=val['balanced_accuracy']; best=(source,fam,name,params,C,Ktr,Kev)
                assert best is not None
                source,fam,name,params,C,Ktr,Kev=best
                for rec in reversed(selection):
                    if rec['seed']==seed and rec['train_sample_cap']==str(tcap) and rec['eval_sample_cap']==str(ecap) and rec['kernel_name']==name and rec['C']==C and rec['source']==source:
                        rec['selected']=True; break
                for split in cfg['splits']:
                    m=fit_eval(Ktr,ytr,Kev[split],arrays[split][1][eval_indices[split]],C)
                    rows.append({'seed':seed,'train_sample_cap':str(tcap),'eval_sample_cap':str(ecap),'split':split,'model_name':'best_learned_classical','kernel_source':source,'kernel_family':fam,'kernel_name':name,'C':C,'selection_score':best_score,**m})
                # compare all registered QFC kernels against the same selected learned representation kernel.
                for model_name in models:
                    qdir=topology[(model_name, seed)]
                    trained=model_from_run(qdir, trained=True); untrained=model_from_run(qdir, trained=False)
                    st_tr=qfc_states(trained,Xtr); st_va=qfc_states(trained,Xva); su_tr=qfc_states(untrained,Xtr); su_va=qfc_states(untrained,Xva)
                    for qsource, mod, Str, Sva in [('qfc_trained',trained,st_tr,st_va),('qfc_untrained',untrained,su_tr,su_va)]:
                        Kt=qfc_kernel(Str,Str); Kv=qfc_kernel(Sva,Str); qbestC=None; qscore=-1.0
                        for Cq in Cs:
                            val=fit_eval(Kt,ytr,Kv,yva,Cq)
                            selection.append({'seed':seed,'train_sample_cap':str(tcap),'eval_sample_cap':str(ecap),'source':qsource,'kernel_family':'state_overlap','kernel_name':model_name,'C':Cq,'selected':False,'val_balanced_accuracy':val['balanced_accuracy'],'params_json':'{}'})
                            if val['balanced_accuracy'] > qscore: qscore=val['balanced_accuracy']; qbestC=Cq
                        for rec in reversed(selection):
                            if rec['seed']==seed and rec['train_sample_cap']==str(tcap) and rec['eval_sample_cap']==str(ecap) and rec['source']==qsource and rec['kernel_name']==model_name and rec['C']==qbestC:
                                rec['selected']=True; break
                        for split in cfg['splits']:
                            Xev=eval_arrays[split]; yev=arrays[split][1][eval_indices[split]]; Sev=qfc_states(mod,Xev); Ke=qfc_kernel(Sev,Str)
                            m=fit_eval(Kt,ytr,Ke,yev,float(qbestC))
                            rows.append({'seed':seed,'train_sample_cap':str(tcap),'eval_sample_cap':str(ecap),'split':split,'model_name':model_name,'kernel_source':qsource,'kernel_family':'state_overlap','kernel_name':model_name,'C':qbestC,'selection_score':qscore,**m})
    df=pd.DataFrame(rows); sel=pd.DataFrame(selection); samples=pd.DataFrame(sample_rows).drop_duplicates()
    df.to_csv(out/'stage6e_kernel_classifier_long.csv',index=False); sel.to_csv(out/'stage6e_kernel_selection_candidates.csv',index=False); samples.to_csv(out/'stage6e_sampled_indices.csv',index=False)
    comparison=[]
    for (model,split,tcap,ecap),g in df[df.kernel_source=='qfc_trained'].groupby(['model_name','split','train_sample_cap','eval_sample_cap']):
        q=g[['seed','balanced_accuracy']].rename(columns={'balanced_accuracy':'qfc_balanced_accuracy'})
        c=df[(df.model_name=='best_learned_classical') & (df.split==split) & (df.train_sample_cap==tcap) & (df.eval_sample_cap==ecap)][['seed','balanced_accuracy','kernel_source','kernel_name']].rename(columns={'balanced_accuracy':'classical_balanced_accuracy'})
        u=df[(df.kernel_source=='qfc_untrained') & (df.model_name==model) & (df.split==split) & (df.train_sample_cap==tcap) & (df.eval_sample_cap==ecap)][['seed','balanced_accuracy']].rename(columns={'balanced_accuracy':'untrained_balanced_accuracy'})
        z=q.merge(c,on='seed').merge(u,on='seed'); z['delta_qfc_minus_learned_classical']=z.qfc_balanced_accuracy-z.classical_balanced_accuracy; z['delta_trained_minus_untrained']=z.qfc_balanced_accuracy-z.untrained_balanced_accuracy; z['model_name']=model; z['split']=split; z['train_sample_cap']=tcap; z['eval_sample_cap']=ecap; comparison.append(z)
    comp=pd.concat(comparison,ignore_index=True); comp.to_csv(out/'stage6e_paired_kernel_deltas_by_cap.csv',index=False)
    averaged=comp.groupby(['model_name','split','seed'],as_index=False).agg(qfc_balanced_accuracy=('qfc_balanced_accuracy','mean'),classical_balanced_accuracy=('classical_balanced_accuracy','mean'),untrained_balanced_accuracy=('untrained_balanced_accuracy','mean'),delta_qfc_minus_learned_classical=('delta_qfc_minus_learned_classical','mean'),delta_trained_minus_untrained=('delta_trained_minus_untrained','mean'))
    averaged.to_csv(out/'stage6e_seed_averaged_paired_deltas.csv',index=False)
    summ=[]
    for (model,split),g in averaged.groupby(['model_name','split']):
        m,lo,hi=bootstrap_ci(g.delta_qfc_minus_learned_classical.to_numpy(),int(cfg.get('bootstrap_resamples',10000)),int(cfg.get('random_seed',6113))+len(model)+len(split))
        t,tl,th=bootstrap_ci(g.delta_trained_minus_untrained.to_numpy(),int(cfg.get('bootstrap_resamples',10000)),int(cfg.get('random_seed',6113))+100+len(model)+len(split))
        summ.append({'model_name':model,'split':split,'n_seeds':g.seed.nunique(),'qfc_balanced_accuracy_mean':g.qfc_balanced_accuracy.mean(),'learned_classical_balanced_accuracy_mean':g.classical_balanced_accuracy.mean(),'untrained_qfc_balanced_accuracy_mean':g.untrained_balanced_accuracy.mean(),'delta_qfc_minus_learned_classical_mean':m,'delta_qfc_minus_learned_classical_ci_low':lo,'delta_qfc_minus_learned_classical_ci_high':hi,'positive_seed_count':int((g.delta_qfc_minus_learned_classical>0).sum()),'delta_trained_minus_untrained_mean':t,'delta_trained_minus_untrained_ci_low':tl,'delta_trained_minus_untrained_ci_high':th})
    summary=pd.DataFrame(summ); summary.to_csv(out/'stage6e_learned_kernel_challenge_summary.csv',index=False)
    piv=summary.pivot(index='model_name',columns='split',values='delta_qfc_minus_learned_classical_mean').reindex(columns=cfg['splits'])
    fig,ax=plt.subplots(figsize=(8,max(3.5,0.55*len(piv)))); mx=max(.02,float(np.nanmax(np.abs(piv.to_numpy())))); im=ax.imshow(piv.fillna(0).to_numpy(),cmap='coolwarm',vmin=-mx,vmax=mx,aspect='auto'); ax.set_xticks(range(len(cfg['splits'])),[x.replace('_',' ') for x in cfg['splits']]); ax.set_yticks(range(len(piv.index)),piv.index); ax.set_title('Stage 6E QFC kernel minus best learned classical representation kernel')
    for r in range(len(piv.index)):
        for c in range(len(cfg['splits'])): ax.text(c,r,f'{piv.iloc[r,c]:+.3f}',ha='center',va='center')
    fig.colorbar(im,ax=ax,label='Paired balanced-accuracy delta'); fig.tight_layout(); fig.savefig(figures/'stage6e_qfc_minus_learned_kernel_delta.pdf'); fig.savefig(figures/'stage6e_qfc_minus_learned_kernel_delta.png',dpi=180); plt.close(fig)
    write_json({'stage':cfg['stage_label'],'config':args.config,'stage6d_results_dir':str(base),'models':models,'n_seeds':len(seeds),'n_rows':len(df),'n_selection_candidates':len(sel),'n_sample_indices':len(samples),'output_files':['stage6e_kernel_classifier_long.csv','stage6e_kernel_selection_candidates.csv','stage6e_sampled_indices.csv','stage6e_seed_averaged_paired_deltas.csv','stage6e_learned_kernel_challenge_summary.csv'],'claim_boundary':cfg['claim_boundary']},out/'stage6e_manifest.json')
    print(f'Stage 6E completed for {len(seeds)} seeds and {len(models)} QFC models; output={out}')

if __name__ == '__main__':
    main()
