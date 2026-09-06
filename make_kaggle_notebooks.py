"""Generate self-contained Kaggle notebooks from kaggle_qas_core.py."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "kaggle_notebooks"
OUT.mkdir(exist_ok=True)
CORE = (ROOT / "kaggle_qas_core.py").read_text()


INSTALL = '''# Install only when Kaggle's image does not already provide PennyLane.
import importlib.util, subprocess, sys
if importlib.util.find_spec("pennylane") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pennylane>=0.40,<0.44"])
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip() + "\n"}


def notebook(title, purpose, task_cells):
    intro = f'''# {title}

{purpose}

### Kaggle execution

1. Select a Python notebook and enable Internet for the first dependency installation if PennyLane is absent.
2. For publication runs leave `FAST_MODE = False`. Set it to `True` only for a short pipeline check.
3. Outputs are written to `/kaggle/working/qas_materials/`. Download them or create a Kaggle Dataset version for reuse.
4. The code uses exact state-vector simulation; a GPU accelerates neural training but not every PennyLane operation.

The notebook is self-contained and does not require cloning the GitHub repository.
'''
    cells = [md(intro), code(INSTALL), code(CORE), code(
        '# False = requested publication configuration; True = quick functional check\nFAST_MODE = False\nseed_everything(20260906)\nprint("Output directory:", output_dir())'
    )] + task_cells
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"},
                         "kaggle": {"accelerator": "gpu", "dataSources": []}},
            "nbformat": 4, "nbformat_minor": 5}


NOTEBOOKS = {}

NOTEBOOKS["00_build_hubbard_multiseed_dataset.ipynb"] = notebook(
    "Dataset: five architecture seeds and dense Fermi–Hubbard sweep",
    "Generates the common VQE-label table used by the Hubbard experiments: five independently seeded, exactly disjoint architecture pools and $U/t=0,1,\\ldots,8$.",
    [
        md("## Generate or load labels\n\nPublication mode evaluates 24 architectures × 5 seeds × 9 interactions = **1080 VQE labels**, using 60 steps and two restarts. A checkpoint is written after every completed interaction value."),
        code('''df = ensure_hubbard_dataset(fast=FAST_MODE)
display(df.head())
print(df.groupby(["parameter", "arch_seed"]).size().unstack())
assert df["arch_seed"].nunique() >= (2 if FAST_MODE else 5)
assert set(df["parameter"].unique()) == set([0,2,4,6,8] if FAST_MODE else range(9))
assert np.isfinite(df[["energy", "exact", "gap"]]).all().all()
assert (df["gap"] >= -1e-10).all()
assert df.groupby(["arch_seed","arch_id"]).gates_json.nunique().max() == 1
architectures = df.drop_duplicates(["arch_seed","arch_id"])
assert architectures.gates_json.nunique() == len(architectures)
print("Rows:", len(df), "| unique architectures:", df[["arch_seed","arch_id"]].drop_duplicates().shape[0])'''),
        code('''roundtrip=[]
for U in sorted(df.parameter.unique()):
    H,n,_=hubbard_problem(float(U))
    roundtrip.append({"U_over_t":U,"max_coefficient_error":validate_factor_graph_roundtrip(H,n)})
roundtrip=pd.DataFrame(roundtrip); display(roundtrip)
assert roundtrip.max_coefficient_error.max() < 1e-10'''),
        code('''summary = df.groupby("parameter")["gap"].agg(["min", "median", "mean", "max"])
display(summary)
fig, ax = plt.subplots(figsize=(8,4))
sns.lineplot(data=df, x="parameter", y="gap", estimator="median", errorbar=("pi",95), marker="o", ax=ax)
ax.set(xlabel="$U/t$", ylabel="VQE gap", title="Dense Hubbard label distribution")
ax.grid(alpha=.2); plt.show()'''),
        code('''manifest = {
    "rows": int(len(df)), "architecture_seeds": sorted(df.arch_seed.unique().tolist()),
    "u_values": sorted(df.parameter.unique().tolist()),
    "label_steps": sorted(df.label_steps.unique().tolist()),
    "label_restarts": sorted(df.label_restarts.unique().tolist()),
    "dataset_version": sorted(df.dataset_version.unique().tolist()),
}
manifest_path = output_dir()/"hubbard_dense_multiseed_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
print("Download:", output_dir()/("hubbard_dense_multiseed_fast.csv" if FAST_MODE else "hubbard_dense_multiseed.csv"))''')
    ])


NOTEBOOKS["01_multiseed_bootstrap_statistics.ipynb"] = notebook(
    "Five-seed ranking statistics with bootstrap confidence intervals",
    "Runs an independent leakage-free joint-graph ranking experiment for every architecture seed, then reports mean Kendall $\\tau$, sample standard deviation, and a seed-level bootstrap 95% confidence interval.",
    [
        code('df = ensure_hubbard_dataset(fast=FAST_MODE)'),
        md("## Independent repeated experiments\n\nWithin each seed, architectures are divided by `arch_id`; interaction values are also disjoint. Thus every reported test point has both an unseen architecture and unseen $U/t$. Each replicate changes both the sampled architecture pool and neural-training RNG, so the interval measures end-to-end pipeline variability rather than architecture sampling alone."),
        code('''seed_rows = []
for replicate_seed, group in df.groupby("arch_seed", sort=True):
    group = group.reset_index(drop=True)
    ids = sorted(group.arch_id.unique())
    n = len(ids); train_ids = ids[:max(4, int(.60*n))]
    val_ids = ids[len(train_ids):max(len(train_ids)+2, int(.80*n))]
    test_ids = ids[max(len(train_ids)+2, int(.80*n)):]
    train_u = [0,1,2,4,6,8] if not FAST_MODE else [0,2,4]
    val_u = [3] if not FAST_MODE else [6]
    test_u = [5,7] if not FAST_MODE else [8]
    tr = np.flatnonzero(group.arch_id.isin(train_ids) & group.parameter.isin(train_u))
    va = np.flatnonzero(group.arch_id.isin(val_ids) & group.parameter.isin(val_u))
    te = np.flatnonzero(group.arch_id.isin(test_ids) & group.parameter.isin(test_u))
    assert_architecture_disjoint(group, tr, va)
    assert_architecture_disjoint(group, np.r_[tr,va], te)
    _, pred, info = train_graph_model(group, "joint", tr, va, te, head="mlp",
                                      seed=int(replicate_seed), epochs=60 if FAST_MODE else 180)
    test_frame = group.iloc[te].copy(); test_frame["prediction"] = pred
    per_u = test_frame.groupby("parameter").apply(
        lambda g: kendalltau(g.gap, g.prediction).statistic)
    seed_rows.append({"arch_seed": replicate_seed, "mean_tau": float(per_u.mean()),
                      "min_tau": float(per_u.min()), "val_tau": info["val_tau"],
                      "epochs": info["epochs"], "n_test": len(te)})
    print(seed_rows[-1])
seed_results = pd.DataFrame(seed_rows)
display(seed_results)'''),
        code('''mean_tau, sd_tau, ci = bootstrap_mean_ci(seed_results.mean_tau, n_boot=10000)
report = pd.DataFrame([{"n_architecture_seeds": len(seed_results),
                        "mean_Kendall_tau": mean_tau, "sample_SD": sd_tau,
                        "bootstrap_95_CI_low": ci[0], "bootstrap_95_CI_high": ci[1]}])
display(report)
seed_results.to_csv(output_dir()/"multiseed_tau_by_seed.csv", index=False)
report.to_csv(output_dir()/"multiseed_bootstrap_summary.csv", index=False)'''),
        code('''fig, ax = plt.subplots(figsize=(7,4))
ax.scatter(seed_results.arch_seed.astype(str), seed_results.mean_tau, s=65, color="#008c8c")
ax.axhline(mean_tau, color="#12304a", label=f"mean={mean_tau:.3f}")
ax.fill_between([-0.5, len(seed_results)-0.5], ci[0], ci[1], alpha=.18, color="#008c8c", label="bootstrap 95% CI")
ax.set(xlabel="Architecture-generation seed", ylabel="Mean test Kendall $\\tau$", title="Independent seed robustness")
ax.legend(); ax.grid(alpha=.2); plt.show()''')
    ])


NOTEBOOKS["02_unseen_architecture_and_parameter_test.ipynb"] = notebook(
    "Joint unseen-architecture and unseen-Hamiltonian test",
    "Separates architecture pools and Hubbard parameters, then reports the three generalization quadrants: unseen parameter, unseen architecture, and both unseen simultaneously.",
    [
        code('df = ensure_hubbard_dataset(fast=FAST_MODE)'),
        code('''all_seeds = sorted(df.arch_seed.unique())
arch_train = all_seeds[:max(1,len(all_seeds)-2)]
arch_val, arch_test = [all_seeds[-2]], [all_seeds[-1]]
param_train = [0,1,2,4,6,8] if not FAST_MODE else [0,2,4]
param_val = [3] if not FAST_MODE else [6]
param_test = [5,7] if not FAST_MODE else [8]

train = np.flatnonzero(df.arch_seed.isin(arch_train) & df.parameter.isin(param_train))
val = np.flatnonzero(df.arch_seed.isin(arch_val) & df.parameter.isin(param_val))
quadrants = {
    "unseen U/t only": np.flatnonzero(df.arch_seed.isin(arch_train) & df.parameter.isin(param_test)),
    "unseen architecture only": np.flatnonzero(df.arch_seed.isin(arch_test) & df.parameter.isin(param_train)),
    "both unseen": np.flatnonzero(df.arch_seed.isin(arch_test) & df.parameter.isin(param_test)),
}
test = np.unique(np.concatenate(list(quadrants.values())))
assert not set(train) & set(val) and not set(train) & set(test)
unseen_arch_rows = np.unique(np.r_[quadrants["unseen architecture only"], quadrants["both unseen"]])
assert_architecture_disjoint(df, np.r_[train,val], unseen_arch_rows)
print({"arch_train":arch_train,"arch_val":arch_val,"arch_test":arch_test,
       "U_train":param_train,"U_val":param_val,"U_test":param_test,
       "n_train":len(train),"n_val":len(val),"n_test_union":len(test)})'''),
        code('''_, pred, info = train_graph_model(df, "joint", train, val, test, head="mlp",
                                  seed=2026, epochs=70 if FAST_MODE else 200)
pred_map = dict(zip(test, pred))
rows = []
for name, idx in quadrants.items():
    p = np.array([pred_map[i] for i in idx]); metric = macro_ranking_metrics(df,idx,p)
    rows.append({"split":name,"n":len(idx),"Kendall_tau":metric["test_tau"],
                 "Spearman_rho":metric["test_spearman"],"n_H":metric["n_hamiltonians"]})
quadrant_results = pd.DataFrame(rows)
display(quadrant_results)
quadrant_results.to_csv(output_dir()/"unseen_architecture_parameter_results.csv",index=False)'''),
        code('''fig, ax = plt.subplots(figsize=(7,4))
sns.barplot(data=quadrant_results, x="split", y="Kendall_tau", color="#4fc3b3", ax=ax)
ax.axhline(0,color="black",lw=.8); ax.set(title="Factorial transfer test", xlabel="", ylabel="Kendall $\\tau$")
ax.tick_params(axis="x",rotation=15); plt.show()''')
    ])


NOTEBOOKS["03_vqe_convergence_check.ipynb"] = notebook(
    "High-precision VQE convergence and ranking-stability check",
    "Re-evaluates the leading circuits with 160 optimization steps and five restarts, compares low- and high-budget rankings, and saves convergence traces.",
    [
        code('df = ensure_hubbard_dataset(fast=FAST_MODE)'),
        code('''check_u = sorted(set(df.parameter.unique()) & set([0,4,8]))
top_k = 2 if FAST_MODE else 5
selected = (df[df.parameter.isin(check_u)].sort_values("gap")
            .groupby(["arch_seed","parameter"],as_index=False).head(top_k).copy())
print("High-precision evaluations:",len(selected))
display(selected[["parameter","arch_seed","arch_id","gap"]].head(10))'''),
        code('''hp_rows=[]; example_trace=None; started=time.time()
hp_steps, hp_restarts = (30,2) if FAST_MODE else (160,5)
for count, (_, row) in enumerate(selected.iterrows(),1):
    H,n,initial=hubbard_problem(float(row.parameter))
    energy,trace=evaluate_vqe(H,n,initial,gates_from_json(row.gates_json),
                              steps=hp_steps,restarts=hp_restarts,
                              seed=900000+int(row.arch_seed)*100+int(row.arch_id))
    hp_rows.append({**row.to_dict(),"high_energy":energy,
                    "high_gap":max(0.,energy-float(row.exact)),
                    "high_steps":hp_steps,"high_restarts":hp_restarts})
    if example_trace is None: example_trace=trace
    if count%10==0 or count==len(selected): print(count,"/",len(selected),"elapsed",round(time.time()-started,1),"s")
hp=pd.DataFrame(hp_rows)
hp.to_csv(output_dir()/"vqe_high_precision_check.csv",index=False)'''),
        code('''stability=[]
for key,g in hp.groupby(["arch_seed","parameter"]):
    stability.append({"arch_seed":key[0],"parameter":key[1],"n":len(g),
                      "Kendall_tau":kendalltau(g.gap,g.high_gap).statistic,
                      "top1_unchanged":int(g.sort_values("gap").iloc[0].arch_id==g.sort_values("high_gap").iloc[0].arch_id),
                      "median_gap_change":float(np.median(g.high_gap-g.gap))})
stability=pd.DataFrame(stability)
display(stability)
print("Mean ranking stability:",stability.Kendall_tau.mean())
print("Top-1 agreement:",stability.top1_unchanged.mean())
stability.to_csv(output_dir()/"vqe_ranking_stability.csv",index=False)'''),
        code('''fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].plot(example_trace); axes[0].set(xlabel="Optimization step",ylabel="Energy",title="Representative high-precision trace")
axes[1].scatter(hp.gap,hp.high_gap,alpha=.65); lim=max(hp.gap.max(),hp.high_gap.max()); axes[1].plot([0,lim],[0,lim],'--',color='gray')
axes[1].set(xlabel="Original gap",ylabel="160-step/5-restart gap",title="Label convergence")
plt.tight_layout();plt.show()''')
    ])


NOTEBOOKS["04_trainable_gat_kan_ablation.ipynb"] = notebook(
    "Trainable GAT/KAN Hamiltonian-encoding ablation",
    "Compares circuit-only GAT, Hamiltonian-only GAT, legacy lossy pairwise encoding, lossless Pauli-factor encoding, GAT+KAN, and a depth/parameter-count ridge baseline under one identical leakage-free split.",
    [
        code('df = ensure_hubbard_dataset(fast=FAST_MODE)'),
        code('''seeds=sorted(df.arch_seed.unique())
arch_train,arch_val,arch_test=(seeds[:-2],[seeds[-2]],[seeds[-1]]) if len(seeds)>=3 else ([seeds[0]],[seeds[0]],[seeds[-1]])
u_train=[0,1,2,4,6,8] if not FAST_MODE else [0,2,4]; u_val=[3] if not FAST_MODE else [6]; u_test=[5,7] if not FAST_MODE else [8]
split=split_mask(df,arch_train,arch_val,arch_test,u_train,u_val,u_test)
assert_architecture_disjoint(df,np.r_[split["train"],split["val"]],split["test"])
print({k:len(v) for k,v in split.items()})'''),
        code('''experiments=[("circuit GAT","circuit","mlp"),("Hamiltonian factor GAT (negative control)","hamiltonian","mlp"),
             ("joint pairwise GAT (lossy)","joint_pairwise","mlp"),
             ("joint factor GAT","joint","mlp"),("joint factor GAT+KAN","joint","kan")]
model_seeds=[101] if FAST_MODE else [101,202,303]
rows=[]
for label,mode,head in experiments:
    for model_seed in model_seeds:
        trained,pred,info=train_graph_model(df,mode,split["train"],split["val"],split["test"],
                                            head=head,seed=model_seed,epochs=60 if FAST_MODE else 220)
        rows.append({"model":label,"model_seed":model_seed,
                     "trainable_parameters":sum(p.numel() for p in trained.parameters()),**info})
        print(rows[-1])
_,base=scalar_ridge_baseline(df,split["train"],split["val"],split["test"])
rows.append({"model":"depth/count ridge","model_seed":-1,"trainable_parameters":5,**base,"epochs":0})
ablation=pd.DataFrame(rows); display(ablation)
ablation.to_csv(output_dir()/"gat_kan_ablation.csv",index=False)'''),
        code('''summary=(ablation.groupby("model").test_tau.agg(["mean","std","count"]).sort_values("mean",ascending=False))
display(summary)
fig,ax=plt.subplots(figsize=(9,4)); sns.barplot(data=ablation,x="model",y="test_tau",errorbar="sd",color="#4fc3b3",ax=ax)
sns.stripplot(data=ablation,x="model",y="test_tau",color="#12304a",size=6,ax=ax)
ax.set(xlabel="",ylabel="Held-out Kendall $\\tau$",title="Equal-split trainable-model ablation"); ax.tick_params(axis="x",rotation=18)
plt.tight_layout();plt.show()'''),
        md("**Interpretation rules.** Hamiltonian-only is intentionally a negative control: without circuit information it must assign the same score to all architectures of one Hamiltonian, making Kendall $\\tau$ undefined rather than zero. Attribute a representation benefit only if the joint Pauli-factor model consistently exceeds both circuit-only and joint-pairwise GAT across initialization seeds. GAT+KAN has a larger head, so its comparison is a practical model ablation, not a parameter-matched proof that splines alone cause an improvement.")
    ])


NOTEBOOKS["05_dense_hubbard_sweep_analysis.ipynb"] = notebook(
    "Dense weak-to-strong-correlation Hubbard analysis",
    "Analyzes $U/t=0,1,\\ldots,8$, including adjacent-parameter rank stability, architecture crossovers, and VQE-gap distributions.",
    [
        code('df=ensure_hubbard_dataset(fast=FAST_MODE)'),
        code('''summary=df.groupby("parameter").gap.agg(best="min",median="median",mean="mean",worst="max").reset_index()
display(summary)
pivot=df.pivot_table(index=["arch_seed","arch_id"],columns="parameter",values="gap")
rank_corr=pd.DataFrame(index=pivot.columns,columns=pivot.columns,dtype=float)
for a in pivot.columns:
    for b in pivot.columns: rank_corr.loc[a,b]=kendalltau(pivot[a],pivot[b]).statistic
display(rank_corr.round(2))
summary.to_csv(output_dir()/"dense_hubbard_gap_summary.csv",index=False)
rank_corr.to_csv(output_dir()/"dense_hubbard_rank_correlation.csv")'''),
        code('''fig,axes=plt.subplots(1,2,figsize=(12,4.5))
axes[0].fill_between(summary.parameter,summary.best,summary.worst,alpha=.15,color="#008c8c",label="min–max")
axes[0].plot(summary.parameter,summary["median"],marker="o",color="#12304a",label="median")
axes[0].set(xlabel="$U/t$",ylabel="VQE gap",title="Weak-to-strong correlation sweep");axes[0].legend();axes[0].grid(alpha=.2)
sns.heatmap(rank_corr.astype(float),vmin=-1,vmax=1,cmap="vlag",annot=True,fmt=".2f",ax=axes[1])
axes[1].set(title="Architecture-ranking stability",xlabel="$U/t$",ylabel="$U/t$")
plt.tight_layout();plt.show()'''),
        code('''adjacent=[]
u=sorted(pivot.columns)
for a,b in zip(u[:-1],u[1:]): adjacent.append({"U_left":a,"U_right":b,"Kendall_tau":kendalltau(pivot[a],pivot[b]).statistic})
adjacent=pd.DataFrame(adjacent);display(adjacent)
adjacent.to_csv(output_dir()/"dense_hubbard_adjacent_tau.csv",index=False)''')
    ])


NOTEBOOKS["06_lih_bond_length_transfer.ipynb"] = notebook(
    "LiH bond-length transfer across equilibrium and stretched geometries",
    "Builds six-qubit Jordan–Wigner LiH Hamiltonians in an STO-3G (2e,3o) active space, then tests circuit and joint graph predictors on unseen architectures and unseen bond lengths.",
    [
        md("## Generate molecular labels\n\nPublication mode uses nine bond lengths, five architecture seeds, 18 candidates per seed, 70 VQE steps, and two restarts. This is the most computationally expensive notebook."),
        code('''lih=ensure_lih_dataset(fast=FAST_MODE); display(lih.head())
roundtrip=[]
for bond in sorted(lih.parameter.unique()):
    H,n,_=lih_problem(float(bond))
    roundtrip.append({"bond_A":bond,"max_coefficient_error":validate_factor_graph_roundtrip(H,n)})
roundtrip=pd.DataFrame(roundtrip);display(roundtrip)
assert roundtrip.max_coefficient_error.max() < 1e-10'''),
        code('''seeds=sorted(lih.arch_seed.unique())
arch_train,arch_val,arch_test=(seeds[:-2],[seeds[-2]],[seeds[-1]]) if len(seeds)>=3 else ([seeds[0]],[seeds[0]],[seeds[-1]])
b_train=[1.0,1.4,1.8,2.4,3.2] if not FAST_MODE else [1.0,1.4,2.4]
b_val=[1.2] if not FAST_MODE else [1.8]
b_test=[1.6,2.0,2.8] if not FAST_MODE else [3.2]
split=split_mask(lih,arch_train,arch_val,arch_test,b_train,b_val,b_test)
assert_architecture_disjoint(lih,np.r_[split["train"],split["val"]],split["test"])
print({k:len(v) for k,v in split.items()})'''),
        code('''rows=[]; predictions={}
for label,mode,head in [("circuit GAT","circuit","mlp"),("joint pairwise GAT","joint_pairwise","mlp"),
                        ("joint factor GAT","joint","mlp"),("joint factor GAT+KAN","joint","kan")]:
    _,pred,info=train_graph_model(lih,mode,split["train"],split["val"],split["test"],head=head,
                                  seed=2026,epochs=60 if FAST_MODE else 220)
    rows.append({"model":label,**info}); predictions[label]=pred
results=pd.DataFrame(rows);display(results)
results.to_csv(output_dir()/"lih_transfer_model_summary.csv",index=False)'''),
        code('''test_frame=lih.iloc[split["test"]].copy().reset_index(drop=True)
per_bond=[]
for label,pred in predictions.items():
    test_frame[label]=pred
    for bond,g in test_frame.groupby("parameter"):
        per_bond.append({"model":label,"bond_A":bond,"n":len(g),
                         "Kendall_tau":kendalltau(np.log10(g.gap+1e-8),g[label]).statistic})
per_bond=pd.DataFrame(per_bond);display(per_bond)
per_bond.to_csv(output_dir()/"lih_transfer_by_bond.csv",index=False)'''),
        code('''gap_summary=lih.groupby("parameter").gap.agg(best="min",median="median").reset_index()
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].semilogy(gap_summary.parameter,gap_summary.best,marker="o",label="best")
axes[0].semilogy(gap_summary.parameter,gap_summary["median"],marker="s",label="median")
axes[0].axhline(1.6e-3,color="#d95f59",ls="--",label="1.6 mHa active-space threshold")
axes[0].set(xlabel="Li–H distance (Å)",ylabel="Active-space VQE gap (Ha)",title="Bond-stretch difficulty");axes[0].legend();axes[0].grid(alpha=.2)
sns.lineplot(data=per_bond,x="bond_A",y="Kendall_tau",hue="model",marker="o",ax=axes[1])
axes[1].set(title="Unseen-geometry ranking",xlabel="Held-out Li–H distance (Å)",ylabel="Kendall $\\tau$");axes[1].axhline(0,color="black",lw=.8)
plt.tight_layout();plt.show()'''),
        md("## High-precision audit of the selected molecular finalists"),
        code('''finalists=(test_frame.sort_values("gap").groupby("parameter",as_index=False).head(2 if FAST_MODE else 3))
audit=[]; audit_steps,audit_restarts=((30,2) if FAST_MODE else (160,5))
for _,row in finalists.iterrows():
    H,n,initial=lih_problem(float(row.parameter))
    energy,_=evaluate_vqe(H,n,initial,gates_from_json(row.gates_json),steps=audit_steps,
                          restarts=audit_restarts,seed=700000+int(row.arch_seed)*100+int(row.arch_id))
    audit.append({"bond_A":row.parameter,"arch_seed":row.arch_seed,"arch_id":row.arch_id,
                  "base_gap":row.gap,"high_precision_gap":max(0.,energy-row.exact)})
audit=pd.DataFrame(audit);display(audit)
audit.to_csv(output_dir()/"lih_finalist_convergence_audit.csv",index=False)'''),
        md("The molecular error is measured relative to exact diagonalization **within the same minimal active space**. It is not an error against complete-basis or experimental energies. Canonical molecular orbitals are recomputed independently at each geometry; this pilot does not track orbital identity by overlap. Inspect orbital continuity before attributing a sharp transfer change solely to electronic correlation.")
    ])


for filename, nb in NOTEBOOKS.items():
    path = OUT / filename
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("wrote", path)
