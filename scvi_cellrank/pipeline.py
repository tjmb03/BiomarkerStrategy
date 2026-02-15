# file: simulate_scvi_cellrank_workflow.py
"""
End-to-end, self-contained demo:
1) Simulate scRNA-seq counts with donor batches + pseudotime.
2) Fit scVI (latent only).
3) Build 2 kNN graphs (PCA vs scVI), ensuring graph is (nearly) connected.
4) Build CellRank transitions twice (PCA-graph vs scVI-graph), run GPCCA, set terminal macrostates robustly.
5) Compare absorption probabilities across runs (cellwise corr, JS, assignment match, macro NMI).
6) Plot:
   - Violin of per-cell corr
   - Heatmap of terminal mass deltas
   - UMAP overlay of instability (1 - corr) + low-stability mask
Outputs:
- {outdir}/synthetic.h5ad
- {outdir}/out.h5ad
- {outdir}/robustness_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot

import matplotlib.pyplot as plt

import scanpy as sc
import scvi
import cellrank as cr
from cellrank.kernels import ConnectivityKernel, PseudotimeKernel


# -----------------------------
# Reproducibility
# -----------------------------
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# -----------------------------
# Synthetic data generation
# -----------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def nb_counts(mu: np.ndarray, theta: float, rng: np.random.Generator) -> np.ndarray:
    """
    Negative binomial sampling parameterized by mean mu and dispersion theta.
    Var = mu + mu^2/theta
    """
    mu = np.clip(mu, 1e-12, None)
    p = theta / (theta + mu)
    n = theta
    return rng.negative_binomial(n=n, p=p, size=mu.shape)


def simulate_anndata(
    *,
    n_cells: int,
    n_genes: int,
    n_donors: int,
    n_branches: int,
    seed: int,
) -> sc.AnnData:
    rng = np.random.default_rng(seed)

    donors = rng.integers(0, n_donors, size=n_cells)
    donor_names = np.array([f"donor{d+1}" for d in donors], dtype=object)

    pt = rng.uniform(0, 1, size=n_cells)

    logits = rng.normal(0, 0.5, size=(n_cells, n_branches))
    for b in range(n_branches):
        logits[:, b] += (b - (n_branches - 1) / 2.0) * (pt - 0.5) * 3.0
    fate = softmax(logits, axis=1).argmax(axis=1)
    fate_names = np.array([f"T{f+1}" for f in fate], dtype=object)

    base = rng.normal(1.5, 0.4, size=n_genes)
    time_prog = rng.normal(0, 1.0, size=n_genes)
    time_prog = time_prog / (np.std(time_prog) + 1e-12)

    branch_prog = rng.normal(0, 1.0, size=(n_branches, n_genes))
    branch_prog = branch_prog / (np.std(branch_prog, axis=1, keepdims=True) + 1e-12)

    donor_eff = rng.normal(0, 0.25, size=(n_donors, n_genes))

    lib = rng.lognormal(mean=9.5, sigma=0.4, size=n_cells)
    lib = lib / np.median(lib)

    a = 0.8
    b = 0.7
    log_mu = (
        base[None, :]
        + a * (pt[:, None] - 0.5) * time_prog[None, :]
        + b * branch_prog[fate, :]
        + donor_eff[donors, :]
        + rng.normal(0, 0.15, size=(n_cells, n_genes))
    )
    mu = np.exp(log_mu) * lib[:, None]

    X = nb_counts(mu, theta=10.0, rng=rng).astype(np.int32)

    adata = sc.AnnData(X=X)
    adata.obs_names = [f"cell{i:06d}" for i in range(n_cells)]
    adata.var_names = [f"gene{i:04d}" for i in range(n_genes)]

    adata.obs["donor"] = pd.Categorical(donor_names)
    adata.obs["pt"] = pt.astype(float)
    adata.obs["terminal_truth"] = pd.Categorical(fate_names)

    stage = np.where(pt < 0.33, "early", np.where(pt < 0.66, "mid", "late"))
    adata.obs["cluster_truth"] = pd.Categorical([f"{s}_{t}" for s, t in zip(stage, fate_names)])

    return adata


# -----------------------------
# Graph utilities
# -----------------------------
def count_components_from_connectivities(conn) -> int:
    g = conn.tocsr() if sp.issparse(conn) else sp.csr_matrix(conn)
    return int(connected_components(g, directed=False, return_labels=False))


def ensure_connected_neighbors(
    adata: sc.AnnData,
    *,
    use_rep: str,
    start_k: int,
    max_k: int,
    target_components: int,
    random_state: int,
) -> Tuple[int, int]:
    k = start_k
    while True:
        sc.pp.neighbors(adata, n_neighbors=k, use_rep=use_rep, random_state=random_state)
        n_comp = count_components_from_connectivities(adata.obsp["connectivities"])
        if n_comp <= target_components or k >= max_k:
            return k, n_comp
        k = min(max_k, int(k * 1.5))


# -----------------------------
# CellRank robustness core
# -----------------------------
@dataclass(frozen=True)
class CellRankRunConfig:
    name: str
    connectivities_key: str
    time_key: str = "pt"
    w_dir: float = 0.7
    w_conn: float = 0.3
    n_terminal: int = 3
    random_state: int = 0


class ConnectivitiesSwap:
    def __init__(self, adata: sc.AnnData, connectivities_key: str):
        self.adata = adata
        self.connectivities_key = connectivities_key
        self._backup = None

    def __enter__(self):
        if self.connectivities_key not in self.adata.obsp:
            raise KeyError(
                f"Missing adata.obsp['{self.connectivities_key}']. "
                f"Available: {list(self.adata.obsp.keys())}"
            )
        self._backup = self.adata.obsp.get("connectivities", None)
        self.adata.obsp["connectivities"] = self.adata.obsp[self.connectivities_key]
        return self.adata

    def __exit__(self, exc_type, exc, tb):
        if self._backup is None:
            self.adata.obsp.pop("connectivities", None)
        else:
            self.adata.obsp["connectivities"] = self._backup
        return False


def rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0 = a - a.mean(axis=1, keepdims=True)
    b0 = b - b.mean(axis=1, keepdims=True)
    num = np.sum(a0 * b0, axis=1)
    den = np.sqrt(np.sum(a0 * a0, axis=1) * np.sum(b0 * b0, axis=1))
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den != 0)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


def pick_terminal_macrostates_by_pseudotime(
    adata: sc.AnnData,
    *,
    macrostate_series: pd.Series,
    pt_key: str,
    n_terminal: int,
) -> List[str]:
    pt = adata.obs[pt_key].to_numpy(dtype=float)
    macro = macrostate_series.astype("object")

    df = pd.DataFrame({"pt": pt, "macro": macro}, index=adata.obs_names)
    df = df.dropna(subset=["macro"]).copy()
    df["macro"] = df["macro"].astype(str)

    means = df.groupby("macro")["pt"].mean().sort_values(ascending=False)
    if len(means) < n_terminal:
        raise ValueError(f"Only {len(means)} non-NaN macrostates available, need {n_terminal}.")
    chosen = list(means.index[:n_terminal])
    print(
        f"[terminal macrostates by pt] {chosen} "
        f"(mean pt: {means.iloc[:n_terminal].round(3).to_dict()})"
    )
    return chosen

def compute_fates(g) -> None:
    """CellRank 2.x uses compute_fate_probabilities; older versions use compute_absorption_probabilities."""
    if hasattr(g, "compute_fate_probabilities"):
        g.compute_fate_probabilities()
        return
    if hasattr(g, "compute_absorption_probabilities"):
        g.compute_absorption_probabilities()
        return
    raise AttributeError("GPCCA estimator has no fate/absorption probability computation method.")


def get_fates_matrix_and_names(g) -> Tuple[np.ndarray, List[str]]:
    """
    Return (X, colnames) for fate/absorption probabilities.
    Works for CellRank 2.x and older variants.
    """
    if hasattr(g, "fate_probabilities") and g.fate_probabilities is not None:
        fp = g.fate_probabilities
    elif hasattr(g, "absorption_probabilities") and g.absorption_probabilities is not None:
        fp = g.absorption_probabilities
    else:
        raise AttributeError("GPCCA estimator has no stored fate/absorption probabilities.")

    # DataFrame
    if isinstance(fp, pd.DataFrame):
        X = fp.to_numpy()
        cols = list(map(str, fp.columns))
        return X, cols

    # numpy-like
    X = np.asarray(fp)
    cols = list(map(str, getattr(fp, "names", [])))  # Lineage often has .names
    if not cols:
        cols = [str(i) for i in range(X.shape[1])]
    return X, cols



def run_cellrank_gpcca(
    adata: sc.AnnData,
    cfg: CellRankRunConfig,
    *,
    store_key_prefix: str = "cellrank_robust",
) -> Dict[str, Any]:
    set_seeds(cfg.random_state)

    if cfg.w_dir + cfg.w_conn <= 0:
        raise ValueError("w_dir + w_conn must be > 0")
    w_dir = cfg.w_dir / (cfg.w_dir + cfg.w_conn)
    w_conn = cfg.w_conn / (cfg.w_dir + cfg.w_conn)

    with ConnectivitiesSwap(adata, cfg.connectivities_key):
        # kernels
        k_conn = ConnectivityKernel(adata).compute_transition_matrix()
        k_dir = PseudotimeKernel(adata, time_key=cfg.time_key).compute_transition_matrix()
        k = (w_dir * k_dir) + (w_conn * k_conn)

        g = cr.estimators.GPCCA(k)
        g.compute_schur()  # CellRank 2.0.7: no random_state kwarg

        # GPCCA cannot cluster into fewer macrostates than disconnected components.
        n_comp = count_components_from_connectivities(adata.obsp["connectivities"])
        n_states = max(cfg.n_terminal, n_comp)
        print(f"[gpcca:{cfg.name}] requested_terminal={cfg.n_terminal}, components={n_comp} => n_states={n_states}")

        g.compute_macrostates(n_states=n_states)

        macro_ser = pd.Series(pd.Categorical(g.macrostates), index=adata.obs_names, name="macrostates")

        term_macro_names = pick_terminal_macrostates_by_pseudotime(
            adata,
            macrostate_series=macro_ser,
            pt_key=cfg.time_key,
            n_terminal=cfg.n_terminal,
        )

        g.set_terminal_states(term_macro_names)
        compute_fates(g)

        X_fates, cols = get_fates_matrix_and_names(g)

        abs_df = pd.DataFrame(X_fates, index=adata.obs_names, columns=cols)

# Store
        uns_key = f"{store_key_prefix}:{cfg.name}"
        adata.uns[uns_key] = {"config": cfg.__dict__, "terminal_states": list(abs_df.columns)}
        adata.obsm[f"{uns_key}:abs_probs"] = abs_df.to_numpy()
        adata.uns[f"{uns_key}:abs_probs_cols"] = list(abs_df.columns)
        adata.obs[f"{uns_key}:macrostates"] = macro_ser.astype("category")


    return {"g": g, "abs_probs": abs_df, "macrostates": macro_ser, "uns_key": uns_key}


def compare_runs(
    adata: sc.AnnData,
    run_a: Dict[str, Any],
    run_b: Dict[str, Any],
    *,
    label_a: str,
    label_b: str,
    store_key: str = "cellrank_robust:comparison",
) -> Dict[str, Any]:
    abs_a: pd.DataFrame = run_a["abs_probs"].loc[adata.obs_names]
    abs_b: pd.DataFrame = run_b["abs_probs"].loc[adata.obs_names]

    abs_a: pd.DataFrame = run_a["abs_probs"]
    abs_b: pd.DataFrame = run_b["abs_probs"]

# Align on shared index (should match obs_names now, but safe)
    common_cells = abs_a.index.intersection(abs_b.index).intersection(adata.obs_names)
    abs_a = abs_a.loc[common_cells]
    abs_b = abs_b.loc[common_cells]

    common = [c for c in abs_a.columns if c in set(abs_b.columns)]
    if not common:
        raise ValueError("No overlapping terminal states between runs.")
    abs_a = abs_a[common]
    abs_b = abs_b[common]

    A = abs_a.to_numpy()
    B = abs_b.to_numpy()
    cell_corr = rowwise_pearson(A, B)

    t_a = abs_a.idxmax(axis=1)
    t_b = abs_b.idxmax(axis=1)

    js_overall = float(js_divergence(abs_a.sum(axis=0).to_numpy(), abs_b.sum(axis=0).to_numpy()))
    js_per_state = {c: float(js_divergence(abs_a[c].to_numpy(), abs_b[c].to_numpy())) for c in common}

    macro_a = run_a["macrostates"].loc[adata.obs_names].astype(str)
    macro_b = run_b["macrostates"].loc[adata.obs_names].astype(str)

    try:
        from sklearn.metrics import normalized_mutual_info_score

        macro_nmi = float(normalized_mutual_info_score(macro_a, macro_b))
    except Exception:
        macro_nmi = float("nan")

    report = {
        "terminal_states_common": common,
        "mean_cellwise_absprob_corr": float(np.nanmean(cell_corr)),
        "median_cellwise_absprob_corr": float(np.nanmedian(cell_corr)),
        "terminal_assignment_match_rate": float((t_a == t_b).mean()),
        "macrostate_match_rate": float((macro_a == macro_b).mean()),
        "macrostate_nmi": macro_nmi,
        "js_overall_terminal_mass": js_overall,
        "js_per_state": js_per_state,
    }

    adata.uns[store_key] = report
    adata.obs[f"{store_key}:terminal_{label_a}"] = t_a.astype("category")
    adata.obs[f"{store_key}:terminal_{label_b}"] = t_b.astype("category")
    adata.obs[f"{store_key}:absprob_corr"] = cell_corr

    return report


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_absprob_corr_violin(
    adata: sc.AnnData,
    *,
    comparison_key: str,
    out_path: str,
) -> None:
    col = f"{comparison_key}:absprob_corr"
    corr = adata.obs[col].to_numpy(dtype=float)
    corr = corr[np.isfinite(corr)]

    fig, ax = plt.subplots()
    ax.violinplot([corr], showmeans=True, showmedians=True, widths=0.8)
    ax.set_xticks([1])
    ax.set_xticklabels(["cells"])
    ax.set_ylabel("Pearson r")
    ax.set_title("Cell-wise fate-prob correlation (PCA vs scVI)")
    ax.set_ylim(-1.05, 1.05)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_terminal_mass_delta_heatmap(
    adata: sc.AnnData,
    *,
    run_a_key: str,
    run_b_key: str,
    out_path: str,
    normalize: str = "fraction",
) -> None:
    Xa = np.asarray(adata.obsm[f"{run_a_key}:abs_probs"], dtype=float)
    Xb = np.asarray(adata.obsm[f"{run_b_key}:abs_probs"], dtype=float)
    ca = list(map(str, adata.uns[f"{run_a_key}:abs_probs_cols"]))
    cb = list(map(str, adata.uns[f"{run_b_key}:abs_probs_cols"]))

    common = [c for c in ca if c in set(cb)]
    ia = [ca.index(c) for c in common]
    ib = [cb.index(c) for c in common]

    ma = Xa.sum(axis=0)[ia]
    mb = Xb.sum(axis=0)[ib]

    if normalize == "fraction":
        ma = ma / max(ma.sum(), 1e-12)
        mb = mb / max(mb.sum(), 1e-12)
    elif normalize != "sum":
        raise ValueError("normalize must be 'sum' or 'fraction'")

    delta = (mb - ma).reshape(1, -1)

    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(common)), 2.4))
    im = ax.imshow(delta, aspect="auto")
    ax.set_yticks([0])
    ax.set_yticklabels([normalize])
    ax.set_xticks(np.arange(len(common)))
    ax.set_xticklabels(common, rotation=45, ha="right")
    ax.set_title("Terminal fate mass delta (scVI - PCA)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="delta")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ensure_umap_coords(
    adata: sc.AnnData,
    *,
    use_rep: str,
    random_state: int = 0,
) -> None:
    if "X_umap" in adata.obsm:
        return
    sc.pp.neighbors(adata, n_neighbors=30, use_rep=use_rep, random_state=random_state)
    sc.tl.umap(adata, random_state=random_state)


def plot_umap_instability(
    adata: sc.AnnData,
    *,
    comparison_key: str,
    corr_threshold: float,
    umap_use_rep: str,
    out_path_instability: str,
    out_path_low: str,
    random_state: int = 0,
    s: float = 5.0,
    alpha: float = 0.8,
) -> None:
    col = f"{comparison_key}:absprob_corr"
    corr = adata.obs[col].to_numpy(dtype=float)
    instability = 1.0 - corr
    low = corr < corr_threshold

    adata.obs["cellrank_robust:instability"] = instability
    adata.obs["cellrank_robust:low_stability"] = low

    ensure_umap_coords(adata, use_rep=umap_use_rep, random_state=random_state)
    xy = np.asarray(adata.obsm["X_umap"], dtype=float)

    # Continuous instability
    fig1, ax1 = plt.subplots()
    sc1 = ax1.scatter(xy[:, 0], xy[:, 1], c=instability, s=s, alpha=alpha)
    ax1.set_title("Instability (1 - fate-prob corr)")
    ax1.set_xlabel("UMAP1")
    ax1.set_ylabel("UMAP2")
    fig1.colorbar(sc1, ax=ax1, label="instability")
    fig1.savefig(out_path_instability, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # Low-stability mask
    fig2, ax2 = plt.subplots()
    sc2 = ax2.scatter(xy[:, 0], xy[:, 1], c=low.astype(float), s=s, alpha=alpha)
    ax2.set_title(f"Low-stability cells (corr < {corr_threshold})")
    ax2.set_xlabel("UMAP1")
    ax2.set_ylabel("UMAP2")
    fig2.colorbar(sc2, ax=ax2, label="low_stability (0/1)")
    fig2.savefig(out_path_low, dpi=200, bbox_inches="tight")
    plt.close(fig2)



# -----------------------------
# Workflow
# -----------------------------
def run_workflow(
    *,
    outdir: str,
    n_cells: int,
    n_genes: int,
    n_donors: int,
    n_branches: int,
    n_latent: int,
    seed: int,
    max_epochs: int,
    neighbor_start_k: int,
    neighbor_max_k: int,
    neighbor_target_components: int,
    n_terminal: int,
    corr_threshold: float,
    use_mps: bool,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    set_seeds(seed)

    adata = simulate_anndata(
        n_cells=n_cells,
        n_genes=n_genes,
        n_donors=n_donors,
        n_branches=n_branches,
        seed=seed,
    )
    in_path = os.path.join(outdir, "synthetic.h5ad")
    adata.write_h5ad(in_path)

    scvi.model.SCVI.setup_anndata(adata, batch_key="donor")
    m = scvi.model.SCVI(adata, n_latent=n_latent)

    train_kwargs: Dict[str, Any] = {"max_epochs": max_epochs}
    if use_mps:
        train_kwargs["accelerator"] = "mps"

    m.train(**train_kwargs)
    adata.obsm["X_scVI"] = m.get_latent_representation()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(3000, n_genes))
    sc.pp.pca(adata, n_comps=50)

    k_used, n_comp = ensure_connected_neighbors(
        adata,
        use_rep="X_pca",
        start_k=neighbor_start_k,
        max_k=neighbor_max_k,
        target_components=neighbor_target_components,
        random_state=seed,
    )
    adata.obsp["connectivities_pca"] = adata.obsp["connectivities"].copy()
    print(f"[neighbors] PCA: k={k_used}, components={n_comp}")

    k_used, n_comp = ensure_connected_neighbors(
        adata,
        use_rep="X_scVI",
        start_k=neighbor_start_k,
        max_k=neighbor_max_k,
        target_components=neighbor_target_components,
        random_state=seed,
    )
    adata.obsp["connectivities_scvi"] = adata.obsp["connectivities"].copy()
    print(f"[neighbors] scVI: k={k_used}, components={n_comp}")

    cfg_pca = CellRankRunConfig(
        name="pca",
        connectivities_key="connectivities_pca",
        time_key="pt",
        w_dir=0.7,
        w_conn=0.3,
        n_terminal=n_terminal,
        random_state=seed,
    )
    cfg_scvi = CellRankRunConfig(
        name="scvi",
        connectivities_key="connectivities_scvi",
        time_key="pt",
        w_dir=0.7,
        w_conn=0.3,
        n_terminal=n_terminal,
        random_state=seed,
    )

    run_pca = run_cellrank_gpcca(adata, cfg_pca)
    run_scvi = run_cellrank_gpcca(adata, cfg_scvi)

    comparison_key = "cellrank_robust:comparison"
    report = compare_runs(adata, run_pca, run_scvi, label_a="pca", label_b="scvi", store_key=comparison_key)

    out_path = os.path.join(outdir, "out.h5ad")
    adata.write_h5ad(out_path)

    report_path = os.path.join(outdir, "robustness_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Robustness summary ===")
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"  - {kk}: {vv:.4f}")
        else:
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    comparison_key = "cellrank_robust:comparison"

    plot_absprob_corr_violin(
    adata,
    comparison_key=comparison_key,
    out_path=os.path.join(plots_dir, "violin_absprob_corr.png"),
)

    plot_terminal_mass_delta_heatmap(
    adata,
    run_a_key="cellrank_robust:pca",
    run_b_key="cellrank_robust:scvi",
    normalize="fraction",
    out_path=os.path.join(plots_dir, "heatmap_terminal_mass_delta.png"),
)

    plot_umap_instability(
    adata,
    comparison_key=comparison_key,
    corr_threshold=corr_threshold,
    umap_use_rep="X_scVI",
    out_path_instability=os.path.join(plots_dir, "umap_instability.png"),
    out_path_low=os.path.join(plots_dir, "umap_low_stability.png"),
    random_state=seed,
)


    plots_dir = os.path.join(outdir, "plots")
    plot_files = [
    os.path.join(plots_dir, "violin_absprob_corr.png"),
    os.path.join(plots_dir, "heatmap_terminal_mass_delta.png"),
    os.path.join(plots_dir, "umap_instability.png"),
    os.path.join(plots_dir, "umap_low_stability.png"),
]

    print(
    "\nWrote:\n"
    f"- {in_path}\n"
    f"- {out_path}\n"
    f"- {report_path}\n"
    f"- {plots_dir}/\n"
    + "".join([f"  - {p}\n" for p in plot_files])
)



def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate scVI + CellRank robustness workflow (PCA graph vs scVI graph).")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--n-cells", type=int, default=2000)
    ap.add_argument("--n-genes", type=int, default=2000)
    ap.add_argument("--n-donors", type=int, default=4)
    ap.add_argument("--n-branches", type=int, default=3)
    ap.add_argument("--n-latent", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--neighbor-start-k", type=int, default=30)
    ap.add_argument("--neighbor-max-k", type=int, default=200)
    ap.add_argument("--neighbor-target-components", type=int, default=1)
    ap.add_argument("--n-terminal", type=int, default=3)
    ap.add_argument("--corr-threshold", type=float, default=0.7)
    ap.add_argument("--use-mps", action="store_true", help="Use Apple MPS accelerator for scVI training.")

    args = ap.parse_args()

    run_workflow(
        outdir=args.outdir,
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_donors=args.n_donors,
        n_branches=args.n_branches,
        n_latent=args.n_latent,
        seed=args.seed,
        max_epochs=args.max_epochs,
        neighbor_start_k=args.neighbor_start_k,
        neighbor_max_k=args.neighbor_max_k,
        neighbor_target_components=args.neighbor_target_components,
        n_terminal=args.n_terminal,
        corr_threshold=args.corr_threshold,
        use_mps=args.use_mps,
    )


if __name__ == "__main__":
    main()
