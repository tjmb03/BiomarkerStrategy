# scVI → Graph Sensitivity → CellRank (GPCCA) Robustness Pipeline

Location: `pipelines/scvi_cellrank/`

This pipeline demonstrates an **auditable robustness** check for fate inference by running **CellRank GPCCA twice** using two alternative neighbor graphs (**PCA** vs **scVI latent**) and comparing the resulting **fate (absorption) probabilities**.

## What it does

1. **Simulates** scRNA-seq counts with **donor/batch** effects and **pseudotime**
2. Fits **scVI** to learn a latent representation (`X_scVI`)
3. Builds two kNN graphs:
   - **PCA-based** graph (`connectivities_pca`)
   - **scVI-latent** graph (`connectivities_scvi`)
4. Runs **CellRank GPCCA** twice (one per graph)
5. Computes robustness metrics:
   - per-cell fate-prob correlation
   - terminal mass deltas / JS divergence
   - assignment agreement
6. Saves **non-interactive** PNG plots (no GUI blocking)

---

## Contents

- `pipeline.py` — main runnable pipeline (recommended entrypoint)
- `simulate_scvi_cellrank_workflow.py` — legacy/alternate version (optional)
- `results/` — generated outputs (should be gitignored)

---

## Requirements

Tested with:
- `scanpy==1.11.5`
- `scvi-tools==1.4.1`
- `cellrank==2.0.7`
- `python==3.11`

Install (conda + pip recommended):

```bash
conda create -n biomarker-scvi -c conda-forge python=3.11 -y
conda activate biomarker-scvi
pip install scanpy==1.11.5 scvi-tools==1.4.1 cellrank==2.0.7 anndata numpy pandas scipy matplotlib scikit-learn
```

Optional (faster GPCCA; avoids dense fallback warnings):

```bash
conda install -c conda-forge petsc4py slepc4py -y
```

---

## Run

From repo root:

```bash
python pipelines/scvi_cellrank/pipeline.py --outdir pipelines/scvi_cellrank/results --n-cells 2000 --n-genes 2000 --max-epochs 20 --seed 0
```

From inside the folder:

```bash
cd pipelines/scvi_cellrank
python pipeline.py --outdir results --n-cells 2000 --n-genes 2000 --max-epochs 20 --seed 0
```

### Smoke test (fast)

```bash
python pipelines/scvi_cellrank/pipeline.py --outdir pipelines/scvi_cellrank/results --n-cells 500 --n-genes 500 --max-epochs 2 --seed 0
```

---

## Outputs

After a successful run, you should see:

- `results/synthetic.h5ad` — simulated input AnnData
- `results/out.h5ad` — AnnData containing:
  - `obsm["X_scVI"]` (scVI latent)
  - `obsp["connectivities_pca"]`, `obsp["connectivities_scvi"]`
  - CellRank macrostates + fate probabilities stored under `cellrank_robust:*`
- `results/robustness_report.json` — robustness summary metrics
- `results/plots/` — saved PNGs (non-interactive)
  - `violin_absprob_corr.png`
  - `heatmap_terminal_mass_delta.png`
  - `umap_instability.png`
  - `umap_low_stability.png`

---

## Interpreting the robustness report

Key fields in `robustness_report.json`:

- `mean_cellwise_absprob_corr`, `median_cellwise_absprob_corr`  
  Per-cell Pearson correlation between fate-prob vectors across runs.  
  **Closer to 1.0 = more robust.**

- `js_overall_terminal_mass` and `js_per_state`  
  Jensen–Shannon divergence between terminal mass distributions.  
  **Closer to 0.0 = more similar.**

- `terminal_assignment_match_rate`  
  Fraction of cells whose argmax terminal fate matches across runs.

- Macrostate agreement (`macrostate_match_rate`, `macrostate_nmi`)  
  Macrostates can differ even if fate probabilities remain stable.

> Note (CellRank 2.x): “absorption probabilities” are exposed as **fate probabilities**.

---

## Common warnings

- **“Unable to import petsc4py/slepc4py… using method='brandts' … densifying”**  
  Install PETSc/SLEPc (see optional install) for faster, memory-efficient Schur computation.

- **“kNN graph is disconnected”**  
  Can happen depending on simulation and kNN parameters. The pipeline handles connected components by ensuring GPCCA uses at least as many macrostates as components.

---

## Git hygiene

Avoid committing large outputs:

Add to repo `.gitignore`:

```gitignore
pipelines/scvi_cellrank/results/
results/
*.h5ad
```

To showcase results, copy PNGs into a lightweight `assets/` folder and commit those:

```bash
mkdir -p assets/scvi_cellrank
cp pipelines/scvi_cellrank/results/plots/*.png assets/scvi_cellrank/
```

Embed in your main README:

```markdown
![Violin](assets/scvi_cellrank/violin_absprob_corr.png)
![Heatmap](assets/scvi_cellrank/heatmap_terminal_mass_delta.png)
![UMAP Instability](assets/scvi_cellrank/umap_instability.png)
```
