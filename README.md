# PolyMoA-VNN

## Overview

**PolyMoA-VNN (Polypharmacological Mechanism-of-Action Visible Neural Network)** is a mechanism-aware phenotypic prediction framework designed to achieve robust generalization to novel chemical scaffolds while preserving biological interpretability.

The model decouples **structural feature learning** from **disease-specific reasoning** by:

1. **Projecting molecular structure into target-level functional scores** using a frozen, pretrained drug–target interaction model (DTIAM), capturing activation and inhibition tendencies across a broad protein space.
2. **Propagating these scores through a biologically constrained Visible Neural Network (VNN)** constructed from curated Reactome pathway hierarchies, explicitly modeling the sequential mechanism of action  
   *(compound → target → pathway → phenotype)*.

By operating on **functional, mechanism-level representations rather than raw chemical similarity**, PolyMoA-VNN mitigates molecular out-of-distribution (MOOD) effects and enables pathway-level interpretability.

This repository provides a **package-ready snapshot** of the training and evaluation pipeline used in the **Type 2 Diabetes (T2D)** experiments reported in the thesis and manuscript.

# DTI_VNN_release

Package-ready snapshot of the **PolyMoA-VNN / DTI-VNN** training pipeline for **Type 2 Diabetes phenotypic prediction**.

The release contains the **minimal data, graph assets, and scripts** required to reproduce both  
**in-distribution (ID) cross-validation** and **scaffold-based out-of-distribution (OOD)** results reported in the paper, without relying on notebook-only analysis code.

## Layout
- `data/raw/`: protein target probability matrices (activation/inhibition) for positive, negative, and Phase 4 overlap validation sets.
- `data/mapping/`: TargetID→gene symbol maps used to collapse the raw matrices.
- `data/reactome/`: Reactome hierarchy files required by the VNN builder.
- `data/splits/`: precomputed ID (k‑fold) and OOD splits from the notebook.
- `data/aux/Diabetes_related_pathway.pkl`: optional skip‑pathway list (unused by default).
- `src/vnn_release/`: cleaned python package (graph builder, masks, training helpers).
- `scripts/train_vnn.py`: CLI entry point for ID cross‑validation on the bundled data.

## Environment setup
```bash
cd DTI_VNN_release
conda create -n dti_vnn python=3.10
conda activate dti_vnn
pip install -U pip setuptools wheel
pip install -r requirements.txt
```
`requirements.txt` pins torch, scikit-learn, xgboost, rdkit-pypi, etc., matching the tested environment.

## How to run
```bash
# 1) Default: 5 seeds (42–46) × 5-fold ID only, all models (polymoa, fc-dnn, randmasked-vnn, XGB, LR)
PYTHONPATH=src python scripts/train_vnn.py

# 2) ID-only with selected models/seeds (custom outputs optional)
PYTHONPATH=src python scripts/train_vnn.py \
  --seeds 42 43 44 \
  --models polymoa fc xgb \
  --depth 4 \
  --output outputs/id_metrics.csv

# models option
#   polymoa : PolyMoA-VNN (graph VNN)
#   fc      : FC-DNN (dense baseline)
#   rand    : RandMasked-VNN (random masks)
#   xgb     : XGBoost (SMILES→Morgan FP)
#   lr      : Logistic Regression (SMILES→Morgan FP)

# 3) ID+OOD: run ID then OOD (custom outputs optional)
PYTHONPATH=src python scripts/train_vnn.py \
  --run-ood \
  --output outputs/id_metrics.csv \
  --ood-output outputs/ood_metrics.csv

# 4) OOD-only: skip ID, run OOD split only
PYTHONPATH=src python scripts/train_vnn.py \
  --ood-only \
  --seeds 42 43 \
  --models polymoa xgb \
  --depth 4 \
  --ood-output outputs/ood_metrics_custom.csv
```

Outputs: `outputs/id_metrics.csv` (seed × fold) and `outputs/ood_metrics.csv` (seed) contain AUROC/AUPR; mean values are printed to console.

GPU vs CPU: torch models (polymoa, fc, rand) are much faster on GPU; full ID+OOD (all models, 5 seeds) typically takes a few hours on a V100-class GPU, but can take many times longer on CPU. XGB/LR run fine on CPU. Device is auto-picked via `torch.cuda.is_available()`.

## Notes and assumptions
- Models reproduced from the notebook: Polymoa-VNN(base), FC-DNN, RandMasked-VNN (gene-channel inputs), and XGB/LR (SMILES→MorganFP). ChemBERTa and interpretation plots were omitted.
- Reactome assets are loaded relative to this repository (no absolute paths).
- Negative pools are resampled per fold exactly as in the notebook; splits are read from `data/splits/fold_splits.pkl`.
- Skip/skip-reg variants were removed to align with the slim release. If you want skip IDs, extend `build_polymoa_model`.

## External inference (DTIAM → VNN inputs)
BerMol+AutoGluon inference results from DTIAM (`t2d_*_probs.csv`) are not redistributed for licensing reasons. If needed, follow `docs/DTIAM_inference.md`:
1) Run `DTIAM/code/inference_mydata_lots.py` (activation or inhibition) → generates `t2d_pos/neg/val_*_probs.csv` under `DTI_VNN/DTI_OUTPUT/Other_diseases/`.
2) Optionally replace `data/raw/` with regenerated probabilities, then rerun `PYTHONPATH=src python scripts/train_vnn.py` to reproduce ID/OOD scores.
3) DTIAM models/data are external assets and are not included here; only paths/procedure are documented.

## Data provenance
- Raw matrices originate from `DTI_INPUT/Other_diseases/T2D_chembl/matrix_output_new/filtered_new/` in the original project.
- Target maps come from `DTIAM/data/moa/*/tar_gene copy.csv`.
- Reactome files were copied from `Pathway_Reactome/ReactomePathwaysRelation.txt` and `ReactomePathways.gmt`.

## Minimal API surface
- `vnn_release.data_utils.load_vnn_data(raw_dir, mapping_dir)`: loads & cleans the raw matrices into a `vnn_data` dict.
- `vnn_release.data_utils.load_smiles_data(smiles_dir)`: loads SMILES CSVs for XGB/LR.
- `vnn_release.preprocess.build_scores_from_unified_ids(...)`: robust+min-max scaling and ACT/INH channel expansion.
- `vnn_release.training.run_all_id_models(...)`: runs the ID cross-validation loop for all baselines and returns metric records.
- `vnn_release.training.run_all_ood_models(...)`: runs the OOD evaluation loop for all baselines.
