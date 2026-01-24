# Using DTIAM inference outputs with DTI_VNN

## What this covers
- External inference script: `DTIAM/code/inference_mydata_lots.py` (BerMol + AutoGluon from a published paper; models/data not redistributed here).
- Outputs it generates: drug–target probability matrices saved to `DTI_VNN/DTI_OUTPUT/Other_diseases/` (e.g., `t2d_pos_act_probs.csv`, `t2d_pos_inh_probs.csv`, `t2d_neg_act_probs.csv`, `t2d_neg_inh_probs.csv`, `t2d_val_pos_*_probs.csv`).
- How those outputs were used in the original notebook `vnn_train_type2diabetes.ipynb` (cells near the top load these CSVs to build `alz_pos_act_probs`, `alz_pos_inh_probs`, `alz_neg_act_probs`, `alz_neg_inh_probs`).

## Why not shipped here
DTIAM models and cached features are from another published work and are not owned by this project. To avoid licensing/redistribution issues, only the inference script path and usage steps are documented; you must supply the pretrained models/features yourself.

## Reproducing the inference (summary)
1) Requirements (per the DTIAM project): BerMol + ESM-2 + AutoGluon; cached embeddings under `${root}/data/moa/<task>/features/compound_features.pkl` and `protein_features.pkl`; pretrained AutoGluon predictors under `${root}/code/AutogluonModels/<task>`.
2) Default inputs (see script defaults):
   - Drugs: `DTI_VNN/DTI_INPUT/Other_diseases/chembl_t2d_manual.csv` (DrugID, SMILES).
   - Protein panels: `DTI_VNN/DTI_INPUT/tar_seq_act_added.csv` or `tar_seq_inh_added.csv`.
3) Run (example):
   ```bash
   cd DTIAM/code
   python inference_mydata_lots.py --task activation  # or --task inhibition
   # outputs to DTI_VNN/DTI_OUTPUT/Other_diseases/<drug_stem>_activation_probs.csv
   ```
   You can override `--drug-csv`, `--prot-csv`, `--model-path`, `--features-dir`, `--output`.

## Connecting to the VNN pipeline
- The generated `t2d_*_probs.csv` files feed into the VNN preprocessing in `vnn_train_type2diabetes.ipynb` (initial data-loading cells). They become the positive/negative activation/inhibition matrices that are later scaled, merged, and split for ID/OOD training.
- In `DTI_VNN_release`, the downstream matrices derived from those probabilities are already included under `data/raw/`. If you regenerate the raw probabilities, you can swap them in and re-run `scripts/train_vnn.py`.

## Minimal provenance note
- Source script: `DTIAM/code/inference_mydata_lots.py` (user-authored glue over BerMol/AutoGluon; underlying models belong to the published DTIAM work).
- Outputs used here: `DTI_VNN/DTI_OUTPUT/Other_diseases/t2d_*_probs.csv` (not redistributed; referenced by path).
