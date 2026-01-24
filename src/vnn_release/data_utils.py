from pathlib import Path
import pandas as pd


def map_and_collapse(df: pd.DataFrame, mapper, *, name: str = "", how: str = "mean") -> pd.DataFrame:
    """
    1) TargetID → gene symbol 매핑
    2) 매핑 실패 row 제거
    3) 같은 drug column 중복 집계
    4) 같은 gene row 중복 집계
    """
    df = df.copy()
    if "TargetID" not in df.columns:
        raise ValueError(f"[{name}] 'TargetID' column not found.")

    df["Gene"] = df["TargetID"].map(mapper)
    df = df.dropna(subset=["Gene"])

    numeric = df.drop(columns=["TargetID", "Gene"]).apply(pd.to_numeric, errors="coerce")

    # 중복 drug 이름 → 집계
    if numeric.columns.duplicated().any():
        numeric = numeric.T.groupby(level=0).agg(how).T

    # 중복 gene → 집계
    numeric.index = df["Gene"].values
    if numeric.index.duplicated().any():
        numeric = numeric.groupby(level=0).agg(how)

    return numeric.sort_index()


def load_vnn_data(raw_dir: Path, mapping_dir: Path) -> dict:
    """
    raw_dir: matrix_output_new/filtered_new 안의 6개 CSV가 위치한 디렉터리
    mapping_dir: activation/inhibition 타깃 매핑 tsv 가 위치한 디렉터리
    반환: vnn_data dict (pos_act/pos_inh/neg_act/neg_inh/val_act/val_inh)
    """
    raw_dir = Path(raw_dir)
    mapping_dir = Path(mapping_dir)

    act_id_map = pd.read_csv(mapping_dir / "activation_targets.tsv", sep="\t")
    inh_id_map = pd.read_csv(mapping_dir / "inhibition_targets.tsv", sep="\t")
    map_act = dict(zip(act_id_map["TargetID"], act_id_map["Gene_symbol"]))
    map_inh = dict(zip(inh_id_map["TargetID"], inh_id_map["Gene_symbol"]))

    pos_act_raw = pd.read_csv(raw_dir / "T2DM_merged_clean_no_phase4_activation_probs.csv")
    pos_inh_raw = pd.read_csv(raw_dir / "T2DM_merged_clean_no_phase4_inhibition_probs.csv")
    neg_act_raw = pd.read_csv(raw_dir / "neg_otherdisease_clean_activation_probs.csv")
    neg_inh_raw = pd.read_csv(raw_dir / "neg_otherdisease_clean_inhibition_probs.csv")
    val_act_raw = pd.read_csv(raw_dir / "T2DM_overlap_with_phase4_activation_probs.csv")
    val_inh_raw = pd.read_csv(raw_dir / "T2DM_overlap_with_phase4_inhibition_probs.csv")

    pos_act = map_and_collapse(pos_act_raw, map_act, name="pos_act")
    pos_inh = map_and_collapse(pos_inh_raw, map_inh, name="pos_inh")
    neg_act = map_and_collapse(neg_act_raw, map_act, name="neg_act")
    neg_inh = map_and_collapse(neg_inh_raw, map_inh, name="neg_inh")
    val_act = map_and_collapse(val_act_raw, map_act, name="val_act")
    val_inh = map_and_collapse(val_inh_raw, map_inh, name="val_inh")

    genes_all = sorted(
        set(pos_act.index)
        | set(pos_inh.index)
        | set(neg_act.index)
        | set(neg_inh.index)
        | set(val_act.index)
        | set(val_inh.index)
    )

    def _reindex(df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(genes_all).apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return {
        "pos_act": _reindex(pos_act),
        "pos_inh": _reindex(pos_inh),
        "neg_act": _reindex(neg_act),
        "neg_inh": _reindex(neg_inh),
        "val_act": _reindex(val_act),
        "val_inh": _reindex(val_inh),
    }


def load_smiles_data(smiles_dir: Path):
    """
    Load SMILES datasets for XGB/LR baselines.
    Expected files in smiles_dir:
      - T2DM_merged_clean_no_phase4.csv      (positives)
      - neg_otherdisease_clean.csv           (negatives)
      - T2DM_overlap_with_phase4.csv         (validation/approved set; optional)
    Returns
      pos_df, neg_df, val_df  (each with at least DrugID, SMILES, label)
    """
    smiles_dir = Path(smiles_dir)
    pos_df = pd.read_csv(smiles_dir / "T2DM_merged_clean_no_phase4.csv")
    neg_df = pd.read_csv(smiles_dir / "neg_otherdisease_clean.csv")
    val_path = smiles_dir / "T2DM_overlap_with_phase4.csv"
    val_df = pd.read_csv(val_path) if val_path.exists() else pd.DataFrame(columns=pos_df.columns)

    if "label" not in pos_df.columns:
        pos_df = pos_df.assign(label=1)
    if "label" not in neg_df.columns:
        neg_df = neg_df.assign(label=0)
    if "label" not in val_df.columns:
        val_df = val_df.assign(label=1)

    def _norm(df):
        df["DrugID"] = df["DrugID"].astype(str)
        return df

    return _norm(pos_df), _norm(neg_df), _norm(val_df)
