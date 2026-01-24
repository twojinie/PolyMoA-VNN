from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Tuple
from vnn_release.modules.vnn_utils import build_vnn_masks
from vnn_release.modules.vnn_utils import build_skip_mask  # re-export convenience

GENE_CHANNEL_TAGS = ("ACT", "INH")


def _tag_gene_channel(gene: str, tag: str) -> str:
    return f"{gene}::{tag}"


def expand_act_inh_channels(act_df: pd.DataFrame, inh_df: pd.DataFrame) -> pd.DataFrame:
    """두 채널을 gene::ACT / gene::INH row 로 확장 후 결합"""
    union = act_df.index.union(inh_df.index, sort=False)
    act_full = act_df.reindex(union, fill_value=0.0)
    inh_full = inh_df.reindex(union, fill_value=0.0)

    act_full.index = [_tag_gene_channel(g, "ACT") for g in union]
    inh_full.index = [_tag_gene_channel(g, "INH") for g in union]
    return pd.concat([act_full, inh_full])


# ------------- per-gene scaling helpers -------------
def _fit_gene_robust(M: np.ndarray):
    med = np.nanmedian(M, axis=1, keepdims=True)
    q1 = np.nanpercentile(M, 25, axis=1, keepdims=True)
    q3 = np.nanpercentile(M, 75, axis=1, keepdims=True)
    iqr = q3 - q1
    iqr[iqr < 1e-6] = 1.0
    return med, iqr


def _transform_gene_robust(X: np.ndarray, med, iqr):
    return (X - med) / iqr


def _fit_minmax_per_gene(df_scaled: pd.DataFrame):
    row_min = np.nanmin(df_scaled.values, axis=1, keepdims=True)
    row_max = np.nanmax(df_scaled.values, axis=1, keepdims=True)
    denom = (row_max - row_min)
    denom[denom < 1e-9] = 1.0
    return row_min, denom


def _transform_minmax_per_gene(df_scaled: pd.DataFrame, row_min, denom):
    X = (df_scaled.values - row_min) / denom
    X = np.clip(X, 0, 1)
    return pd.DataFrame(X, index=df_scaled.index, columns=df_scaled.columns)


def build_scores_from_unified_ids(
    vnn_data: Dict[str, pd.DataFrame],
    id_groups: Dict[str, Iterable[str]],
    fit_pos_ids: Iterable[str] | None = None,
    fit_neg_ids: Iterable[str] | None = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Robust + min-max scaling (train-only fit) 후 gene::ACT/INH 확장
    id_groups: {"train_pos": [...], "train_neg": [...], "val_pos": [...], ...}
    """
    pos_act_pool = pd.concat(
        [vnn_data["pos_act"], vnn_data.get("val_act")], axis=1
    ).pipe(lambda df: df.loc[:, ~df.columns.duplicated()])
    pos_inh_pool = pd.concat(
        [vnn_data["pos_inh"], vnn_data.get("val_inh")], axis=1
    ).pipe(lambda df: df.loc[:, ~df.columns.duplicated()])
    neg_act_pool = vnn_data["neg_act"]
    neg_inh_pool = vnn_data["neg_inh"]

    if fit_pos_ids is None:
        fit_pos_ids = id_groups.get("train_pos", [])
    if fit_neg_ids is None:
        fit_neg_ids = id_groups.get("train_neg", [])

    def _take(df: pd.DataFrame, ids: Iterable[str]):
        cols = df.columns.intersection(ids)
        return df.loc[:, cols]

    pos_fit_act = _take(pos_act_pool, fit_pos_ids)
    pos_fit_inh = _take(pos_inh_pool, fit_pos_ids)
    neg_fit_act = _take(neg_act_pool, fit_neg_ids)
    neg_fit_inh = _take(neg_inh_pool, fit_neg_ids)

    A_med, A_iqr = _fit_gene_robust(pd.concat([pos_fit_act, neg_fit_act], axis=1).values)
    I_med, I_iqr = _fit_gene_robust(pd.concat([pos_fit_inh, neg_fit_inh], axis=1).values)

    def _scale_block(act_df, inh_df, ids):
        act_sel = _take(act_df, ids)
        inh_sel = _take(inh_df, ids)
        act_r = pd.DataFrame(
            _transform_gene_robust(act_sel.values, A_med, A_iqr),
            index=act_sel.index,
            columns=act_sel.columns,
        )
        inh_r = pd.DataFrame(
            _transform_gene_robust(inh_sel.values, I_med, I_iqr),
            index=inh_sel.index,
            columns=inh_sel.columns,
        )
        return act_r, inh_r

    tmp = {}
    for key, ids in id_groups.items():
        if "pos" in key:
            src_act, src_inh = pos_act_pool, pos_inh_pool
        elif "neg" in key:
            src_act, src_inh = neg_act_pool, neg_inh_pool
        else:
            raise ValueError(f"Unknown group key: {key}")
        tmp[key] = _scale_block(src_act, src_inh, ids)

    train_act_r = pd.concat([tmp["train_pos"][0], tmp["train_neg"][0]], axis=1)
    train_inh_r = pd.concat([tmp["train_pos"][1], tmp["train_neg"][1]], axis=1)
    A_min, A_den = _fit_minmax_per_gene(train_act_r)
    I_min, I_den = _fit_minmax_per_gene(train_inh_r)

    out = {}
    for key, (act_r, inh_r) in tmp.items():
        act_u = _transform_minmax_per_gene(act_r, A_min, A_den)
        inh_u = _transform_minmax_per_gene(inh_r, I_min, I_den)
        out[key] = expand_act_inh_channels(act_u, inh_u)

    scalers = {
        "robust_act": (A_med, A_iqr),
        "robust_inh": (I_med, I_iqr),
        "unit_act": (A_min, A_den),
        "unit_inh": (I_min, I_den),
    }
    return out, scalers


def derive_subset_and_masks(
    scores_all: pd.DataFrame,
    builder,
    channel_tags: Tuple[str, str] = GENE_CHANNEL_TAGS,
):
    """builder 그래프에 존재하는 gene 채널만 남기고 mask 생성"""
    vnn_genes = [
        g for g in builder.G.nodes if builder.G.in_degree(g) == 0 and g not in ("root", "output")
    ]
    available_genes = {name.split("::")[0] for name in scores_all.index}

    target_index = [
        f"{gene}::{tag}"
        for gene in vnn_genes
        if gene in available_genes
        for tag in channel_tags
    ]

    df_out = scores_all.reindex(target_index).fillna(0.0)
    row_has_signal = (df_out != 0).any(axis=1)

    def keep_row(tagged_gene: str) -> bool:
        base, _ = tagged_gene.split("::", 1)
        return row_has_signal[tagged_gene] and base in builder.G

    subset_genes = [idx for idx in df_out.index if keep_row(idx)]
    df_out = df_out.loc[subset_genes]

    masks, skip_nodes, alive_genes = build_vnn_masks(builder, subset_genes)
    df_pruned = df_out.loc[alive_genes]

    return df_pruned, subset_genes, alive_genes, masks, skip_nodes
