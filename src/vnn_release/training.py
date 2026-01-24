from __future__ import annotations

import copy
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from vnn_release.modules.pnet_reactome_builder import PNetBuilder
from vnn_release.modules.vnn_framework import End2End
from vnn_release.modules.vnn_model import LinearMasked, VisibleNN
from vnn_release.modules.vnn_utils import build_vnn_masks
from .preprocess import build_scores_from_unified_ids, derive_subset_and_masks
from .data_utils import load_smiles_data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_polymoa_model(
    builder,
    subset_genes: Sequence[str],
    *,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    masks, _, alive_genes = build_vnn_masks(builder, subset_genes)
    core = VisibleNN([m.to(device) for m in masks])

    new_layers = []
    for i, blk in enumerate(core.net):
        if isinstance(blk, LinearMasked) and i < len(core.net) - 1:
            new_layers += [blk, torch.nn.LayerNorm(blk.out_features), torch.nn.GELU(), torch.nn.Dropout(0.3)]
        elif isinstance(blk, LinearMasked):
            new_layers.append(blk)
    core.net = torch.nn.Sequential(*new_layers)

    inp_dim = len(alive_genes)
    pre = torch.nn.Sequential(torch.nn.LayerNorm(inp_dim), torch.nn.Dropout(0.3))
    model = End2End(pre, core).to(device)

    def safe_init(m):
        if isinstance(m, (torch.nn.Linear, LinearMasked)):
            if hasattr(m, "weight") and m.weight is not None:
                torch.nn.init.kaiming_uniform_(m.weight, a=0.2)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    model.apply(safe_init)
    return model, alive_genes, masks


def _evaluate_logits(model, loader, device):
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pr = torch.sigmoid(model(xb.to(device))).cpu().numpy().ravel()
            preds.append(pr)
            trues.append(yb.numpy().ravel())
    y_true, y_prob = np.concatenate(trues), np.concatenate(preds)
    return y_true, y_prob


def _train_loop(model, train_loader, val_loader, device, max_epochs=500, eval_every=50, patience=5):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_state = None
    best_aupr = -float("inf")
    wait = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = torch.clamp(model(xb), -30, 30)
            loss = criterion(logits, yb)
            if hasattr(model, "skip_reg_loss"):
                loss = loss + model.skip_reg_loss()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        if ep % eval_every == 0:
            model.eval()
            y_true, y_prob = _evaluate_logits(model, val_loader, device)
            aupr = average_precision_score(y_true, y_prob)
            if aupr > best_aupr + 1e-3:
                best_aupr = aupr
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    y_true, y_prob = _evaluate_logits(model, val_loader, device)
    return {"auroc": roc_auc_score(y_true, y_prob), "aupr": average_precision_score(y_true, y_prob)}


def _to_loader(X, y, batch_size=64, shuffle=True):
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle)


# ----------------- SMILES → Morgan FP utils -----------------
def smiles_to_morgan(smiles: str, radius: int = 2, nbits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)
    arr = np.zeros((nbits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def embed_morgan(smiles_list, radius: int = 2, nbits: int = 2048):
    return np.stack([smiles_to_morgan(s, radius=radius, nbits=nbits) for s in smiles_list])


def vnn_layer_info(masks):
    info = []
    for i, m in enumerate(masks, 1):
        out_f, in_f = m.shape
        nnz = int(m.sum().item())
        info.append(
            {"layer": i, "in_dim": in_f, "out_dim": out_f, "edges": nnz, "density": nnz / m.numel()}
        )
    return info


def build_fc_dnn(layer_info, dropout=0.2):
    layers = []
    for i, d in enumerate(layer_info):
        layers.append(torch.nn.Linear(d["in_dim"], d["out_dim"], bias=True))
        if i < len(layer_info) - 1:
            layers += [torch.nn.LeakyReLU(0.1), torch.nn.Dropout(dropout)]
    return torch.nn.Sequential(*layers)


def random_mask_like(mask, keep_same_edges=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    h, w = mask.shape
    n_edges = int(mask.sum().item()) if keep_same_edges else int(mask.numel() * 0.1)
    flat_idx = torch.randperm(h * w, device=mask.device)[:n_edges]
    rand_mask = torch.zeros(h * w, device=mask.device)
    rand_mask[flat_idx] = 1.0
    return rand_mask.view(h, w)


def _prepare_fold_data(
    vnn_data: Dict[str, np.ndarray],
    fold_splits: Dict[int, Dict[str, List[str]]],
    fold_idx: int,
    *,
    builder_depth: int = 4,
    device: Optional[torch.device] = None,
    neg_pool: Optional[Sequence[str]] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids = fold_splits[fold_idx]["train"]
    val_ids = fold_splits[fold_idx]["valid"]

    available_pos = set(vnn_data["pos_act"].columns) | set(vnn_data["val_act"].columns)
    available_neg = set(vnn_data["neg_act"].columns)
    pos_train_ids = [d for d in train_ids if d in available_pos]
    pos_val_ids = [d for d in val_ids if d in available_pos]

    if neg_pool is None:
        neg_pool = list(vnn_data["neg_act"].columns)
    neg_pool = [d for d in neg_pool if d in available_neg]

    neg_train_ids = random.sample(neg_pool, min(len(pos_train_ids), len(neg_pool)))
    remaining_negs = [d for d in neg_pool if d not in neg_train_ids]
    neg_val_ids = random.sample(remaining_negs, min(len(pos_val_ids), len(remaining_negs)))

    scores, _ = build_scores_from_unified_ids(
        vnn_data,
        id_groups={
            "train_pos": pos_train_ids,
            "train_neg": neg_train_ids,
            "val_pos": pos_val_ids,
            "val_neg": neg_val_ids,
        },
    )

    pos_train = scores["train_pos"]
    neg_train = scores["train_neg"]
    pos_val = scores["val_pos"]
    neg_val = scores["val_neg"]

    scores_all = pd.concat([pos_train, neg_train], axis=1)
    train_present = (scores_all != 0).any(axis=1)
    scores_all = scores_all.loc[train_present]

    builder = PNetBuilder(depth=builder_depth, skip_ids=None)
    df_pruned, subset_genes, alive_genes, masks, _ = derive_subset_and_masks(scores_all, builder)

    pos_train = pos_train.reindex(index=alive_genes).fillna(0)
    neg_train = neg_train.reindex(index=alive_genes).fillna(0)
    pos_val = pos_val.reindex(index=alive_genes).fillna(0)
    neg_val = neg_val.reindex(index=alive_genes).fillna(0)

    X_train = np.concatenate([pos_train.T.values, neg_train.T.values])
    y_train = np.array([1] * pos_train.shape[1] + [0] * neg_train.shape[1])
    X_val = np.concatenate([pos_val.T.values, neg_val.T.values])
    y_val = np.array([1] * pos_val.shape[1] + [0] * neg_val.shape[1])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(np.log1p(X_train * 10))
    X_val = scaler.transform(np.log1p(X_val * 10))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "scaler": scaler,
        "alive_genes": alive_genes,
        "builder": builder,
        "subset_genes": subset_genes,
        "masks": masks,
        "device": device,
    }


def _prepare_ood_data(
    vnn_data: Dict[str, np.ndarray],
    ood_split: Dict[str, List[str]],
    *,
    builder_depth: int = 4,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids = ood_split["train"]
    test_ids = ood_split["test"]

    available_pos = set(vnn_data["pos_act"].columns) | set(vnn_data["val_act"].columns)
    available_neg = set(vnn_data["neg_act"].columns)

    def filter_ids(ids):
        pos = [d for d in ids if d in available_pos]
        neg = [d for d in ids if d in available_neg]
        return pos, neg

    train_pos_ids, train_neg_ids = filter_ids(train_ids)
    test_pos_ids, test_neg_ids = filter_ids(test_ids)

    scores, _ = build_scores_from_unified_ids(
        vnn_data,
        id_groups={
            "train_pos": train_pos_ids,
            "train_neg": train_neg_ids,
            "test_pos": test_pos_ids,
            "test_neg": test_neg_ids,
        },
    )

    pos_train = scores["train_pos"]
    neg_train = scores["train_neg"]
    pos_test = scores["test_pos"]
    neg_test = scores["test_neg"]

    scores_all = pd.concat([pos_train, neg_train], axis=1)
    train_present = (scores_all != 0).any(axis=1)
    scores_all = scores_all.loc[train_present]

    builder = PNetBuilder(depth=builder_depth, skip_ids=None)
    df_pruned, subset_genes, alive_genes, masks, _ = derive_subset_and_masks(scores_all, builder)

    pos_train = pos_train.reindex(index=alive_genes).fillna(0)
    neg_train = neg_train.reindex(index=alive_genes).fillna(0)
    pos_test = pos_test.reindex(index=alive_genes).fillna(0)
    neg_test = neg_test.reindex(index=alive_genes).fillna(0)

    X_train = np.concatenate([pos_train.T.values, neg_train.T.values])
    y_train = np.array([1] * pos_train.shape[1] + [0] * neg_train.shape[1])
    X_test = np.concatenate([pos_test.T.values, neg_test.T.values])
    y_test = np.array([1] * pos_test.shape[1] + [0] * neg_test.shape[1])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(np.log1p(X_train * 10))
    X_test = scaler.transform(np.log1p(X_test * 10))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "alive_genes": alive_genes,
        "builder": builder,
        "subset_genes": subset_genes,
        "masks": masks,
        "device": device,
    }


def _prepare_smiles_fold_data(
    pos_smiles_df: pd.DataFrame,
    neg_smiles_df: pd.DataFrame,
    fold_splits: Dict[int, Dict[str, List[str]]],
    fold_idx: int,
):
    train_ids = fold_splits[fold_idx]["train"]
    val_ids = fold_splits[fold_idx]["valid"]

    def _take(df, ids):
        return df[df["DrugID"].isin(ids)].copy()

    train_df = pd.concat([_take(pos_smiles_df, train_ids), _take(neg_smiles_df, train_ids)])
    val_df = pd.concat([_take(pos_smiles_df, val_ids), _take(neg_smiles_df, val_ids)])

    X_train = embed_morgan(train_df["SMILES"])
    y_train = train_df["label"].to_numpy()
    X_val = embed_morgan(val_df["SMILES"])
    y_val = val_df["label"].to_numpy()
    return {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}


def _prepare_smiles_ood_data(
    pos_smiles_df: pd.DataFrame,
    neg_smiles_df: pd.DataFrame,
    ood_split: Dict[str, List[str]],
):
    def _take(df, ids):
        return df[df["DrugID"].isin(ids)].copy()

    train_df = pd.concat([_take(pos_smiles_df, ood_split["train"]), _take(neg_smiles_df, ood_split["train"])])
    test_df = pd.concat([_take(pos_smiles_df, ood_split["test"]), _take(neg_smiles_df, ood_split["test"])])

    X_train = embed_morgan(train_df["SMILES"])
    y_train = train_df["label"].to_numpy()
    X_test = embed_morgan(test_df["SMILES"])
    y_test = test_df["label"].to_numpy()
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}


def run_polymoa_fold(vnn_data, fold_splits, fold_idx, *, builder_depth=4, device=None):
    data = _prepare_fold_data(vnn_data, fold_splits, fold_idx, builder_depth=builder_depth, device=device)
    train_loader = _to_loader(data["X_train"], data["y_train"], batch_size=64, shuffle=True)
    val_loader = _to_loader(data["X_val"], data["y_val"], batch_size=64, shuffle=False)

    model, alive_genes, masks = build_polymoa_model(data["builder"], data["subset_genes"], device=data["device"])
    return _train_loop(model, train_loader, val_loader, data["device"])


def run_polymoa_ood(vnn_data, ood_split, *, builder_depth=4, device=None):
    data = _prepare_ood_data(vnn_data, ood_split, builder_depth=builder_depth, device=device)
    train_loader = _to_loader(data["X_train"], data["y_train"], batch_size=64, shuffle=True)
    test_loader = _to_loader(data["X_test"], data["y_test"], batch_size=64, shuffle=False)

    model, _, _ = build_polymoa_model(data["builder"], data["subset_genes"], device=data["device"])
    return _train_loop(model, train_loader, test_loader, data["device"])


def run_fc_dnn_fold(vnn_data, fold_splits, fold_idx, *, builder_depth=4, device=None):
    data = _prepare_fold_data(vnn_data, fold_splits, fold_idx, builder_depth=builder_depth, device=device)
    layer_info = vnn_layer_info(data["masks"])
    core = build_fc_dnn(layer_info, dropout=0.2).to(data["device"])

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    class FlatHead(torch.nn.Module):
        def __init__(self, core_net):
            super().__init__()
            self.core = core_net

        def forward(self, x):
            return self.core(x).squeeze(-1)

    model = End2End(Identity(), FlatHead(core)).to(data["device"])
    train_loader = _to_loader(data["X_train"], data["y_train"], batch_size=64, shuffle=True)
    val_loader = _to_loader(data["X_val"], data["y_val"], batch_size=64, shuffle=False)
    metrics = _train_loop(model, train_loader, val_loader, data["device"])
    return metrics


def run_fc_dnn_ood(vnn_data, ood_split, *, builder_depth=4, device=None):
    data = _prepare_ood_data(vnn_data, ood_split, builder_depth=builder_depth, device=device)
    layer_info = vnn_layer_info(data["masks"])
    core = build_fc_dnn(layer_info, dropout=0.2).to(data["device"])

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    class FlatHead(torch.nn.Module):
        def __init__(self, core_net):
            super().__init__()
            self.core = core_net

        def forward(self, x):
            return self.core(x).squeeze(-1)

    model = End2End(Identity(), FlatHead(core)).to(data["device"])
    train_loader = _to_loader(data["X_train"], data["y_train"], batch_size=64, shuffle=True)
    test_loader = _to_loader(data["X_test"], data["y_test"], batch_size=64, shuffle=False)
    return _train_loop(model, train_loader, test_loader, data["device"])


def run_rand_vnn_fold(vnn_data, fold_splits, fold_idx, *, builder_depth=4, device=None):
    data = _prepare_fold_data(vnn_data, fold_splits, fold_idx, builder_depth=builder_depth, device=device)
    rand_masks = [random_mask_like(m, seed=42 + i) for i, m in enumerate(data["masks"])]
    rand_vnn = VisibleNN([m.to(data["device"]) for m in rand_masks]).to(data["device"])

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    class FlatHead(torch.nn.Module):
        def __init__(self, core_net):
            super().__init__()
            self.core = core_net

        def forward(self, x):
            return self.core(x).squeeze(-1)

    model = End2End(Identity(), FlatHead(rand_vnn)).to(data["device"])
    train_loader = _to_loader(data["X_train"], data["y_train"], batch_size=64, shuffle=True)
    val_loader = _to_loader(data["X_val"], data["y_val"], batch_size=64, shuffle=False)
    metrics = _train_loop(model, train_loader, val_loader, data["device"])
    return metrics


def run_rand_vnn_ood(vnn_data, ood_split, *, builder_depth=4, device=None):
    data = _prepare_ood_data(vnn_data, ood_split, builder_depth=builder_depth, device=device)
    rand_masks = [random_mask_like(m, seed=42 + i) for i, m in enumerate(data["masks"])]
    rand_vnn = VisibleNN([m.to(data["device"]) for m in rand_masks]).to(data["device"])

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    class FlatHead(torch.nn.Module):
        def __init__(self, core_net):
            super().__init__()
            self.core = core_net

        def forward(self, x):
            return self.core(x).squeeze(-1)

    model = End2End(Identity(), FlatHead(rand_vnn)).to(data["device"])
    train_loader = _to_loader(data["X_train"], data["y_train"], batch_size=64, shuffle=True)
    test_loader = _to_loader(data["X_test"], data["y_test"], batch_size=64, shuffle=False)
    return _train_loop(model, train_loader, test_loader, data["device"])


def run_xgb_fold(pos_smiles_df, neg_smiles_df, fold_splits, fold_idx, *, seed: int = 42):
    data = _prepare_smiles_fold_data(pos_smiles_df, neg_smiles_df, fold_splits, fold_idx)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=8,
    )
    model.fit(data["X_train"], data["y_train"])
    y_prob = model.predict_proba(data["X_val"])[:, 1]
    return {"auroc": roc_auc_score(data["y_val"], y_prob), "aupr": average_precision_score(data["y_val"], y_prob)}


def run_xgb_ood(pos_smiles_df, neg_smiles_df, ood_split, *, seed: int = 42):
    data = _prepare_smiles_ood_data(pos_smiles_df, neg_smiles_df, ood_split)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=8,
    )
    model.fit(data["X_train"], data["y_train"])
    y_prob = model.predict_proba(data["X_test"])[:, 1]
    return {"auroc": roc_auc_score(data["y_test"], y_prob), "aupr": average_precision_score(data["y_test"], y_prob)}


def run_lr_fold(pos_smiles_df, neg_smiles_df, fold_splits, fold_idx, *, seed: int = 42):
    data = _prepare_smiles_fold_data(pos_smiles_df, neg_smiles_df, fold_splits, fold_idx)
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="saga",
        max_iter=2000,
        random_state=seed,
        n_jobs=8,
        verbose=0,
    )
    model.fit(data["X_train"], data["y_train"])
    y_prob = model.predict_proba(data["X_val"])[:, 1]
    return {"auroc": roc_auc_score(data["y_val"], y_prob), "aupr": average_precision_score(data["y_val"], y_prob)}


def run_lr_ood(pos_smiles_df, neg_smiles_df, ood_split, *, seed: int = 42):
    data = _prepare_smiles_ood_data(pos_smiles_df, neg_smiles_df, ood_split)
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="saga",
        max_iter=2000,
        random_state=seed,
        n_jobs=8,
        verbose=0,
    )
    model.fit(data["X_train"], data["y_train"])
    y_prob = model.predict_proba(data["X_test"])[:, 1]
    return {"auroc": roc_auc_score(data["y_test"], y_prob), "aupr": average_precision_score(data["y_test"], y_prob)}


def run_all_id_models(
    vnn_data: Dict[str, np.ndarray],
    pos_smiles_df: pd.DataFrame,
    neg_smiles_df: pd.DataFrame,
    fold_splits: Dict[int, Dict[str, List[str]]],
    seeds: Sequence[int],
    *,
    builder_depth: int = 4,
    device: Optional[torch.device] = None,
):
    records = []
    for seed in seeds:
        set_seed(seed)
        for fold_idx in sorted(fold_splits.keys()):
            metrics = run_polymoa_fold(vnn_data, fold_splits, fold_idx, builder_depth=builder_depth, device=device)
            record = {"seed": seed, "fold": fold_idx, "model": "Polymoa-VNN", **metrics}
            records.append(record)
            print(f"[ID] seed={seed} fold={fold_idx} model=Polymoa-VNN AUROC={metrics['auroc']:.3f} AUPR={metrics['aupr']:.3f}")

            metrics = run_fc_dnn_fold(vnn_data, fold_splits, fold_idx, builder_depth=builder_depth, device=device)
            record = {"seed": seed, "fold": fold_idx, "model": "FC-DNN", **metrics}
            records.append(record)
            print(f"[ID] seed={seed} fold={fold_idx} model=FC-DNN AUROC={metrics['auroc']:.3f} AUPR={metrics['aupr']:.3f}")

            metrics = run_rand_vnn_fold(vnn_data, fold_splits, fold_idx, builder_depth=builder_depth, device=device)
            record = {"seed": seed, "fold": fold_idx, "model": "RandMasked-VNN", **metrics}
            records.append(record)
            print(f"[ID] seed={seed} fold={fold_idx} model=RandMasked-VNN AUROC={metrics['auroc']:.3f} AUPR={metrics['aupr']:.3f}")

            metrics = run_xgb_fold(
                pos_smiles_df, neg_smiles_df, fold_splits, fold_idx, seed=seed
            )
            record = {"seed": seed, "fold": fold_idx, "model": "XGB", **metrics}
            records.append(record)
            print(f"[ID] seed={seed} fold={fold_idx} model=XGB AUROC={metrics['auroc']:.3f} AUPR={metrics['aupr']:.3f}")

            metrics = run_lr_fold(
                pos_smiles_df, neg_smiles_df, fold_splits, fold_idx, seed=seed
            )
            record = {"seed": seed, "fold": fold_idx, "model": "LR", **metrics}
            records.append(record)
            print(f"[ID] seed={seed} fold={fold_idx} model=LR AUROC={metrics['auroc']:.3f} AUPR={metrics['aupr']:.3f}")
    return records


def run_all_ood_models(
    vnn_data: Dict[str, np.ndarray],
    pos_smiles_df: pd.DataFrame,
    neg_smiles_df: pd.DataFrame,
    ood_split: Dict[str, List[str]],
    seeds: Sequence[int],
    *,
    builder_depth: int = 4,
    device: Optional[torch.device] = None,
):
    records = []
    for seed in seeds:
        set_seed(seed)

        metrics = run_polymoa_ood(vnn_data, ood_split, builder_depth=builder_depth, device=device)
        record = {"seed": seed, "model": "Polymoa-VNN", **metrics}
        records.append(record)
        print(f"[OOD] seed={seed} model=Polymoa-VNN AUROC={metrics['auroc']:.3f} AUPR={metrics['aupr']:.3f}")

        metrics = run_fc_dnn_ood(vnn_data, ood_split, builder_depth=builder_depth, device=device)
        record = {"seed": seed, "model": "FC-DNN", **metrics}
        records.append(record)
        print(f"[OOD] seed={seed} model=FC-DNN AUROC={metrics['auroc']:.3f} AUPR={metrics['aupr']:.3f}")

        metrics = run_rand_vnn_ood(vnn_data, ood_split, builder_depth=builder_depth, device=device)
        record = {"seed": seed, "model": "RandMasked-VNN", **metrics}
        records.append(record)
        print(f"[OOD] seed={seed} model=RandMasked-VNN AUROC={metrics['auroc']:.3f} AUPR={metrics['aupr']:.3f}")

        metrics = run_xgb_ood(pos_smiles_df, neg_smiles_df, ood_split, seed=seed)
        record = {"seed": seed, "model": "XGB", **metrics}
        records.append(record)
        print(f"[OOD] seed={seed} model=XGB AUROC={metrics['auroc']:.3f} AUPR={metrics['aupr']:.3f}")

        metrics = run_lr_ood(pos_smiles_df, neg_smiles_df, ood_split, seed=seed)
        record = {"seed": seed, "model": "LR", **metrics}
        records.append(record)
        print(f"[OOD] seed={seed} model=LR AUROC={metrics['auroc']:.3f} AUPR={metrics['aupr']:.3f}")
    return records
