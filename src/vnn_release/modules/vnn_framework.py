# vnn_framework.py
"""
End-to-end pipeline:
    (1) load df_out            – protein scores per compound
    (2) propagate on PPI       – NP or GCN
    (3) Visible NN             – gene→pathway(5)→output
    (4) train / infer
"""
# ───────────────────────── imports
import os, json, math, torch, torch.nn as nn
import pandas as pd, numpy as np, networkx as nx
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional
from .pnet_reactome_builder import PNetBuilder
from .vnn_utils                import build_vnn_masks, build_skip_mask
from .vnn_model                import VisibleNN
from .vnn_model                import VisibleNNWithSkip
from .vnn_model                import VisibleNNWithSkipReg


# ❗ GCN 모드를 쓰려면 PyTorch Geometric 설치
try:
    from torch_geometric.data import Data as PyGData
    from torch_geometric.nn import GCNConv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

# ───────────────────────── 0. 환경
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = Path(__file__).resolve().parents[3]
PPI_CSV   = os.environ.get("PPI_CSV", str(DATA_ROOT / "data" / "ppi_edges.tsv"))  # optional
DF_OUT_CSV = os.environ.get("DF_OUT_CSV")  # optional: protein score matrix
LABEL_CSV  = os.environ.get("LABEL_CSV")   # optional: DrugID + label table
# skip_ids = pd.read_csv('output_related_pathways.csv')['PathwayID'].tolist()

# ───────────────────────── 1. 데이터셋
class CompoundDataset(Dataset):
    """df_out(row=protein, col=compound) + labels(csv) → (input_vec, y)

    Parameters
    ----------
    df_out:
        DataFrame whose columns are compound identifiers.
    label_df:
        Label table containing a ``DrugID`` column.
    vnn_genes:
        Ordering of genes expected by the VNN.
    drop_ids:
        Iterable of compound identifiers to exclude (helps remove validation
        leaks). Identifiers are compared as strings.
    """

    def __init__(
        self,
        df_out: pd.DataFrame,
        label_df: pd.DataFrame,
        vnn_genes,
        *,
        drop_ids: Optional[Iterable[str]] = None,
    ):
        if drop_ids:
            drop_ids = {str(i) for i in drop_ids}
            keep_cols = [c for c in df_out.columns if str(c) not in drop_ids]
            df_out = df_out.loc[:, keep_cols]
            keep_mask = ~label_df['DrugID'].astype(str).isin(drop_ids)
            label_df = label_df.loc[keep_mask].copy()

        self.df = df_out.reindex(vnn_genes).fillna(0)
        self.comp = self.df.columns.tolist()
        self.y = (
            label_df
            .set_index('DrugID')
            .loc[self.comp]['label']
            .values
            .astype(np.float32)
        )

    def __len__(self): return len(self.comp)

    def __getitem__(self, idx):
        cid  = self.comp[idx]
        x    = self.df[cid].values.astype(np.float32)  # shape [N_prot]
        x = np.log1p(x)  # 스케일 업
        y    = self.y[idx]
        return torch.tensor(x), torch.tensor(y) # cid 추가

# ───────────────────────── 2-A. Network-Propagation 레이어
class NetProp(nn.Module):
    """Random Walk with Restart:  x' = (1-α)(I-αW)^-1 x  ≈ power-iteration"""
    def __init__(self, W: torch.Tensor, alpha: float = .7, iters: int = 10):
        super().__init__()
        self.register_buffer('W', W)   # [N,N], row-stochastic
        self.alpha = alpha
        self.iters = iters

    def forward(self, x):  # x: [B,N]
        h = x
        for _ in range(self.iters):
            h = self.alpha * torch.matmul(h, self.W) + (1 - self.alpha) * x
        return h

# ───────────────────────── 2-B. GCN 레이어 (옵션)
class GCNSimple(nn.Module):
    def __init__(self, edge_index, hidden_dim=16):
        super().__init__()
        if not _HAS_PYG: raise ImportError("Install torch-geometric for GCN mode.")
        self.register_buffer('edge_index', edge_index)   # [2,E]
        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, x):          # x: [B,N]
        x = x.unsqueeze(-1)        # [B,N,1]
        # PyG 는 batching 쉬우려면 loop
        outs = []
        for sample in x:
            h = torch.relu(self.conv1(sample, self.edge_index))
            outs.append(self.conv2(h, self.edge_index).squeeze(-1))
        return torch.stack(outs)   # [B,N]

# ───────────────────────── 3. Visible NN


# ───────────────────────── 4. 그래프/마스크 준비
def load_ppi(edge_file: str, weighted=False):
    """
    edge_file : 'src<TAB>dst'  (또는  'src<tab>dst<tab>weight')
    반환
      W         : [N, N] row-stochastic adjacency (tensor, float32)
      prot_list : protein ID 순서 (df_out.index 와 맞춰야 함)
    """
    df = pd.read_csv(edge_file, sep='\t', header=None)
    if weighted and df.shape[1] >= 3:
        df.columns = ['src','dst','w']
    else:
        df['w'] = 1.0
        df.columns = ['src','dst','w']

    prots = sorted(set(df.src) | set(df.dst))
    idx   = {p:i for i,p in enumerate(prots)}

    W = np.zeros((len(prots), len(prots)), dtype=np.float32)
    for s,d,w in df.itertuples(index=False):
        i, j = idx[s], idx[d]
        W[i,j] = W[j,i] = w                         # 무방향 가정
    W /= (W.sum(1, keepdims=True) + 1e-12)         # 행 정규화
    return torch.tensor(W), prots


# ───────────────────────── 5. 모델 통합
class End2End(nn.Module):
    def __init__(self, prop_layer, vnn):
        super().__init__()
        self.prop = prop_layer
        self.vnn  = vnn
    def forward(self, x):                # x: [B, N_gene]
        x = self.prop(x)
        return self.vnn(x)               # [B]

    # ────────────── NEW ──────────────
    @torch.no_grad()
    def propagate_only(self, x):
        """x:[B,N]  →  x*:[B,N]   (VNN 이전 protein score)"""
        return self.prop(x)

    def predict_with_scores(self, x, return_scores=False):
        """
        x:[B,N]  → (logit, x*)   : training 중 모니터링용
        """
        x_star = self.prop(x)
        logit  = self.vnn(x_star)
        if return_scores:
            return logit, x_star
        return logit

# ───────────────────────── 6. 학습 스크립트
def train_loop(model, loader, epochs=10, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(1,epochs+1):
        model.train(); tot=0; correct=0
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            logit = model(x)
            loss  = bce(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += len(y)
            correct += ((logit.sigmoid()>.5)==y.bool()).sum().item()
            breakpoint()
        acc = correct/tot
        print(f"Epoch {ep:2d}  loss {loss.item():.4f}  acc {acc:.3f}")

# ───────────────────────── main
if __name__ == "__main__":
    # 1) load df_out (protein×compound)  +  labels
    builder = PNetBuilder(depth=5, skip_ids=skip_ids)
    vnn_genes = [g for g in builder.G.nodes
                if builder.G.in_degree(g)==0 and g not in ('root','output')]  # 10 k

    # df_out 을 VNN-gene 순서로 재배열 + 0-패딩
    df_out = pd.read_csv(DF_OUT_CSV, index_col=0)\
            .reindex(vnn_genes).fillna(0)
    #df_out   = pd.read_csv(DF_OUT_CSV, index_col=0)
    label_df = pd.read_csv(LABEL_CSV)
    ds = CompoundDataset(df_out, label_df, vnn_genes)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    # 2) PPI layer
    W, prot_order = load_ppi(PPI_CSV)
    print("# proteins in PPI :", len(prot_order))
    print("# overlap with df_out :", (df_out.index.isin(prot_order)).sum())

    prop_mode = 'np'         # 'np' or 'gcn'
    if prop_mode=='np':
        prop_layer = NetProp(W.to(DEVICE))
    elif prop_mode == 'gcn':
        if not _HAS_PYG: raise RuntimeError("Install torch-geometric for GCN.")
        edge_idx = torch.nonzero(W>0, as_tuple=False).t().contiguous()
        prop_layer = GCNSimple(edge_idx.to(DEVICE))

    # 3) VNN masks (import 해옴)
    #builder = PNetBuilder(depth=5)  
       
    masks   = build_vnn_masks(builder, gene_order=prot_order) # prot_order = df_out.index
    skip_m = build_skip_mask(builder, skip_ids).to(DEVICE)
    masks_cuda  = [m.to(DEVICE) for m in masks]
    # vnn     = VisibleNN(masks_cuda)
    vnn = VisibleNNWithSkip([m.to(DEVICE) for m in masks], skip_m)

    # 3) PPI propagation (NetProp 또는 GCNSimple) → VNN 연결
    model = End2End(prop_layer, vnn).to(DEVICE) 
    print("Total trainable params:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ---------------- propagation 결과만 보고 싶을 때 ----------------
    TEST_DRUG = df_out.columns[0]
    x_raw = torch.tensor(
        df_out[TEST_DRUG].reindex(prot_order).fillna(0).values,
        device=DEVICE
    ).unsqueeze(0)                                 # [1, N_prot]

    model.eval()                                   # 드롭아웃 등 끔
    x_star = model.propagate_only(x_raw)           # [1, N_prot]
    print("\n★ PPI propagation output (top-10 proteins)")
    top10 = torch.topk(x_star.squeeze(0), 10).indices.cpu()
    for rk,i in enumerate(top10, 1):
        print(f"{rk:2d}. {prot_order[i]:<12}  {x_star[0,i]:.4f}")

    # 4) train
    #train_loop(model, loader, epochs=10, lr=1e-3)

    # 5) 저장
    #print("✅ model saved.")
