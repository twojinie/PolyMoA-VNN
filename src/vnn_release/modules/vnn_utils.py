# vnn_utils.py
import torch, networkx as nx
from typing import List, Tuple

def _canonical_gene_name(node: str, sep="::") -> str:
    return node.split(sep, 1)[0]

def build_vnn_masks(builder,
                    gene_order: List[str]
                   ) -> Tuple[List[torch.Tensor], List[str], List[str]]:
    """
    Returns
    -------
    masks        : [gene→P5, P5→P4, …, P1→output]  (6 개, 모두 연쇄 차원 일치)
    skip_nodes   : pruning 후 살아남은  P5∥P4∥P3∥P2  노드 순서
    alive_genes  : pruning 후 살아남은  gene  순서
    """
    G, root = builder.G, 'root'

    # 0) pathway 노드와 깊이 계산 (원본만: '_copy' 제외)
    is_path = lambda n: (isinstance(n, str)
                         and (n.startswith('R-HSA-') or n.startswith('HSA-'))
                         and '_copy' not in n)
    # root로부터 최단거리
    dist = nx.single_source_shortest_path_length(G, root)
    # 실제 존재하는 pathway 깊이들의 집합
    levels = sorted({d for n, d in dist.items() if is_path(n) and d >= 1})
    if not levels:
        raise RuntimeError("No pathway nodes found above root.")
    # 최심도(마지막 pathway 레벨)
    last_level = max(levels)                      # == builder.depth 가정이지만 안전하게 계산
    # 깊이별 pathway 노드 사전
    depth_nodes = {d: [] for d in range(1, last_level+1)}
    for n, d in dist.items():
        if is_path(n) and 1 <= d <= last_level:
            depth_nodes[d].append(n)


    # 1) gene → (최심도) pathway
    p_last = depth_nodes.get(last_level, [])
    if len(p_last) == 0:
        # 혹시 last_level이 비어 있으면 존재하는 가장 깊은 레벨로 대체
        fallback = max(d for d in depth_nodes if depth_nodes[d])
        p_last = depth_nodes[fallback]
        last_level = fallback

    # g2plast = torch.zeros((len(p_last), len(gene_order)))
    # g_idx = {g: i for i, g in enumerate(gene_order)} 
    # p_idx = {p: i for i, p in enumerate(p_last)}
    # for g in gene_order:
    #     if g not in G:
    #         continue
    #     # gene → pathway(자식) 에지
    #     for p in G.successors(g):
    #         if p in p_idx:
    #             g2plast[p_idx[p], g_idx[g]] = 1.

    # 변경 @@@@@@@@@@
    g2plast = torch.zeros((len(p_last), len(gene_order)))
    p_idx = {p: i for i, p in enumerate(p_last)}

    for col, token in enumerate(gene_order):
        gene = _canonical_gene_name(token)
        if gene not in G:
            continue
        for p in G.successors(gene):
            if p in p_idx:
                g2plast[p_idx[p], col] = 1.


    # gene/최심도 pathway 프루닝
    keep_col = g2plast.sum(0) != 0                 # 살아남는 gene
    g2plast = g2plast[:, keep_col]
    alive_genes = [g for g, k in zip(gene_order, keep_col) if k]

    keep_row = g2plast.sum(1) != 0                 # 살아남는 최심도 pathway
    g2plast = g2plast[keep_row, :]
    plast_alive = [p for p, k in zip(p_last, keep_row) if k]

    masks = [g2plast]
    skip_nodes = plast_alive[:]                    # skip 기준의 시작(최심도)

    # # 확인용
    # act_idx = alive_genes.index("AKT1::ACT")
    # inh_idx = alive_genes.index("AKT1::INH")
    # torch.allclose(masks[0][:, act_idx], masks[0][:, inh_idx])


    # 2) (최심도→…) 상위 레벨로 올라가며 마스크 생성
    prev_alive = plast_alive
    for lv in range(last_level-1, -1, -1):         # lv=last_level-1 ... 1, 0(output)
        src_nodes = depth_nodes[lv] if lv > 0 else ['output']  # 현재 부모 후보
        dst_nodes = prev_alive                                  # 직전 단계 자식(살아남은)

        src_i = {n: i for i, n in enumerate(src_nodes)}
        dst_i = {n: i for i, n in enumerate(dst_nodes)}
        M = torch.zeros((len(src_nodes), len(dst_nodes)))

        # child(=d) → parent(=s) 방향으로 에지 확인
        for d in dst_nodes:
            neigh = G.predecessors(d) if lv > 0 else G.successors(d)
            for s in neigh:
                if s in src_i:
                    M[src_i[s], dst_i[d]] = 1.

        # 프루닝: 자식 없는 부모/부모 없는 자식 제거
        keep_col = M.sum(0) != 0
        M = M[:, keep_col]
        dst_nodes = [n for n, k in zip(dst_nodes, keep_col) if k]

        keep_row = M.sum(1) != 0
        M = M[keep_row, :]
        src_alive = [n for n, k in zip(src_nodes, keep_row) if k]

        masks.append(M)

        # skip 대상: lv>=2 (즉, pathway 레벨 2..last_level 모두)
        if lv >= 2:
            skip_nodes.extend(src_alive)

        prev_alive = src_alive

    return masks, skip_nodes, alive_genes

def add_skip_to_all(masks):
    """
    masks: gene→P5, P5→P4, … P1→out   6 개의 list
    gene→P5 정보를 누적-전파해서 gene 이 모든 상위 레이어에도
    직접 연결되도록  mask 를 수정해 반환.
    """
    g2p = masks[0].clone()                     # gene→P5 (774 × 528 …)
    new_masks = [g2p]

    # 누적 연결 행렬 준비 (Boolean)
    reach = g2p.bool()

    for k in range(1, len(masks)):             # k = 1..5
        m = masks[k].clone().bool()            # 현재 child→parent
        # 지금까지 reach (gene→Pk) 을 parent 쪽으로 올림
        reach = m @ reach                      # Boolean matmul
        # OR 해서 gene→Pk 연결 추가
        masks[k] |= reach.float()              # in-place OR
        new_masks.append(masks[k])

    return new_masks


def build_skip_mask(skip_nodes: List[str],
                    skip_ids  : List[str]) -> torch.Tensor:
    """
    Parameters
    ----------
    skip_nodes :  List[str]
        build_vnn_masks() 가 반환한 P5∥P4∥P3∥P2 노드 순서 (len = 877)
    skip_ids   :  Iterable[str]
        알츠하이머‑연관 pathway R‑HSA‑ID 리스트

    Returns
    -------
    mask : Tensor[1, len(skip_nodes)]
        skip2out LinearMasked 에 바로 넣을 0/1 마스크
    """
    import torch
    col   = {n:i for i,n in enumerate(skip_nodes)}
    valid = set(col) #set(skip_ids) & 
    mask  = torch.zeros((1, len(skip_nodes)))
    for p in valid:
        mask[0, col[p]] = 1.
    return mask