# pnet_reactome_builder.py
import networkx as nx, pandas as pd, re
from collections import defaultdict
from os.path import join
from pathlib import Path
from .gmt_reader import GMT

# Reactome assets are bundled under ../../../../data/reactome relative to this file
BASE = Path(__file__).resolve().parents[3] / "data" / "reactome"
REL  = 'ReactomePathwaysRelation.txt'
GMTF = 'ReactomePathways.gmt'

class PNetBuilder:
    # Avoid spamming depth stats when builder is instantiated many times
    _stats_logged = False
    def __init__(self, depth:int=5, skip_ids=None):
        self.depth = depth
        self.skip_ids = set(skip_ids or [])
        self._load_data()
        self._build_graph()

    # ────────────────────
    def _load_data(self):
        reactome_dir = Path(BASE)
        rel_path = reactome_dir / REL
        gmt_path = reactome_dir / GMTF

        rel = pd.read_csv(rel_path, sep='\t', names=['parent','child'])
        self.hierarchy = rel[rel['parent'].str.contains('HSA') & rel['child'].str.contains('HSA')]

        # ID ↔ name 매핑 구축
        name_map = {}
        with open(gmt_path) as f:
            for line in f:
                cols = line.strip().split('\t')
                if len(cols) < 2:
                    continue
                name, reactome_id = cols[0], cols[1]
                name_map[reactome_id] = name
        self.pathway_name_map = name_map

        # gene-to-pathway 테이블 (ID 기준)
        self.gene_map  = GMT().load_data(gmt_path, pathway_col=1, genes_col=2) # 3->2

    # ──────────────────── core
    def _build_graph(self):
        G = nx.DiGraph()

        # 1) pathway hierarchy
        G.add_edges_from(self.hierarchy.itertuples(index=False, name=None))

        # 2) add root
        roots = [n for n,d in G.in_degree() if d==0]
        G.add_node('root')
        G.add_edges_from([('root', r) for r in roots])

        # 3) **depth cut 먼저** (pathway만 남김)
        G = nx.ego_graph(G, 'root', radius=self.depth, center=True).copy()

        # 4) 현 시점 leaf(terminal) 집합
        leaf_set = {n for n in G.nodes if G.out_degree(n)==0 and n!='root'}

        dist = nx.single_source_shortest_path_length(G, 'root')
        real_leafs = {
            n for n, d in dist.items()
            if d == self.depth and isinstance(n, str)
            and (n.startswith('R-HSA-') or n.startswith('HSA-'))
        }

        first_log = not PNetBuilder._stats_logged
        if first_log:
            print(f"[DEBUG] leaf pathways after cut: {len(leaf_set):,}")
            print(f"[INFO] True leaf pathways at depth={self.depth}: {len(real_leafs):,}")


        # 5) gene → leaf 엣지 (skip-off)
        add_cnt = 0
        real_edge_cnt = 0
        for _, row in self.gene_map.iterrows():
            gene = row['gene']
            path = row['group']          # pathway ID
            #print(gene, path)
            if path in leaf_set:
                G.add_edge(gene, path)
                add_cnt += 1
                # ✅ 실제 leaf(pathway)만 카운트
                if path in real_leafs:
                    real_edge_cnt += 1
        if first_log:
            print(f"[DEBUG] gene→pathway edges added: {add_cnt:,}")
            print(f"[INFO] gene→REAL leaf pathway edges (true biological links): {real_edge_cnt:,}")
            PNetBuilder._stats_logged = True


        # 6) 패딩(copy)로 모든 branch 깊이=depth
        copy_idx = 0
        for leaf in list(leaf_set):              # 원래 leaf 기준
            d = nx.shortest_path_length(G, 'root', leaf)
            cur = leaf
            while d < self.depth:
                copy_idx += 1
                cp = f"{leaf}_copy{copy_idx}"
                G.add_edge(cur, cp)
                cur, d = cp, d+1

        # 7) output
        G.add_node('output')
        for n in G.successors('root'):
            G.add_edge(n, 'output') # 방향:  level-1  →  output
        
        # NEW: 말단 ~ 2차 pathway → output 직접 연결
        valid_skip = set()
        if self.skip_ids:
            # NEW: skip‑ids 중 그래프에 존재하는 것만 연결
            valid_skip = self.skip_ids & set(G.nodes)
            if missing := self.skip_ids - valid_skip:        # 빠진 ID 로그
                print(f"[WARN] {len(missing)} skip_ids not in graph depth≤{self.depth}")
            for n in valid_skip:
                G.add_edge(n, 'output')

        # 무결성 체크 (선택)
        assert valid_skip <= set(G.nodes), "Some skip_ids not in graph"

        self.G = G

    # ──────────────────── stats
    def show_stats(self, sample=10):
        G,r,o = self.G,'root','output'
        genes = [n for n in G.nodes if G.in_degree(n)==0 and n not in (r,o)]
        print(f"\n# Gene nodes: {len(genes):,}")

        # layer summary
        lvl = defaultdict(lambda: {'orig':0,'pad':0})
        for n,d in nx.single_source_shortest_path_length(G,r).items():
            if n in (r,o) or n in genes: continue
            key='pad' if re.search(r'_copy\d+$',n) else 'orig'
            lvl[d][key]+=1
        print("── Pathway layer summary")
        for k in sorted(lvl):
            print(f" level {k}: original {lvl[k]['orig']:>5} | padding {lvl[k]['pad']:>5}")

        # sample edges
        rows=[(g,p,nx.shortest_path_length(G,r,p)) for g in genes for p in G.successors(g)]
        df=pd.DataFrame(rows, columns=['gene','pathway','layer'])
        print("\nSample gene→pathway mappings")
        print(df.head(sample).to_string(index=False))
        df.to_csv("gene2terpath.csv", index = False)

# # ──────────────────── run
# if __name__ == "__main__":
#     builder = PNetBuilder(depth=5)
#     builder.show_stats()
