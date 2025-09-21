# src/network_analysis/deal_network_centrality.py
import os
import warnings
from tqdm import tqdm
from typing import Dict, List, Optional

import pandas as pd
import networkx as nx

# 분석할 가중치 컬럼
WEIGHT_COLS: List[str] = ["거래횟수", "거래관계", "거래액"]


def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _aggregate_edges(df: pd.DataFrame, weight_col: str, by_year: bool = True) -> pd.DataFrame:
    """(seller, buyer, [year])로 합산. self-loop 제거."""
    must = ["시군구코드_seller", "시군구코드_buyer", weight_col]
    for c in must:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    use_cols = must + (["기준연도"] if "기준연도" in df.columns and by_year else [])
    work = df[use_cols].copy()
    work["시군구코드_seller"] = work["시군구코드_seller"].astype(str).str.zfill(5)
    work["시군구코드_buyer"]  = work["시군구코드_buyer"].astype(str).str.zfill(5)

    group_cols = ["시군구코드_seller", "시군구코드_buyer"]
    if "기준연도" in work.columns and by_year:
        group_cols.append("기준연도")

    agg = (
        work.dropna(subset=["시군구코드_seller", "시군구코드_buyer"])
            .groupby(group_cols, as_index=False)[weight_col]
            .sum()
    )
    agg = agg[agg["시군구코드_seller"] != agg["시군구코드_buyer"]].reset_index(drop=True)
    return agg


def _build_digraph(edges: pd.DataFrame, weight_col: str) -> nx.DiGraph:
    """가중치 weight, 최단거리용 length=1/(weight+eps)."""
    G = nx.DiGraph()
    eps = 1e-9
    for _, r in edges.iterrows():
        u = r["시군구코드_seller"]
        v = r["시군구코드_buyer"]
        w = float(r[weight_col]) if pd.notna(r[weight_col]) else 0.0
        w = max(0.0, w)
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 0.0)
        data["length"] = 1.0 / (w + eps) if w > 0 else 1.0 / eps
    return G


def _safe_eigenvector_centrality(G: nx.DiGraph):
    try:
        return nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        return nx.eigenvector_centrality(G, weight="weight", max_iter=5000, tol=1e-6)


def _compute_centralities(G: nx.DiGraph) -> pd.DataFrame:
    nodes = list(G.nodes())
    if not nodes:
        return pd.DataFrame(columns=["node"])

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    in_strength = dict(G.in_degree(weight="weight"))
    out_strength = dict(G.out_degree(weight="weight"))
    in_deg_c = nx.in_degree_centrality(G)
    out_deg_c = nx.out_degree_centrality(G)
    
    UG = G.to_undirected()
    closeness = nx.closeness_centrality(UG, distance="length") if len(nodes) > 1 else {u: 0.0 for u in nodes}
    betweenness = nx.betweenness_centrality(UG, weight="length", normalized=True) if len(nodes) > 2 else {u: 0.0 for u in nodes}
    eigen = _safe_eigenvector_centrality(UG) if len(nodes) > 1 else {u: 0.0 for u in nodes}
    pagerank = nx.pagerank(UG, alpha=0.85, weight="weight") if len(nodes) > 1 else {u: 1.0/len(nodes) for u in nodes}

    df = pd.DataFrame({
        "node": nodes,
        "in_degree": [in_deg.get(u, 0) for u in nodes],
        "out_degree": [out_deg.get(u, 0) for u in nodes],
        "in_strength": [float(in_strength.get(u, 0.0)) for u in nodes],
        "out_strength": [float(out_strength.get(u, 0.0)) for u in nodes],
        "in_degree_c": [float(in_deg_c.get(u, 0.0)) for u in nodes],
        "out_degree_c": [float(out_deg_c.get(u, 0.0)) for u in nodes],
        "closeness": [float(closeness.get(u, 0.0)) for u in nodes],
        "betweenness": [float(betweenness.get(u, 0.0)) for u in nodes],
        "eigenvector": [float(eigen.get(u, 0.0)) for u in nodes],
        "pagerank": [float(pagerank.get(u, 0.0)) for u in nodes],
    })
    return df


def analyze_single_year(df: pd.DataFrame, weight: str, year: Optional[int]) -> pd.DataFrame:
    """특정 연도/가중치의 중심성 테이블 생성."""
    if year is not None:
        sub = df[df["기준연도"] == year].copy()
    else:
        sub = df.copy()
    edges = _aggregate_edges(sub, weight_col=weight, by_year=False)
    G = _build_digraph(edges, weight_col=weight)
    c = _compute_centralities(G)
    c = c.rename(columns={
        "in_strength": f"in_strength_{weight}",
        "out_strength": f"out_strength_{weight}",
    })
    return c


def run_all_by_year(
    networks: Dict[str, pd.DataFrame],
    out_tbl_dir: str = "data/deal_network",
    weights: Optional[List[str]] = None,
    overall: bool = False,
    drop_nodes: Optional[List[str]] = None,
) -> None:
    """
    네트워크×가중치×연도별 중심성 CSV 저장.
    - 저장 위치: {out_tbl_dir}/{net}/{weight}/centrality_{year}.csv
    - overall=True이면 centrality_overall.csv도 추가 저장
    - drop_nodes: e.g., ["9999"] (마스킹 코드 제거)
    """
    weights = weights or WEIGHT_COLS
    drop_nodes = set((drop_nodes or ["9999"]))
    _ensure_dirs(out_tbl_dir)

    for net_name, df in tqdm(networks.items()):
        if "기준연도" not in df.columns:
            raise ValueError(f"[{net_name}] missing '기준연도' column.")

        # 코드 5자리 zero-fill
        for col in ["시군구코드_seller", "시군구코드_buyer"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.zfill(5)

        years = sorted(pd.unique(df["기준연도"].dropna()))
        for w in weights:
            subdir = os.path.join(out_tbl_dir, net_name, w)
            _ensure_dirs(subdir)

            # per-year 저장
            for y in years:
                central = analyze_single_year(df, weight=w, year=y)
                if not central.empty:
                    # 마스킹 코드 제거 (9999 또는 09999 등)
                    drop5 = {s.zfill(5) for s in drop_nodes}
                    central = central[~central["node"].isin(drop_nodes) & ~central["node"].isin(drop5)]
                out_csv = os.path.join(subdir, f"centrality_{y}.csv")
                central.to_csv(out_csv, index=False, encoding="utf-8-sig")

            # overall (옵션)
            if overall:
                central_all = analyze_single_year(df, weight=w, year=None)
                if not central_all.empty:
                    drop5 = {s.zfill(5) for s in drop_nodes}
                    central_all = central_all[~central_all["node"].isin(drop_nodes) & ~central_all["node"].isin(drop5)]
                out_csv = os.path.join(subdir, f"centrality_overall.csv")
                central_all.to_csv(out_csv, index=False, encoding="utf-8-sig")
