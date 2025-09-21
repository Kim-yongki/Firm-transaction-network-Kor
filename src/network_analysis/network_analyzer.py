"""
네트워크 분석 모듈

NetworkX 기반의 그래프 생성, 커뮤니티 탐지, 중심성 분석, 노드/엣지 데이터 생성 기능을 제공합니다.
"""

import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg
from infomap import Infomap
import geopandas as gpd
from shapely.geometry import Point
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class NetworkAnalyzer:
    """네트워크 분석 통합 클래스"""
    
    def __init__(self, polygon_files: Optional[Dict[str, str]] = None):
        """
        네트워크 분석기 초기화
        
        Args:
            polygon_files: unizone별 폴리곤 파일 경로 딕셔너리
        """
        self.graphs = {}
        self.community_results = {}
        self.centrality_results = {}
        self.nodes_data = {}
        self.edges_data = {}
        
        # 폴리곤 파일 경로 설정
        self.polygon_files = polygon_files or {
            'unizone1': 'data/spatial/polygon_162.gpkg',
            'unizone2': 'data/spatial/polygon_229.gpkg', 
            'unizone3': 'data/spatial/polygon_167.gpkg'
        }
        self.centroids = {}
    
    def load_polygon_centroids(self, unizone_type: str) -> pd.DataFrame:
        """
        폴리곤 파일을 로드하고 centroid 좌표를 계산
        
        Args:
            unizone_type: 'unizone1', 'unizone2', 'unizone3' 중 하나
            
        Returns:
            zone 코드, adm_nm, longitude, latitude 컬럼을 포함한 centroid 데이터
        """
        if unizone_type not in self.polygon_files:
            raise ValueError(f"지원되는 unizone_type: {list(self.polygon_files.keys())}")
        
        try:
            # 폴리곤 파일 로드
            gdf = gpd.read_file(self.polygon_files[unizone_type])
            
            # WGS84로 좌표계 변환
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            
            # centroid 계산
            gdf['centroid'] = gdf.geometry.centroid
            gdf['longitude'] = gdf.centroid.x
            gdf['latitude'] = gdf.centroid.y
            
            # unizone_type에 해당하는 컬럼 검증
            if unizone_type not in gdf.columns:
                raise ValueError(f"{unizone_type} 컬럼이 폴리곤 파일에 없습니다.")
            
            # 결과 데이터프레임 생성
            centroids_df = gdf[[unizone_type, 'adm_nm', 'longitude', 'latitude']].copy()
            centroids_df = centroids_df.rename(columns={unizone_type: 'zone'})
            
            # 캐시에 저장
            self.centroids[unizone_type] = centroids_df
            
            logger.info(f"✅ {unizone_type} 폴리곤 centroid 로드 완료: {len(centroids_df)}개 zone")
            logger.debug(f"좌표 범위 - 경도: {centroids_df['longitude'].min():.3f} ~ {centroids_df['longitude'].max():.3f}")
            logger.debug(f"좌표 범위 - 위도: {centroids_df['latitude'].min():.3f} ~ {centroids_df['latitude'].max():.3f}")
            
            return centroids_df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"폴리곤 파일을 찾을 수 없습니다: {self.polygon_files[unizone_type]}")
        except Exception as e:
            raise Exception(f"폴리곤 파일 로드 중 오류 발생: {str(e)}")
    
    def prepare_network_data(self, flow_instance, year: int, unizone_type: str, 
                            include_self_loops: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Flow 인스턴스의 converted_data를 네트워크 분석용 데이터로 변환
        
        Args:
            flow_instance: KTDB_PSN, KTDB_FreightTon, 또는 KED 클래스 인스턴스
            year: 분석할 연도
            unizone_type: 사용할 unizone 타입
            include_self_loops: self-loop 포함 여부
            
        Returns:
            (nodes_df, edges_df) 튜플
        """
        # 데이터 검증
        if not hasattr(flow_instance, 'converted_data') or not flow_instance.converted_data:
            raise ValueError("converted_data가 없습니다. change_zone()을 먼저 실행해주세요.")

        if year not in flow_instance.converted_data:
            raise ValueError(f"{year}년 데이터가 없습니다. 사용 가능한 연도: {list(flow_instance.converted_data.keys())}")

        # 원본 네트워크 데이터
        original_network_data = flow_instance.converted_data[year].copy()

        # Self-loop 데이터 추출
        self_loops = original_network_data[original_network_data['source'] == original_network_data['target']]
        self_loop_weights = self_loops.groupby('source')['weights'].sum().rename('self_loop')

        # 네트워크 데이터 준비
        if not include_self_loops:
            network_data = original_network_data[original_network_data['source'] != original_network_data['target']]
        else:
            network_data = original_network_data

        # centroid 데이터 로드
        if unizone_type not in self.centroids:
            self.load_polygon_centroids(unizone_type)
        centroids_df = self.centroids[unizone_type]

        # 노드 및 엣지 데이터 생성
        nodes_df, edges_df = self._create_nodes_edges_data(network_data, centroids_df, self_loop_weights)

        # 중심성 지수 계산 및 병합
        centrality_df = self.calculate_centrality(edges_df)
        nodes_df = self._merge_centrality_data(nodes_df, centrality_df)

        # 결과 저장
        self.nodes_data[year] = nodes_df
        self.edges_data[year] = edges_df

        logger.info(f"✅ 네트워크 데이터 준비 완료:")
        logger.info(f"   - 노드 수: {len(nodes_df)}")
        logger.info(f"   - 엣지 수: {len(edges_df)}")
        logger.info(f"   - 총 가중치: {edges_df['weights'].sum():,.0f}")
        logger.info(f"   - 중심성 지수 계산 완료")

        return nodes_df, edges_df
    
    def _create_nodes_edges_data(self, network_data: pd.DataFrame, centroids_df: pd.DataFrame, 
                                self_loop_weights: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """노드와 엣지 데이터를 생성하고 좌표 정보를 병합"""
        
        # 노드 데이터 생성
        all_nodes = pd.concat([
            network_data[['source']].rename(columns={'source': 'zone'}),
            network_data[['target']].rename(columns={'target': 'zone'})
        ]).drop_duplicates()

        nodes_df = all_nodes.merge(centroids_df, on='zone', how='left')

        # Self-loop 가중치를 노드 속성으로 추가
        if self_loop_weights is not None:
            nodes_df = nodes_df.merge(self_loop_weights, left_on='zone', right_index=True, how='left')
            nodes_df['self_loop'] = nodes_df['self_loop'].fillna(0)
        else:
            nodes_df['self_loop'] = 0

        # 좌표가 없는 노드 처리
        missing_coords = nodes_df[nodes_df['longitude'].isna()]
        if len(missing_coords) > 0:
            logger.warning(f"⚠️ 좌표 정보가 없는 {len(missing_coords)}개 노드: {missing_coords['zone'].tolist()}")
            nodes_df = nodes_df.dropna(subset=['longitude', 'latitude'])

        # 엣지 데이터에 좌표 및 라벨 정보 추가
        edges_df = network_data.copy()

        # source 정보 병합
        edges_df = edges_df.merge(
            centroids_df.rename(columns={
                'zone': 'source', 'longitude': 'source_lng', 
                'latitude': 'source_lat', 'adm_nm': 'source_label'
            }),
            on='source', how='left'
        )

        # target 정보 병합
        edges_df = edges_df.merge(
            centroids_df.rename(columns={
                'zone': 'target', 'longitude': 'target_lng', 
                'latitude': 'target_lat', 'adm_nm': 'target_label'
            }),
            on='target', how='left'
        )

        # 좌표가 없는 엣지 제거
        edges_df = edges_df.dropna(subset=['source_lng', 'source_lat', 'target_lng', 'target_lat'])

        # RSI 계산 추가
        edges_df = self._calculate_rsi(edges_df)

        return nodes_df, edges_df
    
    def _calculate_rsi(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """
        RSI (Relative Strength Index) 계산
        
        RSI_ij = F_ij / (∑_i ∑_j F_ij) * 100
        각 플로우의 가중치를 전체 플로우 합계로 나눈 상대적 비율(%)
        """
        total_flow = edges_df['weights'].sum()
        
        if total_flow > 0:
            edges_df['RSI'] = edges_df['weights'] / total_flow * 100
        else:
            edges_df['RSI'] = 0
        
        return edges_df
    
    def create_graph_from_dataframe(self, df: pd.DataFrame, include_self: bool = True, 
                                   thres: float = 0) -> nx.DiGraph:
        """
        DataFrame으로부터 방향 그래프 생성
        
        Args:
            df: source, target, weights 컬럼을 포함한 DataFrame
            include_self: 셀프 루프 포함 여부
            thres: 가중치 임계값
            
        Returns:
            생성된 NetworkX 방향 그래프
        """
        # 필터링
        if not include_self:
            df_filtered = df[df["source"] != df["target"]]
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered[df_filtered['weights'] >= thres]
        
        # 그래프 생성
        G = nx.from_pandas_edgelist(
            df_filtered, 
            source="source", 
            target="target", 
            edge_attr="weights", 
            create_using=nx.DiGraph()
        )
        
        return G
    
    def create_graphs_from_data(self, converted_data: Dict[int, pd.DataFrame], 
                               include_self: bool = True, thres: float = 0) -> Dict[int, nx.DiGraph]:
        """
        변환된 데이터로부터 모든 연도의 그래프 생성
        
        Args:
            converted_data: {year: DataFrame} 형태의 변환된 데이터
            include_self: 셀프 루프 포함 여부
            thres: 가중치 임계값
            
        Returns:
            {year: nx.DiGraph} 형태의 그래프 딕셔너리
        """
        if not converted_data:
            raise ValueError("변환된 데이터가 없습니다.")
        
        graphs = {}
        for year, df in converted_data.items():
            try:
                graphs[year] = self.create_graph_from_dataframe(df, include_self, thres)
                logger.info(f"✅ {year}년 그래프 생성 완료: {len(graphs[year].nodes())} 노드, {len(graphs[year].edges())} 엣지")
            except Exception as e:
                logger.error(f"❌ {year}년 그래프 생성 실패: {e}")
                continue
        
        self.graphs = graphs
        return graphs
    
    def community_detection(self, G: nx.DiGraph, method: str = "Leiden") -> pd.DataFrame:
        """
        커뮤니티 탐지 실행
        
        Args:
            G: NetworkX 방향 그래프
            method: 커뮤니티 탐지 방법 ("Infomap", "Infomap_igraph", "Louvain", "Leiden", "Leiden_igraph)
            
        Returns:
            노드와 커뮤니티 ID를 포함한 DataFrame
        """
        if method == "Infomap":
            return self._infomap_detection(G)
        elif method == "Infomap_igraph":
            return self._infomap_igraph_detection(G)
        elif method == "Louvain":
            return self._louvain_detection(G)
        elif method == "Leiden":
            return self._leiden_detection(G)
        elif method == "Leiden_igraph":
            return self._leiden_igraph_detection(G)
        else:
            raise ValueError(f"지원되지 않는 커뮤니티 탐지 알고리즘 입니다: {method}")
    
    def _infomap_detection(self, G: nx.DiGraph) -> pd.DataFrame:
        """Infomap 커뮤니티 탐지"""
        im = Infomap(directed=True, two_level=True, no_self_links = True)
        im.add_networkx_graph(G, weight='weights')
        im.run()
        result = im.get_dataframe(columns=["name", "module_id"]).sort_values(by='name')
        return result
    
    def _infomap_igraph_detection(self, G: nx.DiGraph) -> pd.DataFrame:
        """iGraph Infomap 커뮤니티 탐지"""
        G_ig = ig.Graph.from_networkx(G)
        partition = G_ig.community_infomap(edge_weights='weights')
        result = pd.DataFrame({
            "name": G_ig.vs["_nx_name"],
            "module_id": partition.membership
        }).sort_values(by='name')
        return result
    
    def _louvain_detection(self, G: nx.DiGraph) -> pd.DataFrame:
        """Louvain 커뮤니티 탐지"""
        communities = nx.community.louvain_communities(G, weight='weights', seed=42)
        result = pd.DataFrame([
            {"name": node, "module_id": i}
            for i, community in enumerate(communities)
            for node in community
        ]).sort_values(by='name')
        return result
    
    def _leiden_detection(self, G: nx.DiGraph) -> pd.DataFrame:
        """Leiden 커뮤니티 탐지"""
        UG = G.to_undirected()
        G_ig = ig.Graph.from_networkx(UG)
        partition = leidenalg.find_partition(
            G_ig, weights="weights", 
            partition_type=leidenalg.ModularityVertexPartition
        )
        result = pd.DataFrame({
            "name": G_ig.vs["_nx_name"],
            "module_id": partition.membership
        }).sort_values(by='name')
        return result
    
    def _leiden_igraph_detection(self, G: nx.DiGraph) -> pd.DataFrame:
        ## Leiden(igraph) 커뮤니티 탐지
        UG = G.to_undirected()
        G_ig = ig.Graph.from_networkx(UG)
        partition = G_ig.community_leiden(objective_function = 'modularity',
                                          weights = 'weights',
                                          n_iterations = 100)
        result = pd.DataFrame({
            "name": G_ig.vs["_nx_name"],
            "module_id": partition.membership
        }).sort_values(by='name')
        return result
    
    def analyze_communities_all_years(self, graphs: Optional[Dict[int, nx.DiGraph]] = None, 
                                     method: str = "Leiden") -> Dict[int, pd.DataFrame]:
        """
        모든 연도의 그래프에 대해 커뮤니티 분석 수행
        
        Args:
            graphs: 그래프 딕셔너리 (None이면 내부 저장된 그래프 사용)
            method: 커뮤니티 탐지 방법
            
        Returns:
            {year: DataFrame} 형태의 커뮤니티 분석 결과
        """
        if graphs is None:
            graphs = self.graphs
        
        if not graphs:
            raise ValueError("그래프가 없습니다.")
        
        community_results = {}
        for year, graph in graphs.items():
            try:
                community_results[year] = self.community_detection(graph, method)
                logger.info(f"✅ {year}년 커뮤니티 분석 완료: {community_results[year]['module_id'].nunique()} 개 커뮤니티")
            except Exception as e:
                logger.error(f"❌ {year}년 커뮤니티 분석 실패: {e}")
                continue
        
        self.community_results = community_results
        return community_results
    
    def calculate_centrality(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """
        엣지 데이터로부터 다양한 노드 중심성 지수를 계산
        
        계산하는 중심성 지수:
        - in_degree: 가중치 기반 유입 연결도
        - out_degree: 가중치 기반 유출 연결도
        - total_degree: in_degree + out_degree
        - pagerank: 가중치 기반 페이지랭크 중심성
        - DII (Dominance Index): 유입+유출 합계의 평균 대비 비율
        - EI (Entropy Index): 유입 연결의 다양성 (엔트로피 기반)
        - NSI (Node Symmetry Index): 유입과 유출 연결의 균등성
        
        Args:
            edges_df: source, target, weights 컬럼을 포함한 엣지 데이터
            
        Returns:
            노드별 중심성 지수를 포함한 데이터프레임
        """

        # NetworkX 그래프 생성
        G = nx.from_pandas_edgelist(
            edges_df, 
            source="source", 
            target="target", 
            edge_attr="weights", 
            create_using=nx.DiGraph()
        )
        
        # 기본 중심성 지수 계산
        in_degrees = dict(G.in_degree(weight='weights'))
        out_degrees = dict(G.out_degree(weight='weights'))
        pagerank = nx.pagerank(G, weight='weights')
        
        # DataFrame으로 변환
        centrality_df = pd.DataFrame({
            'zone': list(in_degrees.keys()),
            'in_degree': list(in_degrees.values()),
            'out_degree': list(out_degrees.values()),
            'pagerank': [pagerank[node] for node in in_degrees.keys()]
        })
        
        # 총 degree 계산
        centrality_df['total_degree'] = centrality_df['in_degree'] + centrality_df['out_degree']

        # 고급 중심성 지수 계산
        centrality_df = self._calculate_advanced_centrality(centrality_df, edges_df)
        
        return centrality_df
    
    def _calculate_advanced_centrality(self, centrality_df: pd.DataFrame, 
                                      edges_df: pd.DataFrame) -> pd.DataFrame:
        """
        고급 중심성 지수 (DII, EI, NSI)를 계산하여 centrality_df에 추가
        
        Args:
            centrality_df: 기본 중심성 지수가 포함된 데이터프레임
            edges_df: 엣지 데이터
        """
        # DII 계산: 각 노드의 inflow + outflow 합을 평균값으로 정규화
        inflow = edges_df.groupby('target')['weights'].sum().rename('inflow')
        outflow = edges_df.groupby('source')['weights'].sum().rename('outflow')
        inflow_outflow = pd.concat([inflow, outflow], axis=1).fillna(0)
        flow_sum_node = inflow_outflow.sum(axis=1)
        mean_flow_sum_node = flow_sum_node.mean()
        DII = flow_sum_node / mean_flow_sum_node if mean_flow_sum_node > 0 else flow_sum_node

        # EI 계산: 노드로 유입되는 연결의 엔트로피 (연결 다양성 지표)
        inflow_details = edges_df.groupby(['target', 'source'])['weights'].sum().reset_index()
        inflow_details['pij'] = inflow_details.groupby('target')['weights'].transform(lambda x: x / x.sum())
        inflow_details = inflow_details[inflow_details['pij'] > 0]
        
        EI = -inflow_details.groupby('target').apply(
            lambda x: np.sum(x['pij'] * np.log(x['pij'])) / np.log(len(x)) if len(x) > 1 else 0
        ).rename('EI')

        # NSI 계산: 유입/유출의 비대칭성 정도 (inflow - outflow) / (inflow + outflow)
        total_flow = inflow_outflow['inflow'] + inflow_outflow['outflow']
        NSI = (inflow_outflow['inflow'] - inflow_outflow['outflow']) / total_flow
        NSI = NSI.replace([np.inf, -np.inf], 0).fillna(0)

        # 중심성 데이터 병합
        centrality_df = centrality_df.set_index('zone')
        centrality_df['DII'] = DII
        centrality_df['EI'] = EI
        centrality_df['NSI'] = NSI
        centrality_df.reset_index(inplace=True)

        return centrality_df
    
    def _merge_centrality_data(self, nodes_df: pd.DataFrame, 
                              centrality_df: pd.DataFrame) -> pd.DataFrame:
        """중심성 데이터를 노드 데이터에 병합"""
        nodes_df = nodes_df.merge(centrality_df, on='zone', how='left')
        
        # 중심성 지수가 없는 노드는 0으로 채움
        centrality_cols = ['in_degree', 'out_degree', 'pagerank', 'total_degree', 'DII', 'EI', 'NSI']
        for col in centrality_cols:
            if col in nodes_df.columns:
                nodes_df[col] = nodes_df[col].fillna(0)
        
        # Self-containment 지수 계산
        # self-containment_in = self-loop / (indegree + self-loop)
        denominator_in = nodes_df['in_degree'] + nodes_df['self_loop']
        nodes_df['self_containment_in'] = np.where(
            denominator_in > 0, 
            nodes_df['self_loop'] / denominator_in, 
            0
        )
        
        # self-containment_out = self-loop / (outdegree + self-loop)
        denominator_out = nodes_df['out_degree'] + nodes_df['self_loop']
        nodes_df['self_containment_out'] = np.where(
            denominator_out > 0, 
            nodes_df['self_loop'] / denominator_out, 
            0
        )
        
        return nodes_df
    
    def _compute_recursive_metrics(self, G: nx.DiGraph, directed: bool = True, 
                                  weight_attr: str = 'weights') -> Tuple[Dict, Dict]:
        """
        재귀 중심성과 재귀 권력 계산
        
        Args:
            G: NetworkX 그래프
            directed: 방향 그래프 여부
            weight_attr: 가중치 속성명
            
        Returns:
            (recursive_centrality, recursive_power) 딕셔너리 튜플
        """
        if directed:
            if not G.is_directed():
                raise ValueError("directed=True일 때는 방향 그래프만 사용할 수 있습니다.")
            
            # In-degree 중심성 계산
            in_degree_centrality = {
                node: sum(G[u][node].get(weight_attr, 1.0) for u in G.predecessors(node))
                for node in G.nodes()
            }
            
            recursive_centrality = {}
            recursive_power = {}
            
            for i in G.nodes():
                rc = 0
                rp = 0
                for j in G.successors(i):
                    weight = G[i][j].get(weight_attr, 1.0)
                    dc_j = in_degree_centrality.get(j, 0.0)
                    if dc_j > 0:
                        rc += weight * dc_j
                        rp += weight / dc_j
                recursive_centrality[i] = rc
                recursive_power[i] = rp
        else:
            # 무방향 그래프 처리
            if G.is_directed():
                G = G.to_undirected()
            
            degree_centrality = {
                node: sum(data.get(weight_attr, 1.0) for _, _, data in G.edges(node, data=True))
                for node in G.nodes()
            }
            
            recursive_centrality = {}
            recursive_power = {}
            
            for i in G.nodes():
                rc = 0
                rp = 0
                for j in G.neighbors(i):
                    weight = G[i][j].get(weight_attr, 1.0)
                    dc_j = degree_centrality.get(j, 0.0)
                    if dc_j > 0:
                        rc += weight * dc_j
                        rp += weight / dc_j
                recursive_centrality[i] = rc
                recursive_power[i] = rp
        
        return recursive_centrality, recursive_power
    
    def calculate_node_centrality_from_graph(self, G: nx.DiGraph) -> pd.DataFrame:
        """
        NetworkX 그래프로부터 노드 중심성 지수 계산 (그래프 분석용)
        
        Args:
            G: NetworkX 방향 그래프
            
        Returns:
            중심성 지수를 포함한 DataFrame
        """
        # 기본 중심성 지수
        in_degrees = dict(G.in_degree(weight='weights'))
        out_degrees = dict(G.out_degree(weight='weights'))
        pagerank = nx.pagerank(G, weight='weights')
        
        # 재귀 중심성
        rc, rp = self._compute_recursive_metrics(G, directed=True, weight_attr='weights')
        
        # DataFrame 생성
        nodes = list(G.nodes())
        result = pd.DataFrame({
            'name': nodes,
            'InDegrees': [in_degrees.get(node, 0) for node in nodes],
            'OutDegrees': [out_degrees.get(node, 0) for node in nodes],
            'PageRank': [pagerank.get(node, 0) for node in nodes],
            'RecursiveCentrality': [rc.get(node, 0) for node in nodes],
            'RecursivePower': [rp.get(node, 0) for node in nodes]
        }).sort_values(by='name')
        
        return result
    
    def export_network_data(self, filename: str, years: List[int]):
        """
        처리된 네트워크 데이터를 엑셀 파일로 내보내기
        
        Args:
            filename: 저장할 파일명
            years: 내보낼 연도 리스트
        """
        if not self.nodes_data and not self.edges_data:
            raise ValueError("처리된 데이터가 없습니다. prepare_network_data를 먼저 실행해주세요.")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 각 연도별 노드 데이터 저장
            for year in years:
                if year in self.nodes_data:
                    self.nodes_data[year].to_excel(writer, sheet_name=f'Nodes_{year}', index=False)
                
                if year in self.edges_data:
                    self.edges_data[year].to_excel(writer, sheet_name=f'Edges_{year}', index=False)
        
        logger.info(f"✅ 네트워크 데이터가 '{filename}' 파일로 저장되었습니다.")
    
    def get_network_data(self, year: int) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        특정 연도의 네트워크 데이터 반환
        
        Args:
            year: 조회할 연도
            
        Returns:
            (nodes_df, edges_df) 튜플 또는 (None, None)
        """
        nodes_df = self.nodes_data.get(year)
        edges_df = self.edges_data.get(year)
        return nodes_df, edges_df


class CommunityMapper:
    """커뮤니티 매핑 및 연속성 분석 클래스"""
    
    @staticmethod
    def calculate_jaccard_similarity(set1: set, set2: set) -> float:
        """두 집합 간의 Jaccard 유사도 계산"""
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def map_communities_between_years(self, community_results: Dict[int, pd.DataFrame], 
                                     reference_year: int, target_year: int, 
                                     similarity_threshold: float = 0.3) -> Tuple[pd.DataFrame, Dict]:
        """
        두 시점의 커뮤니티 결과를 비교하여 유사한 커뮤니티에 동일한 번호 부여
        
        Args:
            community_results: 커뮤니티 분석 결과 딕셔너리
            reference_year: 기준 연도
            target_year: 조정할 연도
            similarity_threshold: 최소 유사도 임계값
            
        Returns:
            (조정된 target_year DataFrame, 매핑 정보 딕셔너리)
        """
        if reference_year not in community_results or target_year not in community_results:
            raise ValueError(f"지정된 연도({reference_year}, {target_year})의 커뮤니티 결과가 없습니다.")
        
        # 기준 연도와 대상 연도의 커뮤니티 결과
        ref_df = community_results[reference_year].copy()
        target_df = community_results[target_year].copy()
        
        logger.info(f"커뮤니티 매핑: {reference_year}년(기준) → {target_year}년(조정)")
        logger.info(f"기준 연도 커뮤니티 수: {ref_df['module_id'].nunique()}")
        logger.info(f"대상 연도 커뮤니티 수: {target_df['module_id'].nunique()}")
        
        # 커뮤니티 구성원 정보 생성
        ref_communities = self._get_community_members(ref_df)
        target_communities = self._get_community_members(target_df)
        
        # 유사도 계산 및 매핑
        all_similarities = self._calculate_all_similarities(
            target_communities, ref_communities, similarity_threshold
        )
        
        # 최적 매핑 선택
        final_mappings = self._select_optimal_mappings(all_similarities)
        
        # 새 번호 할당
        final_mappings = self._assign_new_numbers(
            final_mappings, target_communities, ref_communities
        )
        
        # DataFrame 업데이트
        adjusted_target_df = self._update_dataframe(target_df, final_mappings)
        
        self._log_mapping_results(final_mappings, adjusted_target_df)
        
        return adjusted_target_df, final_mappings
    
    def _get_community_members(self, df: pd.DataFrame) -> Dict[int, set]:
        """커뮤니티별 구성원 딕셔너리 생성"""
        communities = {}
        for comm_id in sorted(df['module_id'].unique()):
            members = set(df[df['module_id'] == comm_id]['name'].values)
            communities[comm_id] = members
        return communities
    
    def _calculate_all_similarities(self, target_communities: Dict[int, set], 
                                   ref_communities: Dict[int, set], 
                                   similarity_threshold: float) -> List[Dict]:
        """모든 커뮤니티 간 유사도 계산"""
        all_similarities = []
        
        for target_comm_id, target_members in target_communities.items():
            best_similarity = 0
            best_ref_id = None
            best_ref_size = 0
            
            for ref_comm_id, ref_members in ref_communities.items():
                similarity = self.calculate_jaccard_similarity(target_members, ref_members)
                
                if similarity >= similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_ref_id = ref_comm_id
                    best_ref_size = len(ref_members)
            
            if best_ref_id is not None:
                all_similarities.append({
                    'target_id': target_comm_id,
                    'ref_id': best_ref_id,
                    'similarity': best_similarity,
                    'target_size': len(target_members),
                    'ref_size': best_ref_size
                })
        
        # 유사도 순으로 정렬
        all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return all_similarities
    
    def _select_optimal_mappings(self, all_similarities: List[Dict]) -> Dict[int, Dict]:
        """최적 매핑 선택 (greedy algorithm)"""
        used_ref_ids = set()
        used_target_ids = set()
        final_mappings = {}
        
        for candidate in all_similarities:
            target_id = candidate['target_id']
            ref_id = candidate['ref_id']
            similarity = candidate['similarity']
            
            if target_id not in used_target_ids and ref_id not in used_ref_ids:
                final_mappings[target_id] = {
                    'new_id': ref_id,
                    'similarity': similarity,
                    'type': 'mapped'
                }
                used_ref_ids.add(ref_id)
                used_target_ids.add(target_id)
        
        return final_mappings
    
    def _assign_new_numbers(self, final_mappings: Dict[int, Dict], 
                           target_communities: Dict[int, set], 
                           ref_communities: Dict[int, set]) -> Dict[int, Dict]:
        """매핑되지 않은 커뮤니티들에 새 번호 할당"""
        max_ref_id = max(ref_communities.keys())
        next_new_id = max_ref_id + 1
        
        for target_comm_id in target_communities.keys():
            if target_comm_id not in final_mappings:
                final_mappings[target_comm_id] = {
                    'new_id': next_new_id,
                    'similarity': 0,
                    'type': 'new'
                }
                next_new_id += 1
        
        return final_mappings
    
    def _update_dataframe(self, target_df: pd.DataFrame, 
                         final_mappings: Dict[int, Dict]) -> pd.DataFrame:
        """대상 연도 DataFrame 업데이트"""
        adjusted_target_df = target_df.copy()
        
        for target_comm_id, mapping_info in final_mappings.items():
            mask = adjusted_target_df['module_id'] == target_comm_id
            adjusted_target_df.loc[mask, 'module_id'] = mapping_info['new_id']
        
        return adjusted_target_df
    
    def _log_mapping_results(self, final_mappings: Dict[int, Dict], 
                            adjusted_target_df: pd.DataFrame):
        """매핑 결과 로깅"""
        mapped_count = sum(1 for info in final_mappings.values() if info['type'] == 'mapped')
        new_count = sum(1 for info in final_mappings.values() if info['type'] == 'new')
        
        logger.info(f"매핑된 커뮤니티: {mapped_count}개")
        logger.info(f"새 커뮤니티: {new_count}개")
        logger.info(f"최종 커뮤니티 수: {adjusted_target_df['module_id'].nunique()}개")
    
    def map_communities_sequential(self, community_results: Dict[int, pd.DataFrame], 
                                  years_list: List[int], 
                                  similarity_threshold: float = 0.3) -> Tuple[Dict[int, pd.DataFrame], Dict]:
        """
        여러 연도에 대해 순차적으로 커뮤니티 매핑 수행
        
        Args:
            community_results: 커뮤니티 분석 결과
            years_list: 매핑할 연도 리스트 (첫 번째가 기준)
            similarity_threshold: 최소 유사도 임계값
            
        Returns:
            (조정된 커뮤니티 결과, 매핑 정보)
        """
        if len(years_list) < 2:
            raise ValueError("최소 2개 연도가 필요합니다.")
        
        logger.info(f"순차적 커뮤니티 매핑 시작: {' → '.join(map(str, years_list))}")
        
        all_mapping_info = {}
        adjusted_results = community_results.copy()
        
        # 첫 번째 연도를 기준으로 설정
        reference_year = years_list[0]
        
        # 나머지 연도들을 순차적으로 매핑
        for i in range(1, len(years_list)):
            target_year = years_list[i]
            
            adjusted_df, mapping_info = self.map_communities_between_years(
                adjusted_results, reference_year, target_year, similarity_threshold
            )
            
            # 결과 업데이트
            adjusted_results[target_year] = adjusted_df
            all_mapping_info[target_year] = mapping_info
            
            # 다음 매핑을 위해 기준 연도 업데이트
            reference_year = target_year
        
        logger.info("모든 연도의 커뮤니티 매핑이 완료되었습니다!")
        
    def analyze_centrality_all_years(self, graphs: Optional[Dict[int, nx.DiGraph]] = None) -> Dict[int, pd.DataFrame]:
        """
        모든 연도의 그래프에 대해 중심성 분석 수행 (그래프 기반)
        
        Args:
            graphs: 그래프 딕셔너리 (None이면 내부 저장된 그래프 사용)
            
        Returns:
            {year: DataFrame} 형태의 중심성 분석 결과
        """
        if graphs is None:
            graphs = self.graphs
        
        if not graphs:
            raise ValueError("그래프가 없습니다.")
        
        centrality_results = {}
        for year, graph in graphs.items():
            try:
                centrality_results[year] = self.calculate_node_centrality_from_graph(graph)
                logger.info(f"✅ {year}년 중심성 분석 완료")
            except Exception as e:
                logger.error(f"❌ {year}년 중심성 분석 실패: {e}")
                continue
        
        self.centrality_results = centrality_results
        return centrality_results