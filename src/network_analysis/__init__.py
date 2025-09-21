"""
Network Analysis Library
========================

한국 교통 데이터베이스(KTDB) 및 기업체 이전 데이터(KED)를 활용한 
네트워크 분석 및 시각화 라이브러리

주요 기능:
- 다양한 데이터 소스 지원 (KTDB 인구이동, 화물이동, KED)
- 네트워크 분석 (중심성, 커뮤니티 탐지)
- 계층적 클러스터링 분석
- 정적/인터랙티브 시각화
- 공간 분석 및 지도 매핑

사용 예시:
    >>> from network_analysis import KTDB_PSN, NetworkAnalyzer, NetworkVisualizer
    >>> 
    >>> # 데이터 로드 및 전처리
    >>> ktdb = KTDB_PSN('zonecode.csv')
    >>> ktdb.load_multiple_years([2020, 2021, 2022])
    >>> data = ktdb.change_zone('unizone3', 'total')
    >>> 
    >>> # 네트워크 분석
    >>> analyzer = NetworkAnalyzer()
    >>> graphs = analyzer.create_graphs_from_data(data)
    >>> centrality_results = analyzer.analyze_centrality_all_years(graphs)
    >>> 
    >>> # 시각화
    >>> viz = NetworkVisualizer()
    >>> fig, ax = viz.visualize_static_network(ktdb, 2022, 'unizone3')

저자: Network Analysis Team
라이센스: MIT
"""

# 버전 정보
__version__ = "1.0.0"
__author__ = "Network Analysis Team"

# 핵심 모듈 import
from .data_processors import KTDB_PSN, KTDB_FreightTon, KED, SIMS
from .network_analyzer import NetworkAnalyzer


# 메인 클래스들을 패키지 레벨에서 접근 가능하게 설정
__all__ = [
    # 데이터 처리
    'KTDB_PSN',
    'KTDB_FreightTon', 
    'KED',
    'SIMS',
    
    # 분석
    'NetworkAnalyzer',
    'HierarchicalClusterer',
    'KMeansClusterer',
    
    # 시각화
    'FlowMapper',
    'CommunityMapper',
    
    # 유틸리티
    'FontManager',
    'DataExporter'
]

# 설정값
DEFAULT_CONFIG = {
    'polygon_files': {
        'unizone1': 'data/spatial/polygon_162.gpkg',
        'unizone2': 'data/spatial/polygon_229.gpkg', 
        'unizone3': 'data/spatial/polygon_167.gpkg'
    },
    'font_settings': {
        'use_korean': True,
        'auto_detect': True
    },
    'visualization': {
        'default_figsize': (12, 10),
        'default_dpi': 300,
        'color_schemes': {
            'nodes': 'viridis',
            'edges': 'red_yellow_gradient'
        }
    },
    'analysis': {
        'community_detection_method': 'Leiden',
        'clustering_method': 'hierarchical',
        'max_clusters': 10
    }
}

def get_config():
    """라이브러리 설정값 반환"""
    return DEFAULT_CONFIG.copy()

def set_config(**kwargs):
    """라이브러리 설정값 업데이트"""
    for key, value in kwargs.items():
        if key in DEFAULT_CONFIG:
            if isinstance(DEFAULT_CONFIG[key], dict) and isinstance(value, dict):
                DEFAULT_CONFIG[key].update(value)
            else:
                DEFAULT_CONFIG[key] = value

# 로거 설정
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

def setup_logging(level=logging.INFO):
    """로깅 설정"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
