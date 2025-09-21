"""
데이터 전처리 모듈

KTDB 및 KED 데이터를 로드하고 네트워크 분석용으로 변환하는 클래스들을 제공합니다.
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class BaseDataProcessor(ABC):
    """데이터 처리기 기본 클래스"""
    
    def __init__(self, zonecode_path: Optional[str] = None):
        """
        기본 초기화
        
        Args:
            zonecode_path: zonecode.csv 파일 경로
        """
        self.zonecode_path = zonecode_path
        self.zone_df = None
        self.dataframes = {}
        self.converted_data = {}
        
        if zonecode_path:
            self._load_zonecode()
    
    def _load_zonecode(self):
        """Zonecode 파일 로드"""
        try:
            self.zone_df = pd.read_csv(self.zonecode_path, encoding='utf-8')
            logger.info(f"Zonecode 파일 로드 완료: {len(self.zone_df)} 개 zone")
        except Exception as e:
            logger.error(f"Zonecode 파일 로드 실패: {e}")
            raise
    
    @abstractmethod
    def read_data_file(self, year: int, path: str = '.') -> pd.DataFrame:
        """데이터 파일 읽기 (각 클래스에서 구현)"""
        pass
    
    def load_multiple_years(self, years: List[int], path: str = '.') -> Dict[int, pd.DataFrame]:
        """
        여러 연도 데이터를 한 번에 로드
        
        Args:
            years: 로드할 연도 리스트
            path: 파일 경로
            
        Returns:
            성공적으로 로드된 데이터 딕셔너리
        """
        successful_loads = {}
        failed_loads = []
        
        for year in years:
            try:
                df = self.read_data_file(year, path)
                successful_loads[year] = df
                logger.info(f"✅ {year}년 데이터 로드 성공")
            except Exception as e:
                failed_loads.append(year)
                logger.warning(f"⚠️ {year}년 데이터 로드 실패: {e}")
        
        if successful_loads:
            logger.info(f"총 {len(successful_loads)}개 연도 데이터 로드 완료: {list(successful_loads.keys())}")
        
        if failed_loads:
            logger.warning(f"실패한 연도: {failed_loads}")
            
        return successful_loads
    
    def change_zone(self, unizone_col: str, weight_col: str, 
                   reg_nm: Optional[List[str]] = None) -> Dict[int, pd.DataFrame]:
        """
        Zone 변환 및 가중치 집계
        
        Args:
            unizone_col: 변환할 unizone 컬럼명
            weight_col: 집계할 가중치 컬럼명
            reg_nm: 필터링할 지역명 리스트
            
        Returns:
            변환된 데이터 딕셔너리
        """
        if self.zone_df is None:
            raise ValueError("zonecode 데이터가 로드되지 않았습니다.")
        
        if not self.dataframes:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        zone_keys = ['zone248', 'zone249', 'zone251', 'zone252', 'zone250']
        converted_data = {}
        
        for year, df in self.dataframes.items():
            try:
                # Zone key 결정
                n_unique = df['o_sgg'].nunique()
                key_col = f"zone{n_unique}"
                
                if key_col not in zone_keys:
                    logger.warning(f"⚠️ {year}년: 유효한 zone key가 없습니다. (유니크 o_sgg 개수: {n_unique})")
                    continue
                
                # Zone 변환
                converted_df = self._convert_zones(df, key_col, unizone_col, weight_col, reg_nm)
                converted_data[year] = converted_df
                
            except Exception as e:
                logger.error(f"❌ {year}년 zone 변환 실패: {e}")
                continue
        
        self.converted_data = converted_data
        return converted_data
    
    def _convert_zones(self, df: pd.DataFrame, key_col: str, unizone_col: str, 
                      weight_col: str, reg_nm: Optional[List[str]]) -> pd.DataFrame:
        """Zone 변환 로직"""
        # o_sgg 변환
        df_merged_o = df.merge(
            self.zone_df[[key_col, unizone_col]], 
            how='left', left_on='o_sgg', right_on=key_col
        )
        df_merged_o = df_merged_o.rename(columns={unizone_col: 'o_sgg_unizone'})
        
        # d_sgg 변환
        df_merged_od = df_merged_o.merge(
            self.zone_df[[key_col, unizone_col]], 
            how='left', left_on='d_sgg', right_on=key_col
        )
        df_merged_od = df_merged_od.rename(columns={unizone_col: 'd_sgg_unizone'})
        
        # 지역 필터링
        if reg_nm is not None:
            zone_filter = self.zone_df[self.zone_df['sd_nm2'].isin(reg_nm)]
            valid_zones = zone_filter[unizone_col].unique()
            df_merged_od = df_merged_od[
                (df_merged_od['o_sgg_unizone'].isin(valid_zones)) & 
                (df_merged_od['d_sgg_unizone'].isin(valid_zones))
            ]
        
        # 집계
        grouped = df_merged_od.groupby(
            ['o_sgg_unizone', 'd_sgg_unizone'], 
            dropna=False
        )[weight_col].sum().reset_index()
        
        # 컬럼명 정리
        grouped = grouped.rename(columns={
            'o_sgg_unizone': 'source',
            'd_sgg_unizone': 'target',
            weight_col: 'weights'
        })
        
        return grouped[['source', 'target', 'weights']]


class KTDB_PSN(BaseDataProcessor):
    """KTDB 인구이동 목적 데이터 처리 클래스"""
    
    def read_data_file(self, year: int, path: str = '.') -> pd.DataFrame:
        """
        KTDB 인구이동 데이터 파일 읽기
        
        Args:
            year: 연도 (2005-2022)
            path: 파일 경로
            
        Returns:
            읽어온 데이터프레임
        """
        if year < 2005 or year > 2023:
            raise ValueError("지원되는 연도는 2005년부터 2022년까지입니다.")
        
        filename = f"{path}/sgg_ktdb_od_psn_obj_{year}.csv"
        
        # 연도별 컬럼 구조 정의
        if year <= 2009:
            cols = ["o_sgg", "d_sgg", "cmt", "bsn", "bhome", "sch", "etc", "total"]
        elif year <= 2015:
            cols = ["o_sgg", "d_sgg", "cmt", "sch", "bsn", "shp", "bhome", "lsg", "etc", "total"]
        elif year == 2022:
            cols = ["o_sd", "d_sd", "o_sgg", "d_sgg", "cmt", "sch", "bsn", "bhome", "etc", "total"]
        else:
            cols = ["o_sd", "d_sd", "o_sgg", "d_sgg", "cmt", "sch", "bsn", "shp", "bhome", "lsg", "etc", "total", "trash"]
        
        try:
            df = pd.read_csv(filename, names=cols, header=0, encoding='utf-8')
            df['total_excl_bhome'] = df['total'] - df['bhome']
            self.dataframes[year] = df
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filename}")


class KTDB_FreightTon(BaseDataProcessor):
    """KTDB 화물 물동량 데이터 처리 클래스"""
    
    def read_data_file(self, year: int, path: str = '.') -> pd.DataFrame:
        """
        KTDB 화물 데이터 파일 읽기
        
        Args:
            year: 연도 (2005-2022)
            path: 파일 경로
            
        Returns:
            읽어온 데이터프레임
        """
        if year < 2005 or year > 2023:
            raise ValueError("지원되는 연도는 2005년부터 2022년까지입니다.")
        
        filename = f"{path}/sgg_ktdb_od_fre_ton_{year}.csv"
        
        try:
            df = pd.read_csv(filename, header=0, encoding='utf-8')
            self.dataframes[year] = df
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filename}")


class KED(BaseDataProcessor):
    """KED 기업 간 거래 데이터 처리 클래스"""
    
    def read_data_file(self, year: int, path: str = '.') -> pd.DataFrame:
        """KED는 load_ked_data 메서드를 사용하므로 여기서는 구현하지 않음"""
        raise NotImplementedError("KED 클래스는 load_ked_data 메서드를 사용하세요.")
    
    def load_ked_data(self, years: List[int], file_path: str) -> Dict[int, pd.DataFrame]:
        """
        KED 기업 간 거래 데이터 로드
        
        Args:
            years: 로드할 연도 리스트
            file_path: KED 데이터 파일 경로
            
        Returns:
            연도별 데이터 딕셔너리
        """
        try:
            data = pd.read_csv(file_path, sep='|')
            logger.info(f"✅ KED 데이터 로드 성공: {data.shape}")
            logger.info(f"연도 범위: {data['year'].min()} ~ {data['year'].max()}")
            
            for selected_year in years:
                year_data = data[data['year'] == selected_year].copy()
                self.dataframes[selected_year] = year_data
                logger.info(f"{selected_year}년 데이터 입력 성공: 관측치 수 {len(year_data)}개")
                
            return self.dataframes
        
        except FileNotFoundError:
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    def change_zone(self, unizone_col: str, weight_col: str, 
                   reg_nm: Optional[List[str]] = None) -> Dict[int, pd.DataFrame]:
        """
        KED 데이터의 zone 변환 (오버라이드)
        
        Args:
            unizone_col: unizone 컬럼명
            weight_col: 가중치 컬럼명
            reg_nm: 필터링할 지역명 리스트 (시도 단위)
            
        Returns:
            변환된 데이터 딕셔너리
        """
        if self.zone_df is None:
            raise ValueError("zonecode 데이터가 로드되지 않았습니다.")
        
        # Zonecode 준비
        zone_tmp = self.zone_df.copy()
        zone_tmp = zone_tmp.assign(
            adm_nm=zone_tmp['sd_nm2'] + " " + zone_tmp['sgg_nm2']
        )
        zonecode = zone_tmp[[unizone_col, 'adm_nm']].drop_duplicates().reset_index(drop=True)
        
        converted_data = {}
        
        for year, df in self.dataframes.items():
            try:
                # Source 변환
                df_merged_o = pd.merge(df, zonecode, how='left', 
                                     left_on='ori_adm_nm', right_on='adm_nm')
                df_merged_o = df_merged_o.rename(columns={unizone_col: 'source'}).drop(columns=['adm_nm'])
                
                # Target 변환
                df_merged_od = pd.merge(df_merged_o, zonecode, how='left', 
                                      left_on='des_adm_nm', right_on='adm_nm')
                df_merged_od = df_merged_od.rename(columns={unizone_col: 'target'}).drop(columns=['adm_nm'])
                
                # 지역 필터링
                if reg_nm is not None:
                    df_merged_od['o_sd'] = df_merged_od['ori_adm_nm'].str[:2]
                    df_merged_od['d_sd'] = df_merged_od['des_adm_nm'].str[:2]
                    
                    df_merged_od = df_merged_od[
                        (df_merged_od['o_sd'].isin(reg_nm)) & 
                        (df_merged_od['d_sd'].isin(reg_nm))
                    ]
                else:
                    logger.info(f"✅ {year}년: 지역 범위가 지정되지 않았으므로 모든 지역의 데이터를 사용합니다")
                
                # 집계
                grouped = df_merged_od.groupby(['source', 'target'], dropna=False)[weight_col].sum().reset_index()
                grouped = grouped.rename(columns={weight_col: 'weights'})
                
                # Type 변환 및 정리
                grouped['source'] = grouped['source'].astype(int)
                grouped['target'] = grouped['target'].astype(int)
                
                # 결측값 제거
                grouped_cleaned = grouped.dropna(subset=['source', 'target'])
                missing_rows = len(grouped) - len(grouped_cleaned)
                
                if missing_rows > 0:
                    logger.warning(f'{year}년 데이터에서 총 {missing_rows}개의 결측행이 제거되었습니다.')
                
                converted_data[year] = grouped_cleaned[['source', 'target', 'weights']]
                
            except Exception as e:
                logger.error(f"❌ {year}년 KED zone 변환 실패: {e}")
                continue
        
        self.converted_data = converted_data
        return converted_data
    
    def get_industry_code(self, industry: str) -> List[str]:
        """
        산업 분류에 따른 산업 코드 반환
        
        Args:
            industry: '전산업', '제조업', '서비스업' 중 하나
            
        Returns:
            해당 산업의 코드 리스트
        """
        industry_codes = {
            "전산업": [chr(ord('A') + i) for i in range(20)],
            "제조업": ["C"],
            "서비스업": ["J", "K", "L", "M", "N", "O", "P", "Q", "R", "S"]
        }
        
        if industry not in industry_codes:
            raise ValueError("지원되는 산업: '전산업', '제조업', '서비스업'")
        
        return industry_codes[industry]


class SIMS(BaseDataProcessor):
    """
    SIMS 네트워크(서울대학교 활용 중기부 기업거래 데이터) 데이터 처리 클래스

    - 데이터 구조: year, seller, buyer, tx_count, tx_sum_count, tx_sum_amount가 포함된 csv 파일
    - 지역코드: 행안부 시군구 코드 적용 (zonecode의 sgg_cd_adm과 동일)
    """

    def read_data_file(self, year: int, path: str = '.') -> pd.DataFrame:
        """
        BaseDataProcessor 인터페이스 준수를 위한 메서드.
        SIMS는 단일 CSV에서 여러 연도를 한 번에 로드하므로, 이 메서드는 사용하지 않습니다.
        """
        raise NotImplementedError("SIMS는 load_sims_data(years, file_path)를 사용하세요.")

    def load_sims_data(self, years: List[int], path: str) -> Dict[int, pd.DataFrame]:
        """
        SIMS 네트워크 데이터 로드

        첫 번째 행을 건너뛰고(header=None, skiprows=1),
        열이 정확히 6개인지 검사 후 이름 부여.

        Args:
            years: 로드할 연도 리스트
            path: CSV 파일 경로

        Returns:
            연도별 원본 데이터 딕셔너리
        """
        try:
            # 첫 번째 행 패스
            data = pd.read_csv(path, header=None, skiprows=1, encoding='utf-8')
            logger.info(f"✅ SIMS 데이터 로드 성공: shape={data.shape}")

            expected_cols = ['year', 'seller', 'buyer', 'tx_count', 'tx_sum_count', 'tx_sum_amount']

            if data.shape[1] != len(expected_cols):
                raise ValueError(
                    f"열 개수 불일치: 예상 {len(expected_cols)}개, 실제 {data.shape[1]}개"
                )

            # 컬럼 이름 부여
            data.columns = expected_cols

            for selected_year in years:
                year_df = data[data['year'] == selected_year].copy()
                if year_df.empty:
                    logger.warning(f"⚠️ {selected_year}년에 해당하는 데이터가 없습니다.")
                self.dataframes[selected_year] = year_df
                logger.info(f"{selected_year}년 데이터 저장 완료: {len(year_df)}개 관측치")

            return self.dataframes

        except FileNotFoundError:
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    def change_zone(self, unizone_col: str, weight_col: str,
                    reg_nm: Optional[List[str]] = None) -> Dict[int, pd.DataFrame]:
        """
        SIMS 데이터의 zone 변환 (seller/buyer → unizone)

        Args:
            unizone_col: zonecode에서 사용할 unizone 컬럼명 (예: unizone_1)
            weight_col: 합산할 가중치 컬럼명 (예: 'tx_count', 'tx_sum_count', 'tx_sum_amount' 등)
            reg_nm: 필터링할 시도명 리스트 (zonecode.sd_nm2 기준)

        Returns:
            {year: pd.DataFrame[['source','target','weights']]}
        """
        if self.zone_df is None:
            raise ValueError("zonecode 데이터가 로드되지 않았습니다. BaseDataProcessor(zonecode_path=...)로 초기화하세요.")
        if not self.dataframes:
            raise ValueError("데이터가 로드되지 않았습니다. load_sims_data(...)를 먼저 호출하세요.")

        if 'sgg_cd_adm' not in self.zone_df.columns:
            raise ValueError("zonecode에 'sgg_cd_adm' 컬럼이 필요합니다.")
        if unizone_col not in self.zone_df.columns:
            raise ValueError(f"zonecode에 '{unizone_col}' 컬럼이 필요합니다.")

        zone_map = self.zone_df[['sgg_cd_adm', 'sd_nm2', unizone_col]].drop_duplicates()
        converted_data: Dict[int, pd.DataFrame] = {}

        for year, df in self.dataframes.items():
            try:
                if df.empty:
                    logger.warning(f"⚠️ {year}년: 데이터가 비어 있습니다. 건너뜁니다.")
                    continue

                # 타입 정리
                df['seller'] = df['seller'].astype(str)
                df['buyer'] = df['buyer'].astype(str)
                zone_map_local = zone_map.copy()
                zone_map_local['sgg_cd_adm'] = zone_map_local['sgg_cd_adm'].astype(str)

                # seller → source
                merged_o = df.merge(
                    zone_map_local[['sgg_cd_adm', 'sd_nm2', unizone_col]],
                    how='left', left_on='seller', right_on='sgg_cd_adm'
                ).rename(columns={unizone_col: 'source', 'sd_nm2': 'o_sd'}).drop(columns=['sgg_cd_adm'])

                # buyer → target
                merged_od = merged_o.merge(
                    zone_map_local[['sgg_cd_adm', 'sd_nm2', unizone_col]],
                    how='left', left_on='buyer', right_on='sgg_cd_adm'
                ).rename(columns={unizone_col: 'target', 'sd_nm2': 'd_sd'}).drop(columns=['sgg_cd_adm'])

                # 지역 필터링
                if reg_nm is not None:
                    merged_od = merged_od[
                        (merged_od['o_sd'].isin(reg_nm)) &
                        (merged_od['d_sd'].isin(reg_nm))
                    ]
                else:
                    logger.info(f"✅ {year}년: 지역 범위 미지정 — 전체 지역 사용")

                # 집계
                grouped = (
                    merged_od
                    .groupby(['source', 'target'], dropna=False)[weight_col]
                    .sum()
                    .reset_index()
                    .rename(columns={weight_col: 'weights'})
                )

                # 타입/결측 정리
                grouped['source'] = pd.to_numeric(grouped['source'], errors='coerce').dropna().astype(int)
                grouped['target'] = pd.to_numeric(grouped['target'], errors='coerce').dropna().astype(int)

                before = len(grouped)
                grouped = grouped.dropna(subset=['source', 'target'])
                dropped = before - len(grouped)
                if dropped > 0:
                    logger.warning(f"{year}년: 존 매핑 실패로 {dropped}개 행 제거")

                converted = grouped[['source', 'target', 'weights']].copy()
                converted_data[year] = converted

            except Exception as e:
                logger.error(f"❌ {year}년 SIMS zone 변환 실패: {e}")
                continue

        self.converted_data = converted_data
        return converted_data