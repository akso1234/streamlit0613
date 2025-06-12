# data_loader.py

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import streamlit as st

# 서울 25개 자치구 리스트 (전역 상수)
SEOUL_DISTRICTS = [
    "종로구","중구","용산구","성동구","광진구","동대문구","중랑구","성북구",
    "강북구","도봉구","노원구","은평구","서대문구","마포구","양천구","강서구",
    "구로구","금천구","영등포구","동작구","관악구","서초구","강남구","송파구","강동구"
]

@st.cache_data
def load_raw_data(
    year: int,
    geojson_path: str = "data/seoul.geojson"
):
    """
    ① 연도(year)에 맞춰 "{year}seoul_hospital.csv" 파일을 읽어,
       df_hosp (병원수), df_beds (병상수) DataFrame 생성
    ② geojson_path로부터 GeoDataFrame을 읽어 'gu'별 dissolve
    → df_hosp, df_beds, gdf_gu 반환
    """
    csv_path = f"data/{year}seoul_hospital.csv"
    if not os.path.isfile(csv_path):
        st.error(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
        return None, None, None

    # --- CSV 읽기 및 df_hosp, df_beds 생성 ---
    df_raw = pd.read_csv(csv_path, header=[2, 3], encoding="utf-8")

    # (1) df_hosp: '병원수' 레벨만 슬라이싱
    df_hosp = df_raw.xs("병원수", axis=1, level=1).copy()
    df_hosp["gu"] = df_raw[("자치구별(2)", "자치구별(2)")].str.replace(" ", "", regex=False)
    df_hosp = df_hosp[df_hosp["gu"] != "서울시"]  # "서울시" 전체 합계 제거

    # (2) df_beds: '병상수' 레벨만 슬라이싱
    df_beds = df_raw.xs("병상수", axis=1, level=1).copy()
    df_beds["gu"] = df_hosp["gu"]
    df_beds = df_beds[df_beds["gu"] != "서울시"]

    # --- GeoJSON 읽기 및 구별 dissolve ---
    if not os.path.isfile(geojson_path):
        st.error(f"GeoJSON 파일을 찾을 수 없습니다: {geojson_path}")
        return df_hosp, df_beds, None

    gdf = gpd.read_file(geojson_path)
    gdf["gu"] = gdf["sggnm"].str.replace(" ", "", regex=False)
    gdf_gu = gdf.dissolve(by="gu", as_index=False)

    return df_hosp, df_beds, gdf_gu


@st.cache_data
def load_nursing_sheet0(file_path: str, districts: list[str]) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=0, header=[1,2])
    df = df[df.iloc[:, 0].notna()]
    df.columns = [
        ' '.join([str(c).strip() for c in col if str(c).strip() and 'Unnamed' not in str(c)])
        for col in df.columns
    ]
    df = df.rename(columns={df.columns[0]: 'region'})
    df_gu = df[df['region'].str.endswith('구')].copy()

    # 2) 메트릭 생성
    df_metrics = pd.DataFrame({
        'region':    df_gu['region'],
        'facility':  df_gu.iloc[:, 2].astype(int),
        'capacity':  df_gu.iloc[:, 3].astype(int),
        'occupancy': df_gu.iloc[:, 4].astype(int),
        'staff':     df_gu.iloc[:, 5].astype(int),
    }).set_index('region')
    df_metrics['additional']    = df_metrics['capacity'] - df_metrics['occupancy']
    df_metrics['cap_per_staff'] = df_metrics['capacity'] / df_metrics['staff']
    df_metrics['occ_per_staff'] = df_metrics['occupancy'] / df_metrics['staff']

    # 중복 인덱스 제거
    df_metrics = df_metrics[~df_metrics.index.duplicated(keep='first')]

    # districts 순서대로 정렬
    df_filt = df_metrics.reindex(districts)

    return df_filt


@st.cache_data
def load_nursing_sheet1(file_path: str, districts: list[str]) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=1, header=[1,2])
    df = df[df.iloc[:, 0].notna()]
    df.columns = [
        ' '.join([str(c).strip() for c in col if str(c).strip() and 'Unnamed' not in str(c)])
        for col in df.columns
    ]
    df = df.rename(columns={df.columns[0]: 'region'})
    df_gu = df[df['region'].str.endswith('구')]

    # 중복 인덱스 제거
    df_gu = df_gu[~df_gu['region'].duplicated(keep='first')]

    # 2) 메트릭 생성
    df_metrics = pd.DataFrame({
        'region':    df_gu['region'],
        'facility':  df_gu.iloc[:, 2].astype(int),
        'capacity':  df_gu.iloc[:, 3].astype(int),
        'occupancy': df_gu.iloc[:, 4].astype(int),
        'staff':     df_gu.iloc[:, 5].astype(int),
    }).set_index('region')
    df_metrics['additional']    = df_metrics['capacity'] - df_metrics['occupancy']
    df_metrics['cap_per_staff'] = df_metrics['capacity'] / df_metrics['staff']
    df_metrics['occ_per_staff'] = df_metrics['occupancy'] / df_metrics['staff']

    # 중복 인덱스 제거
    df_metrics = df_metrics[~df_metrics.index.duplicated(keep='first')]

    # districts 순서대로 정렬
    df_metrics = df_metrics.reindex(districts)

    return df_metrics


@st.cache_data
def load_nursing_csv_general(file_path: str, districts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    2020~2023년 노인여가복지시설 CSV 파싱
      1) skiprows=2, header=[0,1] → 멀티헤더 읽기
      2) region이 있는 행만 필터링
      3) 헤더 평탄화(공백, 개행 문자 제거)
      4) 불필요 컬럼(‘65세이상노인인구’, ‘합계’, ‘(단위:개소,명)’) 삭제
      5) 숫자형 컬럼(노인복지관시설수, 종사자수, 경로당, 노인교실) int 변환
      6) region 공백 제거, 중복 삭제, reindex(districts).fillna(0).astype(int)
      7) df_welf, df_centers 반환
    """
    if not os.path.isfile(file_path):
        empty_welf    = pd.DataFrame(0, index=districts, columns=["facility", "staff"])
        empty_centers = pd.DataFrame(0, index=districts, columns=["facility"])
        return empty_welf, empty_centers

    # 1) CSV 읽기
    try:
        df = pd.read_csv(
            file_path,
            skiprows=2,
            header=[0, 1],
            encoding="utf-8"
        )
    except Exception as e:
        st.error(f"CSV를 읽는 중 오류 발생 ({file_path}): {e}")
        empty_welf    = pd.DataFrame(0, index=districts, columns=["facility", "staff"])
        empty_centers = pd.DataFrame(0, index=districts, columns=["facility"])
        return empty_welf, empty_centers

    # 2) region이 비어 있지 않은 행만 남기기
    df = df[df.iloc[:, 0].notna()]

    # 3) 헤더 평탄화 (공백 및 개행 문자 제거)
    flat_cols = []
    for lvl0, lvl1 in df.columns:
        pieces = []
        if pd.notna(lvl0) and "Unnamed" not in str(lvl0):
            pieces.append(str(lvl0).strip())
        if pd.notna(lvl1) and "Unnamed" not in str(lvl1):
            pieces.append(str(lvl1).strip())
        name = "".join(pieces).replace(" ", "").replace("\n", "")
        flat_cols.append(name)
    flat_cols[0] = "region"
    df.columns = flat_cols

    # 4) 불필요 컬럼 삭제 (부분 매칭)
    drop_keys = ["65세이상노인인구", "합계", "(단위:개소,명)"]
    for key in drop_keys:
        df = df.loc[:, ~df.columns.str.contains(key)]

    # 5) 숫자형 컬럼 변환 도움 함수
    def to_int(col_name: str) -> pd.Series:
        """컬럼명이 존재하면 쉼표 제거 후 int로 변환, 없으면 0으로 채움"""
        if col_name not in df.columns:
            return pd.Series(0, index=df.index, dtype=int)
        s = (
            df[col_name].astype(str)
             .str.replace(",", "", regex=False)
             .str.strip()
             .replace({"": np.nan, "-": np.nan})
        )
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

    # 5-1) 필요한 컬럼 매핑
    cols_needed = {
        "노인복지관시설수": "welf_facility",
        "종사자수":        "welf_staff",
        "경로당":          "keiro",
        "노인교실":        "class_"
    }
    # 5-2) 실제 컬럼이름 → to_int → 새 시리즈로 df에 추가
    for old_col, new_col in cols_needed.items():
        df[new_col] = to_int(old_col)

    # 6) region 공백 제거 → 인덱스 설정 → 중복 제거 → districts 순서로 재정렬
    df["region"] = df["region"].astype(str).str.replace(" ", "", regex=False)
    df_indexed = df.drop_duplicates(subset="region", keep="first").set_index("region")
    df_indexed = df_indexed.reindex(districts, fill_value=0)

    # 7) 정수형으로 변환 (fillna 이후)
    df_indexed = df_indexed.astype(int)

    # 8) 최종 df_welf, df_centers 생성
    df_welf = pd.DataFrame(index=df_indexed.index)
    df_welf["facility"] = df_indexed["welf_facility"]
    df_welf["staff"]    = df_indexed["welf_staff"]

    df_centers = pd.DataFrame(index=df_indexed.index)
    df_centers["facility"] = df_indexed["keiro"] + df_indexed["class_"]
    print(df_welf.head(10))
    print(df_centers.head)

    return df_welf, df_centers


@st.cache_data
def load_nursing_csv_2019(file_path: str, districts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    2019년 노인여가복지시설 CSV 파싱
      1) skiprows=4, header=None → names 명시
      2) 숫자형 컬럼(쉼표 제거→pd.to_numeric→fillna(0)→astype(int))
      3) region 공백 제거 → 인덱스 설정 → 중복 제거 → reindex(districts).fillna(0)
      4) df_welf, df_centers 반환
    """
    if not os.path.isfile(file_path):
        empty_welf    = pd.DataFrame(0, index=districts, columns=['facility','staff'])
        empty_centers = pd.DataFrame(0, index=districts, columns=['facility'])
        return empty_welf, empty_centers

    try:
        df_tmp = pd.read_csv(
            file_path,
            skiprows=4,
            header=None,
            encoding="utf-8",
            names=[
                "region",
                "pop65",          # 사용 안 함
                "total",          # 사용 안 함
                "welf_facility",  # 노인복지관 시설 수
                "welf_staff",     # 노인복지관 종사자 수
                "keiro",          # 경로당 수
                "class"           # 노인교실 수
            ]
        )
    except Exception as e:
        st.error(f"2019년 CSV를 읽는 중 오류 발생 ({file_path}): {e}")
        empty_welf    = pd.DataFrame(0, index=districts, columns=['facility','staff'])
        empty_centers = pd.DataFrame(0, index=districts, columns=['facility'])
        return empty_welf, empty_centers

    # 1) 숫자형 컬럼 처리: 쉼표 제거 → to_numeric → fillna(0) → int
    for col in ["welf_facility", "welf_staff", "keiro", "class"]:
        df_tmp[col] = (
            df_tmp[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace("", "0")
        )
        df_tmp[col] = pd.to_numeric(df_tmp[col], errors="coerce").fillna(0).astype(int)

    # 2) region 공백 제거
    df_tmp["region"] = df_tmp["region"].astype(str).str.replace(" ", "", regex=False)

    # 3) 중복 region 제거
    df_indexed = df_tmp.set_index("region")
    df_indexed = df_indexed[~df_indexed.index.duplicated(keep="first")]

    # 4) districts 순서대로 재정렬, 누락 구는 0으로 채움
    df_filtered = df_indexed.reindex(districts, fill_value=0)

    # 5) 최종 df_welf, df_centers 생성
    df_welf = pd.DataFrame(index=df_filtered.index)
    df_welf["facility"] = df_filtered["welf_facility"]
    df_welf["staff"]    = df_filtered["welf_staff"]

    df_centers = pd.DataFrame(index=df_filtered.index)
    df_centers["facility"] = df_filtered["keiro"] + df_filtered["class"]

    return df_welf, df_centers


@st.cache_data
def load_nursing_csv(file_path: str, districts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    래퍼 함수:
      • 파일명이 '2019'로 시작하면 load_nursing_csv_2019 호출
      • 그 외 연도(2020~2023)면 load_nursing_csv_general 호출
    """
    basename = os.path.basename(file_path)
    if basename.startswith("2019"):
        return load_nursing_csv_2019(file_path, districts)
    else:
        return load_nursing_csv_general(file_path, districts)


@st.cache_data
def load_nursing_sheet3(file_path: str, districts: list[str]) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=3, header=[1,2])
    df = df[df.iloc[:, 0].notna()]
    df.columns = [
        ' '.join([str(c).strip() for c in col if str(c).strip() and 'Unnamed' not in str(c)])
        for col in df.columns
    ]
    df = df.rename(columns={df.columns[0]: 'region'})
    df_gu = df[df['region'].str.endswith('구')].copy()

    # 2) 메트릭 생성
    df_metrics = pd.DataFrame({
        'region':    df_gu['region'],
        'facility':  df_gu.iloc[:, 2].astype(int),
        'capacity':  df_gu.iloc[:, 3].astype(int),
        'occupancy': df_gu.iloc[:, 4].astype(int),
        'staff':     df_gu.iloc[:, 5].astype(int),
    }).set_index('region')
    df_metrics['additional']    = df_metrics['capacity'] - df_metrics['occupancy']
    df_metrics['cap_per_staff'] = df_metrics['capacity'] / df_metrics['staff']
    df_metrics['occ_per_staff'] = df_metrics['occupancy'] / df_metrics['staff']

    # 중복 인덱스 제거
    df_metrics = df_metrics[~df_metrics.index.duplicated(keep='first')]

    # 재정렬
    df_filt = df_metrics.reindex(districts)

    return df_filt


@st.cache_data
def load_nursing_sheet4(file_path: str, districts: list[str]) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=4, header=[1,2])
    df = df[df.iloc[:, 0].notna()]
    df.columns = [
        ' '.join([str(c).strip() for c in col if str(c).strip() and 'Unnamed' not in str(c)])
        for col in df.columns
    ]
    df = df.rename(columns={df.columns[0]: 'region'})
    df_gu = df[df['region'].str.endswith('구')].copy()

    # 2) 메트릭 생성
    df_metrics = pd.DataFrame({
        'region': df_gu['region'],
        'facility': df_gu.iloc[:, 3].astype(int),
        'staff':    df_gu.iloc[:, 4].astype(int),
    }).set_index('region')

    # 중복 인덱스 제거
    df_metrics = df_metrics[~df_metrics.index.duplicated(keep='first')]

    # 재정렬
    df_metrics = df_metrics.reindex(districts)

    return df_metrics


@st.cache_data
def load_nursing_sheet5(file_path: str, districts: list[str]) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=6, header=[1,2])
    df = df[df.iloc[:, 0].notna()]
    df.columns = [
        ' '.join([str(c).strip() for c in col if str(c).strip() and 'Unnamed' not in str(c)])
        for col in df.columns
    ]
    df = df.rename(columns={df.columns[0]: 'region'})
    df_gu = df[df['region'].str.endswith('구')].copy()

    # 중복 인덱스 제거
    df_gu = df_gu[~df_gu['region'].duplicated(keep='first')]

    # 2) 메트릭 생성
    df_metrics = pd.DataFrame({
        'region':    df_gu['region'],
        'facility':  df_gu.iloc[:, 2].astype(int),
        'capacity':  df_gu.iloc[:, 3].astype(int),
        'occupancy': df_gu.iloc[:, 4].astype(int),
        'staff':     df_gu.iloc[:, 5].astype(int),
    }).set_index('region')
    df_metrics['additional']    = df_metrics['capacity'] - df_metrics['occupancy']
    df_metrics['cap_per_staff'] = df_metrics['capacity'] / df_metrics['staff']
    df_metrics['occ_per_staff'] = df_metrics['occupancy'] / df_metrics['staff']

    # 중복 인덱스 제거
    df_metrics = df_metrics[~df_metrics.index.duplicated(keep='first')]

    # 재정렬
    df_metrics = df_metrics.reindex(districts)

    return df_metrics
