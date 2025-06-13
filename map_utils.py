import pandas as pd
import folium
from folium.features import DivIcon # folium.features에서 DivIcon 임포트
import streamlit as st
import geopandas as gpd # geopandas 임포트 추가

@st.cache_data
def make_merged_counts(df_hosp: pd.DataFrame, _gdf_gu: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    df_hosp: '병원수' 정보가 담긴 DataFrame (컬럼: gu, 소계, 종합병원, 병원, 의원, 요양병원)
    _gdf_gu: GeoDataFrame (구별 경계)
    → 구별 의료기관 수(소계, 종합병원, 병원, 의원, 요양병원) GeoDataFrame 반환
    """
    # 입력 데이터 유효성 검사
    if df_hosp is None or df_hosp.empty:
        st.warning("make_merged_counts: df_hosp 데이터가 없습니다.")
        return gpd.GeoDataFrame()
    if _gdf_gu is None or _gdf_gu.empty:
        st.warning("make_merged_counts: _gdf_gu 데이터가 없습니다.")
        return gpd.GeoDataFrame()

    cols = ["gu", "소계", "종합병원", "병원", "의원", "요양병원"]
    
    # 필요한 컬럼 존재 여부 확인
    missing_cols_df_hosp = [col for col in cols if col not in df_hosp.columns]
    if missing_cols_df_hosp:
        st.error(f"make_merged_counts: df_hosp에 필요한 컬럼 {missing_cols_df_hosp}이(가) 없습니다.")
        return gpd.GeoDataFrame()
    
    if "gu" not in _gdf_gu.columns: # _gdf_gu에 'gu' 컬럼 확인
        st.error("make_merged_counts: _gdf_gu에 'gu' 컬럼이 없습니다.")
        return gpd.GeoDataFrame()

    df_counts = df_hosp[cols].copy()
    for c in cols[1:]: # 'gu'는 문자열이므로 제외
        df_counts[c] = pd.to_numeric(df_counts[c], errors="coerce").fillna(0).astype(int)

    try:
        merged = _gdf_gu.merge(df_counts, on="gu", how="left")
    except Exception as e:
        st.error(f"make_merged_counts: GeoDataFrame 병합 중 오류 발생: {e}")
        return gpd.GeoDataFrame()
        
    for c in cols[1:]: # 병합 후 NaN 값을 0으로 채움
        if c in merged.columns: # 해당 컬럼이 merged에 있는지 확인
            merged[c] = merged[c].fillna(0).astype(int)
    return merged


@st.cache_data
def make_merged_avg_beds(df_hosp: pd.DataFrame, df_beds: pd.DataFrame, _gdf_gu: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    df_hosp: '병원수' 정보
    df_beds: '병상수' 정보
    _gdf_gu: GeoDataFrame (구별 경계)
    → 구별 평균 병상 수(종합병원, 병원, 의원, 요양병원) GeoDataFrame 반환
    """
    if df_hosp is None or df_hosp.empty:
        st.warning("make_merged_avg_beds: df_hosp 데이터가 없습니다.")
        return gpd.GeoDataFrame()
    if df_beds is None or df_beds.empty:
        st.warning("make_merged_avg_beds: df_beds 데이터가 없습니다.")
        return gpd.GeoDataFrame()
    if _gdf_gu is None or _gdf_gu.empty:
        st.warning("make_merged_avg_beds: _gdf_gu 데이터가 없습니다.")
        return gpd.GeoDataFrame()

    types = ["종합병원", "병원", "의원", "요양병원"]
    records = []

    # 필요한 컬럼 확인
    required_cols_hosp = ["gu"] + types
    required_cols_beds = ["gu"] + types
    if not all(col in df_hosp.columns for col in required_cols_hosp):
        st.error(f"make_merged_avg_beds: df_hosp에 필요한 컬럼이 부족합니다. ({required_cols_hosp})")
        return gpd.GeoDataFrame()
    if not all(col in df_beds.columns for col in required_cols_beds):
        st.error(f"make_merged_avg_beds: df_beds에 필요한 컬럼이 부족합니다. ({required_cols_beds})")
        return gpd.GeoDataFrame()
    if "gu" not in _gdf_gu.columns:
        st.error("make_merged_avg_beds: _gdf_gu에 'gu' 컬럼이 없습니다.")
        return gpd.GeoDataFrame()

    # df_hosp와 df_beds의 인덱스가 동일하고 순서가 맞다고 가정하지 않고 'gu' 기준으로 처리
    df_hosp_indexed = df_hosp.set_index('gu')
    df_beds_indexed = df_beds.set_index('gu')
    
    common_gus = df_hosp_indexed.index.intersection(df_beds_indexed.index)

    for gu_val in common_gus: # df_hosp의 'gu'를 기준으로 순회
        # row_hosp = df_hosp_indexed.loc[gu_val] # 이미 위에서 처리
        # if gu_val not in df_beds_indexed.index: # df_beds에 해당 gu가 없으면 건너뜀
        #     continue
        # row_beds = df_beds_indexed.loc[gu_val]

        for t in types:
            hosp_cnt = pd.to_numeric(df_hosp_indexed.loc[gu_val, t], errors="coerce")
            bed_cnt = pd.to_numeric(df_beds_indexed.loc[gu_val, t], errors="coerce")
            avg = 0.0 # 기본값
            if pd.notna(hosp_cnt) and hosp_cnt > 0 and pd.notna(bed_cnt): # hosp_cnt가 0이 아니고 NaN이 아니어야 함
                avg = bed_cnt / hosp_cnt
            records.append({"gu": gu_val, "type": t, "avg_beds": avg})
    
    if not records:
        st.warning("make_merged_avg_beds: 평균 병상수 계산을 위한 레코드가 생성되지 않았습니다.")
        return gpd.GeoDataFrame()

    stats = pd.DataFrame(records)
    try:
        pivot = stats.pivot(index="gu", columns="type", values="avg_beds").reset_index().fillna(0)
    except Exception as e:
        st.error(f"make_merged_avg_beds: pivot 테이블 생성 중 오류: {e}")
        return gpd.GeoDataFrame()

    try:
        merged = _gdf_gu.merge(pivot, on="gu", how="left")
    except Exception as e:
        st.error(f"make_merged_avg_beds: GeoDataFrame 병합 중 오류: {e}")
        return gpd.GeoDataFrame()
        
    for t in types: # pivot 테이블 생성 후 생긴 컬럼들
        if t in merged.columns:
            merged[t] = merged[t].fillna(0) # NaN 값 0으로 채우기
    return merged


def draw_hospital_count_choropleth(merged_counts: gpd.GeoDataFrame, width=800, height=600):
    if merged_counts is None or merged_counts.empty or 'geometry' not in merged_counts.columns:
        st.info("의료기관 수 지도 데이터를 생성할 수 없습니다.")
        return folium.Map(location=[37.55, 126.98], zoom_start=11) # 빈 지도 반환

    m = folium.Map(
        location=[37.55, 126.98],
        zoom_start=11,
        tiles="CartoDB positron",
        attr="© CartoDB",
    )

    types_counts = ["소계", "종합병원", "병원", "의원", "요양병원"]
    colors_counts = ["YlOrRd", "PuBu", "Greens", "Purples", "OrRd"]
    
    missing_cols = [col for col in types_counts if col not in merged_counts.columns]
    if missing_cols:
        st.warning(f"draw_hospital_count_choropleth: merged_counts에 다음 컬럼이 없습니다: {missing_cols}. 해당 레이어는 생략됩니다.")

    for inst, cmap in zip(types_counts, colors_counts):
        if inst not in merged_counts.columns:
            continue
        folium.Choropleth(
            geo_data=merged_counts, # geopandas df 직접 사용
            name=f"{inst} 수",
            data=merged_counts,
            columns=["gu", inst],
            key_on="feature.properties.gu",
            fill_color=cmap,
            fill_opacity=0.6,
            line_opacity=0.4,
            legend_name=f"{inst} 수",
            highlight=True # 사용자가 제공한 코드에는 없었으나, 일반적으로 유용하여 추가 (제거 가능)
        ).add_to(m)

    # 구 이름 레이블 추가
    # geometry가 유효하고, centroid 계산이 가능한 경우에만 실행
    for _, row in merged_counts.iterrows():
        if row.geometry and hasattr(row.geometry, 'centroid') and row.geometry.centroid:
            try:
                # GeoPandas 0.7.0 이상에서는 centroid가 Point 객체를 반환하므로 .x, .y로 접근
                lon, lat = row.geometry.centroid.x, row.geometry.centroid.y
                folium.Marker( # folium.map.Marker 대신 folium.Marker 사용이 일반적
                    location=[lat, lon],
                    icon=DivIcon(
                        icon_size=(150, 36),
                        icon_anchor=(0, 0), # 아이콘 기준점
                        html=f"<div style='font-size:10px; font-weight:bold; color:black;'>{row['gu']}</div>",
                    ),
                ).add_to(m)
            except Exception as e: # centroid 계산 실패 등 예외 처리
                # st.warning(f"마커 생성 중 오류 ({row.get('gu', '알수없는 구')}) : {e}")
                pass # 개별 마커 오류는 전체 지도 생성을 막지 않음

    # Stamen 타일은 FileNotFoundError 발생 가능성이 있으므로 주석 처리 유지
    #folium.TileLayer(tiles="Stamen Toner", name="Toner", attr="Map tiles by Stamen Design, CC BY 3.0").add_to(m)
    #folium.TileLayer(tiles="Stamen Terrain", name="Terrain", attr="Map tiles by Stamen Design, CC BY 3.0").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def draw_avg_beds_choropleth(merged_avg: gpd.GeoDataFrame, width=800, height=600):
    if merged_avg is None or merged_avg.empty or 'geometry' not in merged_avg.columns:
        st.info("평균 병상 수 지도 데이터를 생성할 수 없습니다.")
        return folium.Map(location=[37.55, 126.98], zoom_start=11) # 빈 지도 반환
        
    m = folium.Map(
        location=[37.55, 126.98],
        zoom_start=11,
        tiles="CartoDB positron",
        attr="© CartoDB",
    )

    types_avg = ["종합병원", "병원", "의원", "요양병원"]
    colors_avg = ["YlOrRd", "PuBu", "Greens", "PuBuGn"]

    missing_cols_avg = [col for col in types_avg if col not in merged_avg.columns]
    if missing_cols_avg:
        st.warning(f"draw_avg_beds_choropleth: merged_avg에 다음 컬럼이 없습니다: {missing_cols_avg}. 해당 레이어는 생략됩니다.")

    for t, cmap in zip(types_avg, colors_avg):
        if t not in merged_avg.columns:
            continue
        folium.Choropleth(
            geo_data=merged_avg, # geopandas df 직접 사용
            name=f"{t} 평균 병상수", # 원본 legend_name과 통일 (또는 "평균 병상/병원")
            data=merged_avg,
            columns=["gu", t],
            key_on="feature.properties.gu",
            fill_color=cmap,
            fill_opacity=0.7,
            line_opacity=0.2, # 원본 값 유지
            legend_name=f"{t} 평균 병상 수", # 원본 값 유지
            highlight=True
        ).add_to(m)

    for _, row in merged_avg.iterrows():
        if row.geometry and hasattr(row.geometry, 'centroid') and row.geometry.centroid:
            try:
                lon, lat = row.geometry.centroid.x, row.geometry.centroid.y
                folium.Marker(
                    location=[lat, lon],
                    icon=DivIcon(
                        icon_size=(150, 36),
                        icon_anchor=(0, 0),
                        html=f"<div style='font-size:10px; font-weight:bold; color:black;'>{row['gu']}</div>",
                    ),
                ).add_to(m)
            except Exception as e:
                # st.warning(f"마커 생성 중 오류 ({row.get('gu', '알수없는 구')}, 평균 병상 수) : {e}")
                pass

    # Stamen 타일 주석 유지
    #folium.TileLayer(tiles="Stamen Toner", name="Toner", attr="Map tiles by Stamen Design, CC BY 3.0").add_to(m)
    #folium.TileLayer(tiles="Stamen Terrain", name="Terrain", attr="Map tiles by Stamen Design, CC BY 3.0").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m
