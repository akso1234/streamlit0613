import pandas as pd
import folium
from folium.features import DivIcon # DivIcon 직접 임포트 확인
import streamlit as st
import geopandas as gpd # geopandas 임포트 추가 (GeoDataFrame 타입 힌트 사용 시 필요)

@st.cache_data
def make_merged_counts(df_hosp: pd.DataFrame, _gdf_gu: gpd.GeoDataFrame) -> gpd.GeoDataFrame: # 반환 타입 명시
    """
    df_hosp: '병원수' 정보가 담긴 DataFrame (컬럼: gu, 소계, 종합병원, 병원, 의원, 요양병원)
    _gdf_gu: GeoDataFrame (구별 경계)
    → 구별 의료기관 수(소계, 종합병원, 병원, 의원, 요양병원) GeoDataFrame 반환
    """
    if df_hosp is None or df_hosp.empty or _gdf_gu is None or _gdf_gu.empty:
        st.warning("make_merged_counts: 입력 데이터가 유효하지 않습니다.")
        return gpd.GeoDataFrame() # 빈 GeoDataFrame 반환

    cols = ["gu", "소계", "종합병원", "병원", "의원", "요양병원"]
    
    # df_hosp에 필요한 모든 컬럼이 있는지 확인
    missing_cols_df_hosp = [col for col in cols if col not in df_hosp.columns]
    if missing_cols_df_hosp:
        st.error(f"make_merged_counts: df_hosp에 필요한 컬럼 {missing_cols_df_hosp}이(가) 없습니다.")
        return gpd.GeoDataFrame()
    
    if "gu" not in _gdf_gu.columns:
        st.error("make_merged_counts: _gdf_gu에 'gu' 컬럼이 없습니다.")
        return gpd.GeoDataFrame()
        
    df_counts = df_hosp[cols].copy()
    for c in cols[1:]: # 'gu' 제외
        df_counts[c] = pd.to_numeric(df_counts[c], errors="coerce").fillna(0).astype(int)

    try:
        # 'gu' 컬럼 기준으로 병합, _gdf_gu의 geometry 정보 유지
        merged = _gdf_gu.merge(df_counts, on="gu", how="left")
    except Exception as e:
        st.error(f"make_merged_counts: GeoDataFrame 병합 중 오류: {e}")
        return gpd.GeoDataFrame()
        
    for c in cols[1:]: # 병합 후 NaN 값 처리 (left merge로 인해 발생 가능)
        if c in merged.columns:
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
    if df_hosp is None or df_hosp.empty or \
       df_beds is None or df_beds.empty or \
       _gdf_gu is None or _gdf_gu.empty:
        st.warning("make_merged_avg_beds: 입력 데이터가 유효하지 않습니다.")
        return gpd.GeoDataFrame()

    types = ["종합병원", "병원", "의원", "요양병원"]
    
    required_cols = ["gu"] + types
    if not all(col in df_hosp.columns for col in required_cols) or \
       not all(col in df_beds.columns for col in required_cols):
        st.error(f"make_merged_avg_beds: df_hosp 또는 df_beds에 필요한 컬럼 ('gu' 및 {types})이(가) 없습니다.")
        return gpd.GeoDataFrame()

    if "gu" not in _gdf_gu.columns:
        st.error("make_merged_avg_beds: _gdf_gu에 'gu' 컬럼이 없습니다.")
        return gpd.GeoDataFrame()

    records = []
    # df_hosp와 df_beds의 행 순서가 동일하고 'gu'를 포함한다고 가정
    # 좀 더 안전하게 하려면 'gu'를 기준으로 merge 하거나 set_index 후 loc 접근
    
    # 'gu'를 기준으로 데이터 정렬 및 인덱싱 (안전한 접근을 위해)
    df_hosp_indexed = df_hosp.set_index("gu")
    df_beds_indexed = df_beds.set_index("gu")
    
    common_gus = df_hosp_indexed.index.intersection(df_beds_indexed.index)

    for gu_name in common_gus:
        row_hosp = df_hosp_indexed.loc[gu_name]
        row_beds = df_beds_indexed.loc[gu_name]
        for t in types:
            hosp_cnt = pd.to_numeric(row_hosp.get(t), errors="coerce") # .get으로 안전하게
            bed_cnt = pd.to_numeric(row_beds.get(t), errors="coerce")
            avg = 0.0
            if pd.notna(hosp_cnt) and hosp_cnt > 0 and pd.notna(bed_cnt):
                avg = bed_cnt / hosp_cnt
            records.append({"gu": gu_name, "type": t, "avg_beds": avg})
    
    if not records:
        st.warning("make_merged_avg_beds: 평균 병상 수 계산을 위한 레코드가 생성되지 않았습니다.")
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
            merged[t] = merged[t].fillna(0)
    return merged


def draw_hospital_count_choropleth(merged_counts: gpd.GeoDataFrame, width=800, height=600): # 타입 힌트 gpd.GeoDataFrame
    """
    Folium Choropleth Map 객체 생성 (구별 의료기관 수)
    merged_counts: make_merged_counts() 결과 GeoDataFrame
    """
    if merged_counts is None or merged_counts.empty or 'geometry' not in merged_counts.columns:
        st.warning("draw_hospital_count_choropleth: 유효한 GeoDataFrame이 없어 지도를 그릴 수 없습니다.")
        m_empty = folium.Map(location=[37.55, 126.98], zoom_start=11, tiles="CartoDB positron")
        return m_empty

    m = folium.Map(
        location=[37.55, 126.98],
        zoom_start=11,
        tiles="CartoDB positron", # 기본 타일
        attr="© CartoDB" # 기본 속성
    )

    types_counts = ["소계", "종합병원", "병원", "의원", "요양병원"]
    colors_counts = ["YlOrRd", "PuBu", "Greens", "Purples", "OrRd"] # 색상 맵

    # geo_data 인자에 GeoDataFrame 직접 전달
    # columns 인자에 key 컬럼('gu')과 value 컬럼(inst) 전달
    # key_on 인자에 GeoJSON 피처의 프로퍼티 경로 전달
    
    missing_cols = [col for col in types_counts if col not in merged_counts.columns]
    if missing_cols:
        st.warning(f"draw_hospital_count_choropleth: merged_counts에 다음 컬럼이 없습니다: {missing_cols}. 해당 레이어는 생략됩니다.")

    for inst, cmap in zip(types_counts, colors_counts):
        if inst not in merged_counts.columns:
            continue
        try:
            folium.Choropleth(
                geo_data=merged_counts, # GeoDataFrame 직접 전달
                name=f"{inst} 수",      # 레이어 이름
                data=merged_counts,     # 데이터
                columns=["gu", inst],   # 키와 값 컬럼
                key_on="feature.properties.gu", # GeoJSON 피처의 'gu' 프로퍼티와 매칭
                fill_color=cmap,
                fill_opacity=0.6,
                line_opacity=0.4,
                legend_name=f"{inst} 수",
                highlight=True
            ).add_to(m)
        except Exception as e:
            st.error(f"Choropleth 레이어 '{inst}' 생성 중 오류: {e}")


    try:
        for _, row in merged_counts.iterrows():
            if row.geometry and row.geometry.centroid: # 유효한 geometry와 centroid 확인
                # GeoPandas 0.7.0 이상에서는 centroid가 Point 객체를 반환하므로 .x, .y로 접근
                lon, lat = row.geometry.centroid.x, row.geometry.centroid.y
                folium.Marker( # folium.map.Marker 대신 folium.Marker
                    location=[lat, lon],
                    icon=DivIcon(
                        icon_size=(150, 36),
                        icon_anchor=(0, 0), # 아이콘 기준점
                        html=f"<div style='font-size:10px; font-weight:bold; color:black;'>{row.get('gu', '')}</div>",
                    ),
                ).add_to(m)
    except Exception as e:
        st.error(f"지도에 구 이름 마커 추가 중 오류: {e}")

    # Stamen 타일 레이어는 현재 FileNotFoundError 발생 가능성 있으므로 주석 유지
    # folium.TileLayer(tiles="Stamen Toner", name="Toner", attr="Map tiles by Stamen Design, CC BY 3.0").add_to(m)
    # folium.TileLayer(tiles="Stamen Terrain", name="Terrain", attr="Map tiles by Stamen Design, CC BY 3.0").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def draw_avg_beds_choropleth(merged_avg: gpd.GeoDataFrame, width=800, height=600): # 타입 힌트 gpd.GeoDataFrame
    """
    Folium Choropleth Map 객체 생성 (구별 평균 병상 수)
    merged_avg: make_merged_avg_beds() 결과 GeoDataFrame
    """
    if merged_avg is None or merged_avg.empty or 'geometry' not in merged_avg.columns:
        st.warning("draw_avg_beds_choropleth: 유효한 GeoDataFrame이 없어 지도를 그릴 수 없습니다.")
        m_empty = folium.Map(location=[37.55, 126.98], zoom_start=11, tiles="CartoDB positron")
        return m_empty
        
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
        try:
            folium.Choropleth(
                geo_data=merged_avg, # GeoDataFrame 직접 전달
                name=f"{t} 평균 병상수", # 원본 legend_name "평균 병상/병원" 대신 "평균 병상수"로 변경
                data=merged_avg,
                columns=["gu", t],
                key_on="feature.properties.gu",
                fill_color=cmap,
                fill_opacity=0.7,
                line_opacity=0.2, # 원본 line_opacity 유지
                legend_name=f"{t} 평균 병상 수", # 원본과 동일
                highlight=True
            ).add_to(m)
        except Exception as e:
            st.error(f"Choropleth 레이어 '{t}' (평균 병상 수) 생성 중 오류: {e}")

    try:
        for _, row in merged_avg.iterrows():
            if row.geometry and row.geometry.centroid:
                lon, lat = row.geometry.centroid.x, row.geometry.centroid.y
                folium.Marker(
                    location=[lat, lon],
                    icon=DivIcon(
                        icon_size=(150, 36),
                        icon_anchor=(0, 0),
                        html=f"<div style='font-size:10px; font-weight:bold; color:black;'>{row.get('gu', '')}</div>",
                    ),
                ).add_to(m)
    except Exception as e:
        st.error(f"지도에 구 이름 마커 추가 중 오류 (평균 병상 수): {e}")

    # Stamen 타일 레이어 주석 유지
    # folium.TileLayer(tiles="Stamen Toner", name="Toner", attr="Map tiles by Stamen Design, CC BY 3.0").add_to(m)
    # folium.TileLayer(tiles="Stamen Terrain", name="Terrain", attr="Map tiles by Stamen Design, CC BY 3.0").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m
