# --- START OF map_utils.py (일부, LayerControl 확인) ---
import pandas as pd
import folium
from folium.features import DivIcon
import streamlit as st
import geopandas as gpd

# ... (make_merged_counts, make_merged_avg_beds 함수는 이전과 동일) ...
@st.cache_data
def make_merged_counts(df_hosp: pd.DataFrame, _gdf_gu: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if df_hosp is None or df_hosp.empty or _gdf_gu is None or _gdf_gu.empty:
        # st.warning("make_merged_counts: 입력 데이터가 유효하지 않습니다.") # 페이지 레벨에서 처리 권장
        return gpd.GeoDataFrame()

    cols = ["gu", "소계", "종합병원", "병원", "의원", "요양병원"]
    
    missing_cols_df_hosp = [col for col in cols if col not in df_hosp.columns]
    if missing_cols_df_hosp:
        # st.error(f"make_merged_counts: df_hosp에 필요한 컬럼 {missing_cols_df_hosp}이(가) 없습니다.")
        return gpd.GeoDataFrame()
    
    if "gu" not in _gdf_gu.columns:
        # st.error("make_merged_counts: _gdf_gu에 'gu' 컬럼이 없습니다.")
        return gpd.GeoDataFrame()
        
    df_counts = df_hosp[cols].copy()
    for c in cols[1:]:
        df_counts[c] = pd.to_numeric(df_counts[c], errors="coerce").fillna(0).astype(int)

    try:
        merged = _gdf_gu.merge(df_counts, on="gu", how="left")
    except Exception as e:
        # st.error(f"make_merged_counts: GeoDataFrame 병합 중 오류: {e}")
        return gpd.GeoDataFrame()
        
    for c in cols[1:]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0).astype(int)
    return merged


@st.cache_data
def make_merged_avg_beds(df_hosp: pd.DataFrame, df_beds: pd.DataFrame, _gdf_gu: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if df_hosp is None or df_hosp.empty or \
       df_beds is None or df_beds.empty or \
       _gdf_gu is None or _gdf_gu.empty:
        # st.warning("make_merged_avg_beds: 입력 데이터가 유효하지 않습니다.")
        return gpd.GeoDataFrame()

    types = ["종합병원", "병원", "의원", "요양병원"]
    records = []

    required_cols_hosp = ["gu"] + types
    required_cols_beds = ["gu"] + types
    if not all(col in df_hosp.columns for col in required_cols_hosp) or \
       not all(col in df_beds.columns for col in required_cols_beds):
        # st.error(f"make_merged_avg_beds: df_hosp 또는 df_beds에 필요한 컬럼이 부족합니다.")
        return gpd.GeoDataFrame()

    if "gu" not in _gdf_gu.columns:
        # st.error("make_merged_avg_beds: _gdf_gu에 'gu' 컬럼이 없습니다.")
        return gpd.GeoDataFrame()

    df_hosp_indexed = df_hosp.set_index('gu')
    df_beds_indexed = df_beds.set_index('gu')
    
    common_gus = df_hosp_indexed.index.intersection(df_beds_indexed.index)

    for gu_val in common_gus:
        for t in types:
            hosp_cnt = pd.to_numeric(df_hosp_indexed.loc[gu_val, t], errors="coerce")
            bed_cnt = pd.to_numeric(df_beds_indexed.loc[gu_val, t], errors="coerce")
            avg = 0.0
            if pd.notna(hosp_cnt) and hosp_cnt > 0 and pd.notna(bed_cnt):
                avg = bed_cnt / hosp_cnt
            records.append({"gu": gu_val, "type": t, "avg_beds": avg})
    
    if not records:
        # st.warning("make_merged_avg_beds: 평균 병상 수 계산을 위한 레코드가 생성되지 않았습니다.")
        return gpd.GeoDataFrame()

    stats = pd.DataFrame(records)
    try:
        pivot = stats.pivot(index="gu", columns="type", values="avg_beds").reset_index().fillna(0)
    except Exception as e:
        # st.error(f"make_merged_avg_beds: pivot 테이블 생성 중 오류: {e}")
        return gpd.GeoDataFrame()

    try:
        merged = _gdf_gu.merge(pivot, on="gu", how="left")
    except Exception as e:
        # st.error(f"make_merged_avg_beds: GeoDataFrame 병합 중 오류: {e}")
        return gpd.GeoDataFrame()
        
    for t in types:
        if t in merged.columns:
            merged[t] = merged[t].fillna(0)
    return merged


def draw_hospital_count_choropleth(merged_counts: gpd.GeoDataFrame, width=800, height=600):
    if merged_counts is None or merged_counts.empty or 'geometry' not in merged_counts.columns:
        # st.warning("draw_hospital_count_choropleth: 유효한 GeoDataFrame이 없어 지도를 그릴 수 없습니다.")
        m_empty = folium.Map(location=[37.55, 126.98], zoom_start=11, tiles="CartoDB positron")
        folium.LayerControl(collapsed=False).add_to(m_empty) # 빈 지도에도 추가 (일관성)
        return m_empty

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
        try:
            folium.Choropleth(
                geo_data=merged_counts, 
                name=f"{inst} 수",      
                data=merged_counts,     
                columns=["gu", inst],   
                key_on="feature.properties.gu", 
                fill_color=cmap,
                fill_opacity=0.6,
                line_opacity=0.4,
                legend_name=f"{inst} 수",
                highlight=True,
                show=True, # 기본값 True, 명시적으로 추가
                overlay=True # 다른 레이어와 중첩 가능하도록
            ).add_to(m)
        except Exception as e:
            st.error(f"Choropleth 레이어 '{inst}' 생성 중 오류: {e}")

    try:
        for _, row in merged_counts.iterrows():
            if row.geometry and hasattr(row.geometry, 'centroid') and row.geometry.centroid:
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
        # st.error(f"지도에 구 이름 마커 추가 중 오류: {e}") # 너무 많은 로그 방지
        pass

    # LayerControl이 항상 추가되도록 보장
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def draw_avg_beds_choropleth(merged_avg: gpd.GeoDataFrame, width=800, height=600):
    if merged_avg is None or merged_avg.empty or 'geometry' not in merged_avg.columns:
        # st.warning("draw_avg_beds_choropleth: 유효한 GeoDataFrame이 없어 지도를 그릴 수 없습니다.")
        m_empty = folium.Map(location=[37.55, 126.98], zoom_start=11, tiles="CartoDB positron")
        folium.LayerControl(collapsed=False).add_to(m_empty) # 빈 지도에도 추가 (일관성)
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
                geo_data=merged_avg, 
                name=f"{t} 평균 병상수", 
                data=merged_avg,
                columns=["gu", t],
                key_on="feature.properties.gu",
                fill_color=cmap,
                fill_opacity=0.7,
                line_opacity=0.2, 
                legend_name=f"{t} 평균 병상 수", 
                highlight=True,
                show=True, # 기본값 True, 명시적으로 추가
                overlay=True # 다른 레이어와 중첩 가능하도록
            ).add_to(m)
        except Exception as e:
            st.error(f"Choropleth 레이어 '{t}' (평균 병상 수) 생성 중 오류: {e}")

    try:
        for _, row in merged_avg.iterrows():
            if row.geometry and hasattr(row.geometry, 'centroid') and row.geometry.centroid:
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
        # st.error(f"지도에 구 이름 마커 추가 중 오류 (평균 병상 수): {e}") # 너무 많은 로그 방지
        pass
        
    # LayerControl이 항상 추가되도록 보장
    folium.LayerControl(collapsed=False).add_to(m)
    return m
# --- END OF map_utils.py ---
