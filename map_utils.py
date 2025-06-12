import pandas as pd
import folium
from folium.features import DivIcon
import streamlit as st

@st.cache_data
def make_merged_counts(df_hosp: pd.DataFrame, _gdf_gu):
    """
    df_hosp: '병원수' 정보가 담긴 DataFrame (컬럼: gu, 소계, 종합병원, 병원, 의원, 요양병원)
    _gdf_gu: GeoDataFrame (구별 경계)
    → 구별 의료기관 수(소계, 종합병원, 병원, 의원, 요양병원) GeoDataFrame 반환
    """
    cols = ["gu", "소계", "종합병원", "병원", "의원", "요양병원"]
    df_counts = df_hosp[cols].copy()
    for c in cols[1:]:
        df_counts[c] = pd.to_numeric(df_counts[c], errors="coerce").fillna(0)

    merged = _gdf_gu.merge(df_counts, on="gu", how="left")
    for c in cols[1:]:
        merged[c] = merged[c].fillna(0)
    return merged


@st.cache_data
def make_merged_avg_beds(df_hosp: pd.DataFrame, df_beds: pd.DataFrame, _gdf_gu):
    """
    df_hosp: '병원수' 정보
    df_beds: '병상수' 정보
    _gdf_gu: GeoDataFrame (구별 경계)
    → 구별 평균 병상 수(종합병원, 병원, 의원, 요양병원) GeoDataFrame 반환
    """
    types = ["종합병원", "병원", "의원", "요양병원"]
    records = []

    for idx, row in df_hosp.iterrows():
        gu = row["gu"]
        for t in types:
            hosp_cnt = pd.to_numeric(row[t], errors="coerce")
            bed_cnt = pd.to_numeric(df_beds.loc[idx, t], errors="coerce")
            avg = (bed_cnt / hosp_cnt) if (hosp_cnt and hosp_cnt > 0 and not pd.isna(bed_cnt)) else 0.0
            records.append({"gu": gu, "type": t, "avg_beds": avg})

    stats = pd.DataFrame(records)
    pivot = stats.pivot(index="gu", columns="type", values="avg_beds").reset_index().fillna(0)
    merged = _gdf_gu.merge(pivot, on="gu", how="left")
    for t in types:
        merged[t] = merged[t].fillna(0)
    return merged


def draw_hospital_count_choropleth(merged_counts, width=800, height=600):
    """
    Folium Choropleth Map 객체 생성 (구별 의료기관 수)
    merged_counts: make_merged_counts() 결과 GeoDataFrame
    """
    m = folium.Map(
        location=[37.55, 126.98],
        zoom_start=11,
        tiles="CartoDB positron",
        attr="&copy; CartoDB",
    )

    types_counts = ["소계", "종합병원", "병원", "의원", "요양병원"]
    colors_counts = ["YlOrRd", "PuBu", "Greens", "Purples", "OrRd"]
    for inst, cmap in zip(types_counts, colors_counts):
        folium.Choropleth(
            geo_data=merged_counts.__geo_interface__,
            name=f"{inst} 수",
            data=merged_counts,
            columns=["gu", inst],
            key_on="feature.properties.gu",
            fill_color=cmap,
            fill_opacity=0.6,
            line_opacity=0.4,
            legend_name=f"{inst} 수",
        ).add_to(m)

    for _, row in merged_counts.iterrows():
        lon, lat = row.geometry.centroid.coords[0]
        folium.map.Marker(
            location=[lat, lon],
            icon=DivIcon(
                icon_size=(150, 36),
                icon_anchor=(0, 0),
                html=f"<div style='font-size:10px; font-weight:bold'>{row['gu']}</div>",
            ),
        ).add_to(m)

    folium.TileLayer(tiles="Stamen Toner", name="Toner", attr="Map tiles by Stamen Design, CC BY 3.0").add_to(m)
    folium.TileLayer(tiles="Stamen Terrain", name="Terrain", attr="Map tiles by Stamen Design, CC BY 3.0").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def draw_avg_beds_choropleth(merged_avg, width=800, height=600):
    """
    Folium Choropleth Map 객체 생성 (구별 평균 병상 수)
    merged_avg: make_merged_avg_beds() 결과 GeoDataFrame
    """
    m = folium.Map(
        location=[37.55, 126.98],
        zoom_start=11,
        tiles="CartoDB positron",
        attr="&copy; CartoDB",
    )

    types_avg = ["종합병원", "병원", "의원", "요양병원"]
    colors_avg = ["YlOrRd", "PuBu", "Greens", "PuBuGn"]
    for t, cmap in zip(types_avg, colors_avg):
        folium.Choropleth(
            geo_data=merged_avg.__geo_interface__,
            name=f"{t} 평균 병상/병원",
            data=merged_avg,
            columns=["gu", t],
            key_on="feature.properties.gu",
            fill_color=cmap,
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=f"{t} 평균 병상 수",
        ).add_to(m)

    for _, row in merged_avg.iterrows():
        lon, lat = row.geometry.centroid.coords[0]
        folium.map.Marker(
            location=[lat, lon],
            icon=DivIcon(
                icon_size=(150, 36),
                icon_anchor=(0, 0),
                html=f"<div style='font-size:10px; font-weight:bold'>{row['gu']}</div>",
            ),
        ).add_to(m)

    folium.TileLayer(tiles="Stamen Toner", name="Toner", attr="Map tiles by Stamen Design, CC BY 3.0").add_to(m)
    folium.TileLayer(tiles="Stamen Terrain", name="Terrain", attr="Map tiles by Stamen Design, CC BY 3.0").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m
