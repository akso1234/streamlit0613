import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from utils import set_korean_font, load_csv, load_geojson # utils.py의 함수 임포트
import os
import numpy as np
from shapely.geometry import shape # 중심점 계산을 위해 추가

# --- 데이터 전처리 및 추출 함수 (이전과 동일) ---
@st.cache_data
def preprocess_park_data_cached(df_raw):
    if df_raw is None:
        st.error("preprocess_park_data_cached: 원본 데이터가 None입니다.")
        return pd.DataFrame(), pd.DataFrame()

    df_processed = df_raw.copy()
    gu_column_tuple = None
    if len(df_processed.columns) > 1:
        potential_gu_column_tuple = df_processed.columns[1]
        gu_column_tuple = potential_gu_column_tuple
    else:
        st.error("공원 데이터의 컬럼 수가 2개 미만입니다. 자치구 정보를 찾을 수 없습니다.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        df_processed = df_processed.set_index(gu_column_tuple)
        df_processed.index.name = '자치구'
        if len(df_raw.columns) > 0:
            first_column_tuple_to_drop = df_raw.columns[0]
            if first_column_tuple_to_drop in df_processed.columns:
                 df_processed = df_processed.drop(columns=[first_column_tuple_to_drop])
    except KeyError as e:
        st.error(f"공원 데이터 인덱스 설정 중 KeyError: {e}. 컬럼명: {gu_column_tuple}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"공원 데이터 인덱스 설정 중 예기치 않은 오류: {e}")
        return pd.DataFrame(), pd.DataFrame()

    df_seoul_total = pd.DataFrame()
    if '소계' in df_processed.index:
        df_seoul_total = df_processed.loc[['소계']].copy()
    else:
        st.warning("가공된 공원 데이터에서 '소계'(서울시 전체 합계) 행을 찾을 수 없습니다.")

    districts_to_exclude = ['소계', '서울대공원']
    districts_df = df_processed.drop(index=districts_to_exclude, errors='ignore')

    if districts_df.empty:
        st.warning("자치구별 공원 데이터(districts_df)가 비어있습니다.")
    return districts_df, df_seoul_total

@st.cache_data
def extract_total_park_stats_cached(districts_df, years_str_list):
    if districts_df is None or districts_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    park_counts_list = []
    park_areas_list = []

    for district_name in districts_df.index:
        try:
            district_data_series = districts_df.loc[district_name]
        except KeyError:
            st.warning(f"'{district_name}' 자치구 데이터를 districts_df에서 찾을 수 없습니다.")
            continue

        counts = {'자치구': district_name}
        areas = {'자치구': district_name}

        for year_str in years_str_list:
            count_col_tuple = (year_str, '합계', '소계', '소계', '공원수 (개소)')
            area_col_tuple = (year_str, '합계', '소계', '소계', '면적 (천㎡)')

            count_val = district_data_series.get(count_col_tuple)
            area_val = district_data_series.get(area_col_tuple)

            counts[year_str] = pd.to_numeric(count_val, errors='coerce')
            areas[year_str] = pd.to_numeric(area_val, errors='coerce')

        park_counts_list.append(counts)
        park_areas_list.append(areas)

    df_park_counts = pd.DataFrame(park_counts_list)
    if not df_park_counts.empty:
        df_park_counts = df_park_counts.set_index('자치구').fillna(0)
        df_park_counts = df_park_counts[[col for col in years_str_list if col in df_park_counts.columns]]

    df_park_area = pd.DataFrame(park_areas_list)
    if not df_park_area.empty:
        df_park_area = df_park_area.set_index('자치구').fillna(0)
        df_park_area = df_park_area[[col for col in years_str_list if col in df_park_area.columns]]

    return df_park_counts, df_park_area

# --- 시각화 함수 ---
def plot_yearly_district_comparison(df_metric, metric_name, unit, selected_year_str, bar_color='skyblue'):
    if df_metric.empty or selected_year_str not in df_metric.columns:
        st.info(f"{selected_year_str}년 {metric_name} 데이터가 없습니다.")
        return

    data_for_year = df_metric[[selected_year_str]].sort_values(by=selected_year_str, ascending=False).reset_index()
    if data_for_year.empty:
        st.info(f"{selected_year_str}년 정렬 후 {metric_name} 데이터가 없습니다.")
        return

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.barplot(x=selected_year_str, y='자치구', data=data_for_year, color=bar_color, ax=ax)
    ax.set_title(f'{selected_year_str}년 자치구별 총 공원 {metric_name}', fontsize=16)
    ax.set_xlabel(f'총 공원 {metric_name} ({unit})', fontsize=12)
    ax.set_ylabel('자치구', fontsize=12)
    ax.tick_params(axis='x', labelsize=10); ax.tick_params(axis='y', labelsize=9)
    for index, value in enumerate(data_for_year[selected_year_str]):
        if pd.notna(value) and value > 0:
            formatted_value = f'{value:,.1f}' if isinstance(value, float) and unit == "천㎡" else f'{int(value):,}'
            ax.text(value, index, f' {formatted_value}', va='center', ha='left', fontsize=8, color='black')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.tight_layout(); st.pyplot(fig)

def plot_seoul_total_distribution(df_seoul_total, selected_year_str):
    if df_seoul_total is None or df_seoul_total.empty:
        st.info(f"{selected_year_str}년 서울시 전체 공원 유형별 면적 데이터가 없어 파이 차트를 생성할 수 없습니다.")
        return
    if '소계' not in df_seoul_total.index:
        st.warning("plot_seoul_total_distribution: '소계' 인덱스를 가진 서울시 전체 데이터가 없습니다.")
        return

    seoul_total_row_data = df_seoul_total.loc['소계']
    park_type_areas = {}
    urban_facility_park_types = ['근린공원', '어린이공원', '소공원', '묘지공원', '문화공원', '체육공원', '역사공원', '수변공원', '생태공원', '가로공원']
    for park_type in urban_facility_park_types:
        col_name = (selected_year_str, '도시공원', '도시계획시설(공원) ', park_type, '면적 (천㎡)')
        area = pd.to_numeric(seoul_total_row_data.get(col_name), errors='coerce')
        if pd.notna(area) and area > 0: park_type_areas[park_type] = area
    col_name_natural = (selected_year_str, '도시공원', '도시자연공원구역', '소계', '면적 (천㎡)')
    area_dz = pd.to_numeric(seoul_total_row_data.get(col_name_natural), errors='coerce')
    if pd.notna(area_dz) and area_dz > 0: park_type_areas['도시자연공원구역'] = area_dz

    if not park_type_areas:
        st.info(f"{selected_year_str}년 서울시 전체 공원 유형별 면적 데이터를 추출할 수 없었습니다.")
        return

    park_types_series = pd.Series(park_type_areas).sort_values(ascending=False)

    # 작은 비율의 항목들을 '기타'로 묶기
    threshold_ratio = 2.0 # 2% 미만은 '기타'로 (조정 가능)
    total_area = park_types_series.sum()
    small_slices = park_types_series[park_types_series / total_area * 100 < threshold_ratio]
    
    if not small_slices.empty and len(park_types_series) > len(small_slices) and len(small_slices) > 1 : # '기타'로 묶을 항목이 2개 이상이고, 모든 항목이 작은 조각은 아닌 경우
        other_sum = small_slices.sum()
        park_types_series_major = park_types_series[park_types_series / total_area * 100 >= threshold_ratio]
        if other_sum > 0:
            park_types_series_major['기타'] = other_sum
        park_types_series_to_plot = park_types_series_major.sort_values(ascending=False)
    else:
        park_types_series_to_plot = park_types_series

    num_slices = len(park_types_series_to_plot)
    explode = [0.0] * num_slices
    if num_slices > 2 : 
        if '기타' in park_types_series_to_plot.index:
            explode[park_types_series_to_plot.index.get_loc('기타')] = 0.05
        else:
            if num_slices > 3: # 작은 조각이 여러 개 있을 경우 약간씩 떼어냄
                 explode[-1] = 0.05 
                 explode[-2] = 0.03


    fig, ax = plt.subplots(figsize=(13, 10)) # figsize 조정

    def autopct_format(pct):
        # 작은 비율의 레이블은 퍼센트만 표시하거나 표시하지 않도록 조정
        return f'{pct:.1f}%' if pct >= 2.5 else '' # 2.5% 이상만 퍼센트 표시 (조정 가능)

    patches, texts, autotexts = ax.pie(
        park_types_series_to_plot,
        labels=park_types_series_to_plot.index,
        autopct=autopct_format, 
        startangle=140,
        pctdistance=0.85, # 퍼센트 텍스트 위치
        labeldistance=1.05, # 레이블 텍스트 위치
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
        explode=explode 
    )

    for text in texts:
        text.set_fontsize(9) 
    for autotext in autotexts:
        autotext.set_fontsize(8) 
        autotext.set_color('black')

    ax.set_title(f'서울시 도시공원 유형별 면적 분포 ({selected_year_str}년)', fontsize=16)
    ax.axis('equal')
    plt.tight_layout()
    st.pyplot(fig)

def create_choropleth_map(df_metric, geo_data, year_str, metric_name, unit, fill_color_map='Blues'):
    if df_metric.empty or year_str not in df_metric.columns:
        st.info(f"{year_str}년 {metric_name} 지도 데이터를 생성할 수 없습니다.")
        return None

    data_to_map = df_metric[[year_str]].copy()
    if data_to_map.index.name != '자치구':
        st.warning("create_choropleth_map: 지도용 데이터의 인덱스가 '자치구'가 아닙니다.")
        return None

    m = folium.Map(location=[37.5665, 126.9780], zoom_start=10.5, tiles='CartoDB positron')

    if geo_data:
        try:
            choropleth_layer = folium.Choropleth(
                geo_data=geo_data,
                data=data_to_map.reset_index(),
                columns=['자치구', year_str],
                key_on='feature.properties.name',
                fill_color=fill_color_map,
                fill_opacity=0.7,
                line_opacity=0.3,
                legend_name=f'{year_str}년 {metric_name} ({unit})',
                highlight=True,
                name=f'{year_str}년 {metric_name}'
            ).add_to(m)

            for feature in choropleth_layer.geojson.data['features']:
                gu_name_geojson = feature['properties'].get('name')
                if not gu_name_geojson:
                    continue
                try:
                    geom = shape(feature['geometry'])
                    center_point = geom.centroid
                    center_lon, center_lat = center_point.x, center_point.y
                except Exception:
                    continue

                folium.Marker(
                    location=[center_lat, center_lon],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 9pt; font-weight: bold; color: black; background-color: transparent; white-space: nowrap;">{gu_name_geojson}</div>'
                    )
                ).add_to(m)
            return m
        except Exception as e:
            st.error(f"Folium 지도 생성 중 오류: {e}")
            return None
    else:
        st.warning("GeoJSON 데이터가 없어 지도를 생성할 수 없습니다.")
        return None

# --- Streamlit 페이지 구성 ---
def run_park_analysis_page():
    st.title("🌳 서울시 공원 분석")
    set_korean_font()

    park_years_int = [2019, 2020, 2021, 2022, 2023]
    park_years_str = [str(y) for y in park_years_int]

    if "selected_year_park" not in st.session_state:
        st.session_state.selected_year_park = park_years_int[-1]

    selected_year_int = st.slider(
        "조회 연도 선택",
        min_value=park_years_int[0],
        max_value=park_years_int[-1],
        step=1,
        value=st.session_state.selected_year_park,
        key="park_year_slider_main"
    )
    st.session_state.selected_year_park = selected_year_int
    selected_year_str = str(selected_year_int)

    df_raw_parks = load_csv("data/공원_20250525010638.csv", header_config=[0,1,2,3,4], na_values_config='-')
    if df_raw_parks is None:
        st.error("공원 데이터를 로드하지 못했습니다. 'data/공원_20250525010638.csv' 파일을 확인해주세요.")
        return

    districts_df_parks, df_seoul_total_parks = preprocess_park_data_cached(df_raw_parks)
    if districts_df_parks is None or districts_df_parks.empty:
        st.error("공원 데이터 전처리 중 오류가 발생했거나, 유효한 자치구 데이터가 없습니다.")
        return

    df_park_counts, df_park_area = extract_total_park_stats_cached(districts_df_parks, park_years_str)

    geojson_url = 'https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json'
    @st.cache_data
    def get_geojson_cached_park_page(url): return load_geojson(url)
    seoul_geo_data_parks = get_geojson_cached_park_page(geojson_url)

    tab1, tab2, tab3 = st.tabs([
        "📊 연도별 자치구 현황",
        "📈 서울시 전체 현황",
        "🗺️ 지도 시각화"
    ])

    with tab1:
        st.subheader(f"{selected_year_str}년 자치구별 공원 현황")
        st.markdown("##### 총 공원 면적 (천㎡)")
        if not df_park_area.empty:
            plot_yearly_district_comparison(df_park_area, "면적", "천㎡", selected_year_str, bar_color='cornflowerblue')
        else: st.info("자치구별 공원 면적 데이터가 없습니다.")

        st.markdown("##### 총 공원 수 (개소)")
        if not df_park_counts.empty:
            plot_yearly_district_comparison(df_park_counts, "수", "개소", selected_year_str, bar_color='mediumseagreen')
        else: st.info("자치구별 공원 수 데이터가 없습니다.")

        st.markdown("##### 공원 1개소당 평균 면적 (천㎡/개소)")
        if not df_park_counts.empty and not df_park_area.empty:
            df_park_counts_for_avg = df_park_counts.replace(0, np.nan)
            if not df_park_counts_for_avg.empty and selected_year_str in df_park_counts_for_avg.columns:
                df_avg_park_area = df_park_area.div(df_park_counts_for_avg).fillna(0)
                plot_yearly_district_comparison(df_avg_park_area, "1개소당 평균 면적", "천㎡/개소", selected_year_str, bar_color='lightcoral')
            else: st.info(f"{selected_year_str}년 공원 수가 모두 0이거나 데이터가 없어 1개소당 평균 면적을 계산할 수 없습니다.")
        else: st.info("자치구별 공원 1개소당 평균 면적을 계산할 데이터가 부족합니다.")

    with tab2:
        st.subheader(f"{selected_year_str}년 서울시 전체 도시공원 유형별 면적 분포")
        plot_seoul_total_distribution(df_seoul_total_parks, selected_year_str)

    with tab3:
        st.subheader(f"{selected_year_str}년 공원 지도 시각화")
        map_metric_parks = st.selectbox("지도 표시 항목 선택:", ["총 공원 수", "총 공원 면적", "1개소당 평균 면적"], key="park_map_metric_sb_main_tab")

        if seoul_geo_data_parks:
            park_map_to_display = None
            if map_metric_parks == "총 공원 수":
                if not df_park_counts.empty:
                    park_map_to_display = create_choropleth_map(df_park_counts, seoul_geo_data_parks, selected_year_str, "총 공원 수", "개소", "Greens")
                else: st.info("총 공원 수 데이터가 없어 지도를 생성할 수 없습니다.")

            elif map_metric_parks == "총 공원 면적":
                if not df_park_area.empty:
                    park_map_to_display = create_choropleth_map(df_park_area, seoul_geo_data_parks, selected_year_str, "총 공원 면적", "천㎡", "Blues")
                else: st.info("총 공원 면적 데이터가 없어 지도를 생성할 수 없습니다.")

            elif map_metric_parks == "1개소당 평균 면적":
                if not df_park_counts.empty and not df_park_area.empty:
                    df_park_counts_for_avg_map = df_park_counts.replace(0, np.nan)
                    if not df_park_counts_for_avg_map.empty and selected_year_str in df_park_counts_for_avg_map.columns:
                        df_avg_park_area_map = df_park_area.div(df_park_counts_for_avg_map).fillna(0)
                        park_map_to_display = create_choropleth_map(df_avg_park_area_map, seoul_geo_data_parks, selected_year_str, "1개소당 평균 면적", "천㎡/개소", "Oranges")
                    else: st.info(f"{selected_year_str}년 1개소당 평균 면적을 계산할 데이터(공원 수)가 없어 지도를 생성할 수 없습니다.")
                else: st.info("1개소당 평균 면적 데이터를 계산할 수 없어 지도를 생성할 수 없습니다.")

            if park_map_to_display:
                st_folium(park_map_to_display, width=800, height=600)
            elif seoul_geo_data_parks :
                 st.info("선택한 항목에 대한 지도를 표시할 데이터가 없습니다.")
        else:
            st.warning("GeoJSON 데이터가 로드되지 않아 지도를 표시할 수 없습니다.")

if __name__ == "__main__":
    run_park_analysis_page()
