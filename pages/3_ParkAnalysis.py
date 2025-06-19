# --- START OF 3_ParkAnalysis.py ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
# utils.py, data_loader.py (필요시), chart_utils.py (필요시)가
# 프로젝트 루트에 있고, 이 파일이 pages 폴더에 있다고 가정합니다.
from utils import set_korean_font, load_csv, load_geojson
# 만약 data_loader를 사용한다면: from data_loader import ...
# 만약 chart_utils를 사용한다면: from chart_utils import ...
import os
import numpy as np
from shapely.geometry import shape
from matplotlib.ticker import FuncFormatter


# --- 데이터 전처리 및 추출 함수 ---
@st.cache_data
def preprocess_park_data_cached(df_raw):
    if df_raw is None:
        # st.error("preprocess_park_data_cached: 원본 데이터가 None입니다.")
        return pd.DataFrame(), pd.DataFrame()

    df_processed = df_raw.copy()
    gu_column_tuple = None
    if len(df_processed.columns) > 1:
        potential_gu_column_tuple = df_processed.columns[1]
        gu_column_tuple = potential_gu_column_tuple
    else:
        # st.error("공원 데이터의 컬럼 수가 2개 미만입니다. 자치구 정보를 찾을 수 없습니다.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        df_processed = df_processed.set_index(gu_column_tuple)
        df_processed.index.name = '자치구'
        if len(df_raw.columns) > 0:
            first_column_tuple_to_drop = df_raw.columns[0]
            if first_column_tuple_to_drop in df_processed.columns:
                 df_processed = df_processed.drop(columns=[first_column_tuple_to_drop])
    except KeyError:
        # st.error(f"공원 데이터 인덱스 설정 중 KeyError: {e}. 컬럼명: {gu_column_tuple}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception:
        # st.error(f"공원 데이터 인덱스 설정 중 예기치 않은 오류: {e}")
        return pd.DataFrame(), pd.DataFrame()

    df_seoul_total = pd.DataFrame()
    if '소계' in df_processed.index:
        df_seoul_total = df_processed.loc[['소계']].copy()
    # else:
        # st.warning("가공된 공원 데이터에서 '소계'(서울시 전체 합계) 행을 찾을 수 없습니다.")

    districts_to_exclude = ['소계', '서울대공원']
    districts_df = df_processed.drop(index=districts_to_exclude, errors='ignore')

    # if districts_df.empty:
        # st.warning("자치구별 공원 데이터(districts_df)가 비어있습니다.")
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
def plot_yearly_district_comparison(
    df_metric,
    metric_name_for_title,
    selected_year_str,
    bar_color='skyblue',
    custom_xlabel: str = None
):
    if df_metric.empty or selected_year_str not in df_metric.columns:
        # st.info 메시지는 run_park_analysis_page 함수에서 처리
        return

    data_for_year = df_metric[[selected_year_str]].sort_values(by=selected_year_str, ascending=False).reset_index()
    if data_for_year.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 12))
    
    final_xlabel = custom_xlabel 

    sns.barplot(x=selected_year_str, y='자치구', data=data_for_year, color=bar_color, ax=ax, label=final_xlabel)
    
    # 그래프 제목을 metric_name_for_title로 직접 설정 (피드백 반영)
    ax.set_title(f'서울시 자치구별 {metric_name_for_title}', fontsize=15) # "서울시" 추가
    ax.set_xlabel(final_xlabel, fontsize=12)
    ax.set_ylabel('자치구', fontsize=12)
    ax.tick_params(axis='x', labelsize=10); ax.tick_params(axis='y', labelsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.legend(fontsize=10)
    plt.tight_layout(); st.pyplot(fig)

def plot_seoul_total_distribution(df_seoul_total, selected_year_str):
    if df_seoul_total is None or df_seoul_total.empty:
        st.info(f"데이터가 없어 서울시 전체 공원 유형별 면적 파이 차트를 생성할 수 없습니다.")
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
        st.info(f"추출된 공원 유형별 면적 데이터가 없습니다.")
        return

    park_types_series = pd.Series(park_type_areas).sort_values(ascending=False)
    
    threshold_ratio = 2.0 
    total_area = park_types_series.sum()
    if total_area == 0:
        st.info(f"공원 면적 데이터가 모두 0입니다.")
        return
        
    small_slices = park_types_series[park_types_series / total_area * 100 < threshold_ratio]
    
    if not small_slices.empty and len(park_types_series) > len(small_slices) and len(small_slices) > 1 :
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

    fig, ax = plt.subplots(figsize=(13, 10))

    target_labels_for_larger_font = ["근린공원", "기타", "도시자연공원구역"]
    
    def func_autopct(pct):
        return f'{pct:.1f}%' if pct >= 2.5 else ''

    pie_labels_outer = [label if label not in target_labels_for_larger_font else '' for label in park_types_series_to_plot.index]
    
    # autopct 함수를 사용하여 내부 텍스트 포맷팅
    def autopct_format_custom(pct_val):
        # autopct는 원래 값(value)이 아닌 퍼센티지(pct_val)를 받음
        # 해당 퍼센티지에 맞는 레이블을 찾기 위해 park_types_series_to_plot와 비교
        # 이 방식은 완벽하지 않을 수 있음 (동일 퍼센티지 값 존재 시)
        # 더 나은 방법은 pie() 반환값인 autotexts를 직접 수정하는 것
        current_label = ""
        # 근사값으로 레이블 매칭 시도
        for idx, val_series in enumerate(park_types_series_to_plot):
            if np.isclose((val_series / total_area * 100), pct_val):
                current_label = park_types_series_to_plot.index[idx]
                break
        
        if current_label in target_labels_for_larger_font:
            return f"{current_label}\n{pct_val:.1f}%"
        elif pct_val >= 2.5:
            return f"{pct_val:.1f}%"
        return ""

    wedges, texts, autotexts = ax.pie(
        park_types_series_to_plot,
        labels=pie_labels_outer,
        autopct=autopct_format_custom, # 수정된 autopct 사용
        startangle=140,
        pctdistance=0.65, 
        labeldistance=1.05,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
        explode=explode,
        textprops={'fontsize': 9} # 외부 레이블 기본 폰트 크기
    )

    # autotexts (내부 텍스트) 후처리로 폰트 크기 조정
    for i, autotext_obj in enumerate(autotexts):
        # autotext_obj의 텍스트 내용을 분석하여 target_label인지 확인
        text_content = autotext_obj.get_text()
        is_target = any(target_label in text_content for target_label in target_labels_for_larger_font)
        
        if is_target:
            autotext_obj.set_fontsize(11) # 대상 항목 폰트 키우기
            autotext_obj.set_color('black')
        else: # func_autopct에서 이미 작은 값은 "" 처리됨
            if text_content: # 빈 문자열이 아니면 (즉, 표시되는 비율이면)
                 autotext_obj.set_fontsize(8)
                 autotext_obj.set_color('black')
            
    for i, text_obj in enumerate(texts): 
        if text_obj.get_text(): # 외부 레이블이 있는 경우
             text_obj.set_fontsize(9)

    ax.set_title('서울시 도시공원 유형별 면적 분포', fontsize=16) # (selected_year_str) 제거
    ax.axis('equal')
    if num_slices > 0 : # 데이터가 있을 때만 범례 표시
        ax.legend(wedges, park_types_series_to_plot.index, title="공원 유형", loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=9)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    st.pyplot(fig)


def create_choropleth_map(df_metric, geo_data, year_str, metric_name, unit, fill_color_map='Blues'):
    if df_metric.empty or year_str not in df_metric.columns:
        return None

    data_to_map = df_metric[[year_str]].copy()
    if data_to_map.index.name != '자치구':
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
                legend_name=f'{metric_name} ({unit})', # 연도 정보 제거
                highlight=True,
                name=f'{metric_name}' # 연도 정보 제거
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
                    if hasattr(geom, 'representative_point'):
                        center_point = geom.representative_point()
                        center_lon, center_lat = center_point.x, center_point.y
                    else:
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
    st.title("서울시 공원 분석") # NameError 발생 지점은 아니지만, 임포트 문제일 가능성 있음
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
        key="park_year_slider_main_page_v5" # 유니크한 키로 변경
    )
    st.session_state.selected_year_park = selected_year_int
    selected_year_str = str(selected_year_int)

    df_raw_parks = load_csv("data/공원_20250525010638.csv", header_config=[0,1,2,3,4], na_values_config='-')
    if df_raw_parks is None:
        st.error("공원 데이터를 로드하지 못했습니다. 'data/공원_20250525010638.csv' 파일을 확인해주세요.")
        return

    districts_df_parks, df_seoul_total_parks = preprocess_park_data_cached(df_raw_parks)
    if districts_df_parks is None: # districts_df_parks.empty 뿐만 아니라 None 체크
        st.error("공원 데이터 전처리 중 오류가 발생했거나, 유효한 자치구 데이터가 없습니다.")
        return

    df_park_counts, df_park_area = extract_total_park_stats_cached(districts_df_parks, park_years_str)
    if df_park_counts is None or df_park_area is None: # None 체크 추가
        st.error("공원 통계 추출 중 오류가 발생했습니다.")
        return


    geojson_url = 'https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json'
    @st.cache_data
    def get_geojson_cached_park_page(url): return load_geojson(url)
    seoul_geo_data_parks = get_geojson_cached_park_page(geojson_url)

    tab1, tab2, tab3 = st.tabs([
        "연도별 자치구 현황",
        "서울시 전체 현황",
        "지도 시각화"
    ])

    with tab1:
        st.subheader(f"{selected_year_str}년 연도별 자치구 현황")

        if not df_park_area.empty:
            plot_yearly_district_comparison(
                df_park_area,
                metric_name_for_title="총 공원 면적", # 제목 수정
                selected_year_str=selected_year_str,
                bar_color='cornflowerblue',
                custom_xlabel="공원 면적 (천㎡)"
            )
        else: st.info(f"{selected_year_str}년 자치구별 공원 면적 데이터가 없습니다.")
        # 구분선 제거됨

        if not df_park_counts.empty:
            plot_yearly_district_comparison(
                df_park_counts,
                metric_name_for_title="총 공원 수", # 제목 수정
                selected_year_str=selected_year_str,
                bar_color='mediumseagreen',
                custom_xlabel="공원 수 (천㎡)" # 단위는 개소이지만, 요청대로 (천㎡)
            )
        else: st.info(f"{selected_year_str}년 자치구별 공원 수 데이터가 없습니다.")
        # 구분선 제거됨

        if not df_park_counts.empty and not df_park_area.empty:
            df_park_counts_for_avg = df_park_counts.replace(0, np.nan)
            if not df_park_counts_for_avg.empty and selected_year_str in df_park_counts_for_avg.columns and \
               not df_park_counts_for_avg[selected_year_str].isnull().all():
                
                current_year_area = df_park_area.get(selected_year_str, pd.Series(dtype='float64'))
                current_year_counts_for_avg = df_park_counts_for_avg.get(selected_year_str, pd.Series(dtype='float64'))

                if current_year_area.sum(skipna=True) > 0 and \
                   current_year_counts_for_avg[current_year_counts_for_avg > 0].sum(skipna=True) > 0 : # skipna=True 추가
                     df_avg_park_area = df_park_area.div(df_park_counts_for_avg).fillna(0)
                     plot_yearly_district_comparison(
                         df_avg_park_area,
                         metric_name_for_title="1개소당 평균 공원 면적", # "공원" 추가
                         selected_year_str=selected_year_str,
                         bar_color='lightcoral',
                         custom_xlabel="공원 평균 면적 (천㎡)"
                     )
                else:
                    st.info(f"{selected_year_str}년 공원 면적 또는 공원 수가 없어 1개소당 평균 면적을 계산할 수 없습니다.")
            else: st.info(f"{selected_year_str}년 공원 수가 모두 0이거나 데이터가 없어 1개소당 평균 면적을 계산할 수 없습니다.")
        else: st.info(f"{selected_year_str}년 자치구별 공원 1개소당 평균 면적을 계산할 데이터가 부족합니다.")

    with tab2:
        st.subheader(f"{selected_year_str}년 서울시 전체 현황") # 요청사항 3
        if df_seoul_total_parks is not None and not df_seoul_total_parks.empty:
            plot_seoul_total_distribution(df_seoul_total_parks, selected_year_str) # 함수 내에서 제목 수정됨
        else:
            st.info(f"{selected_year_str}년 서울시 전체 공원 데이터를 찾을 수 없습니다.")


    with tab3:
        st.subheader(f"{selected_year_str}년도 지도 시각화") # 요청사항 5
        map_metric_parks = st.selectbox("지도 표시 항목 선택:", ["총 공원 수", "총 공원 면적", "1개소당 평균 면적"], key="park_map_metric_sb_main_tab_v4") # 키 변경

        if seoul_geo_data_parks:
            park_map_to_display = None
            if map_metric_parks == "총 공원 수":
                if not df_park_counts.empty:
                    park_map_to_display = create_choropleth_map(df_park_counts, seoul_geo_data_parks, selected_year_str, "총 공원 수", "개소", "Greens")
                else: st.info(f"{selected_year_str}년 총 공원 수 데이터가 없어 지도를 생성할 수 없습니다.")

            elif map_metric_parks == "총 공원 면적":
                if not df_park_area.empty:
                    park_map_to_display = create_choropleth_map(df_park_area, seoul_geo_data_parks, selected_year_str, "총 공원 면적", "천㎡", "Blues")
                else: st.info(f"{selected_year_str}년 총 공원 면적 데이터가 없어 지도를 생성할 수 없습니다.")

            elif map_metric_parks == "1개소당 평균 면적":
                if not df_park_counts.empty and not df_park_area.empty:
                    df_park_counts_for_avg_map = df_park_counts.replace(0, np.nan) 
                    if selected_year_str in df_park_counts_for_avg_map and not df_park_counts_for_avg_map[selected_year_str].isnull().all():
                        current_year_area_map = df_park_area.get(selected_year_str, pd.Series(dtype='float64'))
                        current_year_counts_map = df_park_counts_for_avg_map.get(selected_year_str, pd.Series(dtype='float64'))
                        if current_year_area_map.sum(skipna=True) > 0 and current_year_counts_map[current_year_counts_map > 0].sum(skipna=True) > 0 :
                            df_avg_park_area_map = df_park_area.div(df_park_counts_for_avg_map).fillna(0)
                            park_map_to_display = create_choropleth_map(df_avg_park_area_map, seoul_geo_data_parks, selected_year_str, "1개소당 평균 공원 면적", "천㎡/개소", "Oranges")
                        else:
                            st.info(f"{selected_year_str}년 1개소당 평균 면적을 계산할 데이터(공원 면적 또는 공원 수)가 없어 지도를 생성할 수 없습니다.")
                    else:
                        st.info(f"{selected_year_str}년 공원 수 데이터가 없거나 모두 유효하지 않아 1개소당 평균 면적을 계산할 수 없습니다.")
                else:
                    st.info(f"{selected_year_str}년 1개소당 평균 면적을 계산하기 위한 데이터가 부족합니다.")

            if park_map_to_display:
                st_folium(park_map_to_display, width=800, height=600)
            elif not ( (map_metric_parks == "총 공원 수" and (df_park_counts is None or df_park_counts.empty)) or \
                         (map_metric_parks == "총 공원 면적" and (df_park_area is None or df_park_area.empty)) or \
                         (map_metric_parks == "1개소당 평균 면적" and (df_park_counts is None or df_park_counts.empty or df_park_area is None or df_park_area.empty)) ) :
                 # 위 조건은 지도를 만들 수 없는 명확한 데이터 부재 상황을 제외하고, 그 외의 이유로 park_map_to_display가 None일 때 메시지 표시
                 if park_map_to_display is None: # 데이터는 있으나 지도 생성 실패 시 (예: GeoJSON 문제)
                     pass # create_choropleth_map 내부에서 이미 st.error/warning 처리
                 # else: # 데이터 자체가 없어 위에서 st.info로 이미 알린 경우
                 #    st.info("선택한 항목에 대한 지도를 표시할 데이터가 없습니다.") # 중복 메시지 방지
        else:
            st.warning("GeoJSON 데이터가 로드되지 않아 지도를 표시할 수 없습니다.")

if __name__ == "__main__":
    run_park_analysis_page()
# --- END OF 3_ParkAnalysis.py ---
