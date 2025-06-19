import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from utils import set_korean_font, load_csv, load_geojson # utils.py의 함수 임포트
from shapely.geometry import shape, Polygon, MultiPolygon # 지도 라벨용
import os # 파일 경로를 위해 추가

# --- 기존 데이터 전처리 및 추출 함수 ---
@st.cache_data
def preprocess_dokgo_data_original_cached(df_raw):
    if df_raw is None: return pd.DataFrame(), []
    df = None
    if len(df_raw.columns) == 1 and isinstance(df_raw.columns[0], str) and ',' in df_raw.columns[0]:
        header_string = df_raw.columns[0]
        actual_columns = [col.strip().replace('"', '').replace(' 년', '년') for col in header_string.split(',')]
        if actual_columns and not actual_columns[-1].strip(): actual_columns = actual_columns[:-1]
        data_split = df_raw.iloc[:, 0].str.split(',', expand=True, n=len(actual_columns)-1 if actual_columns else 0)
        if actual_columns and data_split.shape[1] == len(actual_columns):
            df = data_split.copy(); df.columns = actual_columns
        else: st.error("독거노인 데이터(Seoul1923.csv)의 컬럼 분리에 실패했습니다."); return pd.DataFrame(), []
    elif len(df_raw.columns) > 1:
        df = df_raw.copy()
        df.columns = [str(col).replace('"', '').replace(' 년', '년').strip() for col in df.columns]
        if df.columns[-1] == '' or df.columns[-1].isspace(): df = df.iloc[:, :-1]
    else: st.error("독거노인 데이터(Seoul1923.csv)의 형식이 예상과 다릅니다."); return pd.DataFrame(), []
    if df is None: return pd.DataFrame(), []

    all_possible_years_dokgo = [f"{year}년" for year in range(2019, 2024)]
    year_data_cols_in_df = [col for col in all_possible_years_dokgo if col in df.columns]
    if not year_data_cols_in_df:
        year_data_cols_in_df = sorted([col for col in df.columns if isinstance(col, str) and col.endswith('년') and col[:-1].isdigit() and 2019 <= int(col[:-1]) <= 2023])
    for col in year_data_cols_in_df:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('"', '', regex=False).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    if '단위' in df.columns: df.drop(columns=['단위'], inplace=True, errors='ignore')
    categorical_cols_to_clean = ['항목', '독거노인별', '성별', '동별']
    for col_name_strip in categorical_cols_to_clean:
        if col_name_strip in df.columns:
            df[col_name_strip] = df[col_name_strip].astype(str).str.strip().str.replace('"', '', regex=False)
            if col_name_strip == '독거노인별': df[col_name_strip] = df[col_name_strip].replace('일   반', '일반', regex=False)
    return df, year_data_cols_in_df

@st.cache_data
def filter_dokgo_data_cached(df_processed, year_data_cols):
    if df_processed.empty or not year_data_cols or not all(c in df_processed.columns for c in ['동별', '독거노인별', '성별']):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    base_filter = (df_processed['독거노인별'] == '합계')
    if '항목' in df_processed.columns: base_filter &= (df_processed['항목'] == '독거노인현황(성별)')
    df_seoul_data = df_processed[base_filter & (df_processed['동별'] == '서울특별시')]
    df_seoul_total = df_seoul_data[df_seoul_data['성별'] == '계']
    df_seoul_male = df_seoul_data[df_seoul_data['성별'] == '남']
    df_seoul_female = df_seoul_data[df_seoul_data['성별'] == '여']
    df_gu_data = df_processed[base_filter & (df_processed['동별'] != '서울특별시') & (df_processed['동별'].str.endswith('구', na=False)) & (df_processed['성별'] == '계')].copy()
    if not df_gu_data.empty: df_gu_data.set_index('동별', inplace=True)
    return df_seoul_total, df_seoul_male, df_seoul_female, df_gu_data

@st.cache_data
def preprocess_goryeong_data_cached(df_raw): # 고령자현황 원본 데이터 처리
    if df_raw is None: return pd.DataFrame(), pd.Series(dtype='float64')
    df = df_raw.copy()
    try:
        idx_col_name_tuple1 = df.columns[0]; idx_col_name_tuple2 = df.columns[1]
        df = df.set_index([idx_col_name_tuple1, idx_col_name_tuple2]); df.index.names = ['구분_대', '구분_소']
    except Exception as e: st.error(f"고령자현황 데이터 인덱스 설정 중 오류: {e}"); return pd.DataFrame(), pd.Series(dtype='float64')
    for col_tuple in df.columns: df[col_tuple] = pd.to_numeric(df[col_tuple], errors='coerce')
    df = df.fillna(0).astype(int)
    seoul_total_data = pd.Series(dtype='float64')
    if ('합계', '소계') in df.index: seoul_total_data = df.loc[('합계', '소계')]
    df_districts = pd.DataFrame()
    if '합계' in df.index.get_level_values(0):
        df_districts = df.loc['합계']
        if '소계' in df_districts.index: df_districts = df_districts[df_districts.index != '소계']
    return df_districts, seoul_total_data

# --- 시각화 함수 ---
def plot_seoul_total_dokgo_trend(df_seoul_total, df_seoul_male, df_seoul_female, year_data_cols):
    if df_seoul_male.empty or df_seoul_female.empty or not year_data_cols:
        st.info("서울시 전체 독거노인(성별) 추이 데이터를 그릴 수 없습니다."); return
    fig, ax = plt.subplots(figsize=(12, 6))
    if not df_seoul_male.empty and all(col in df_seoul_male.columns for col in year_data_cols):
        ax.plot(year_data_cols, df_seoul_male[year_data_cols].iloc[0], marker='o', linestyle='-', label='남성', color='royalblue')
    if not df_seoul_female.empty and all(col in df_seoul_female.columns for col in year_data_cols):
        ax.plot(year_data_cols, df_seoul_female[year_data_cols].iloc[0], marker='s', linestyle='--', label='여성', color='tomato')
    if not df_seoul_total.empty and all(col in df_seoul_total.columns for col in year_data_cols):
        ax.plot(year_data_cols, df_seoul_total[year_data_cols].iloc[0], marker='^', linestyle=':', label='전체 (계)', color='gray')
    ax.set_title('서울시 전체 연도별 독거노인 수 변화 (성별 구분)', fontsize=16)
    ax.set_xlabel('연도', fontsize=12); ax.set_ylabel('독거노인 수 (명)', fontsize=12)
    ax.set_xticks(year_data_cols); ax.set_xticklabels(year_data_cols)
    ax.legend(title='성별', fontsize=10); ax.grid(True, linestyle=':', alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.tight_layout(); st.pyplot(fig)

def plot_dokgo_by_gu_yearly(df_gu_dokgo, selected_year):
    if df_gu_dokgo.empty or selected_year not in df_gu_dokgo.columns:
        st.info(f"{selected_year} 구별 독거노인 데이터를 그릴 수 없습니다."); return
    df_gu_sorted = df_gu_dokgo.sort_values(by=selected_year, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(df_gu_sorted.index, df_gu_sorted[selected_year], color='skyblue', label=f'{selected_year} 독거노인 수')
    ax.set_title(f'서울시 구별 독거노인 수 ({selected_year} 기준)', fontsize=15)
    ax.set_xlabel('독거노인 수 (명)'); ax.set_ylabel('구'); ax.invert_yaxis()
    max_val = df_gu_sorted[selected_year].max() if not df_gu_sorted.empty else 0
    for bar_obj in bars:
        width = bar_obj.get_width()
        offset = max_val * 0.005 if max_val > 0 else 10
        ax.text(width + offset, bar_obj.get_y() + bar_obj.get_height()/2, f'{int(width):,}', ha='left', va='center', fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.legend(fontsize=10)
    plt.tight_layout(); st.pyplot(fig)

def plot_top_gu_dokgo_trend(df_gu_dokgo, year_data_cols, N=10):
    if df_gu_dokgo.empty or not year_data_cols:
        st.info("상위 구 독거노인 추이 데이터를 그릴 수 없습니다."); return
    latest_year = year_data_cols[-1] if year_data_cols else None
    if not latest_year or latest_year not in df_gu_dokgo.columns:
        st.info(f"최신 연도({latest_year}) 데이터가 없어 상위 구를 선택할 수 없습니다."); return
    df_gu_sorted_latest = df_gu_dokgo.sort_values(by=latest_year, ascending=False)
    top_n_gu_names = df_gu_sorted_latest.head(N).index.tolist()
    if not top_n_gu_names: st.info("상위 구를 찾을 수 없습니다."); return
    df_top_trends = df_gu_dokgo.loc[top_n_gu_names, year_data_cols]
    fig, ax = plt.subplots(figsize=(14, 7))
    for gu_name in top_n_gu_names:
        if gu_name in df_top_trends.index:
            ax.plot(year_data_cols, df_top_trends.loc[gu_name, year_data_cols], marker='o', label=gu_name)
    ax.set_title(f'서울시 상위 {N}개 구 연도별 독거노인 수 변화'); ax.set_xlabel('연도'); ax.set_ylabel('독거노인 수 (명)')
    ax.legend(title='구 이름', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True); ax.set_xticks(year_data_cols); ax.set_xticklabels(year_data_cols)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.tight_layout(rect=[0, 0, 0.85, 1]); st.pyplot(fig)

def create_dokgo_map_yearly(df_gu_dokgo, selected_year, geo_data):
    if df_gu_dokgo.empty or selected_year not in df_gu_dokgo.columns or not geo_data: return None
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=10.5, tiles='CartoDB positron')
    choropleth_data_series = df_gu_dokgo[selected_year]
    choropleth_df = choropleth_data_series.reset_index(); choropleth_df.columns = ['자치구', '독거노인수']
    
    folium.Choropleth(
        geo_data=geo_data, 
        name=f'독거노인 수 ({selected_year})', 
        data=choropleth_df,
        columns=['자치구', '독거노인수'], 
        key_on='feature.properties.name',
        fill_color='YlOrRd', 
        fill_opacity=0.7, 
        line_opacity=0.3,
        legend_name=f'독거노인 수 ({selected_year})', 
        highlight=True, 
        tooltip=None  # Choropleth의 기본 툴팁 비활성화
    ).add_to(m)
    
    try:
        for feature in geo_data['features']:
            gu_name_geojson = feature['properties'].get('name'); geom = shape(feature['geometry'])
            if not gu_name_geojson: continue
            center_point = geom.representative_point() if isinstance(geom, Polygon) else (max(geom.geoms, key=lambda p: p.area).representative_point() if isinstance(geom, MultiPolygon) else None)
            if not center_point: continue
            center_lat, center_lon = center_point.y, center_point.x
            
            folium.Marker(
                location=[center_lat, center_lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 9pt; font-weight: bold; color: black; background-color: transparent; white-space: nowrap;">{gu_name_geojson}</div>'
                )
            ).add_to(m)
    except Exception as e: st.warning(f"지도 라벨 생성 중 오류: {e}")
    return m

def plot_seoul_population_trends(seoul_total_goryeong_data, goryeong_years_str_list):
    if seoul_total_goryeong_data.empty: st.info("서울시 전체 인구 추이 데이터를 그릴 수 없습니다."); return
    seoul_total_pop_trend = [seoul_total_goryeong_data.get((year, '전체인구', '소계', '소계'), 0) for year in goryeong_years_str_list]
    seoul_elderly_pop_trend = [seoul_total_goryeong_data.get((year, '65세이상 인구', '소계', '소계'), 0) for year in goryeong_years_str_list]
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(goryeong_years_str_list, seoul_total_pop_trend, marker='o', linestyle='-', label='서울시 전체 인구', color='navy')
    ax1.plot(goryeong_years_str_list, seoul_elderly_pop_trend, marker='s', linestyle='--', label='서울시 65세 이상 인구', color='darkorange')
    ax1.set_title('서울시 전체 인구 및 65세 이상 인구 변화', fontsize=15)
    ax1.set_xlabel('연도'); ax1.set_ylabel('인구수 (명)'); ax1.legend(); ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.tight_layout(); st.pyplot(fig1)

    seoul_total_pop_np = np.array(seoul_total_pop_trend, dtype=float); seoul_elderly_pop_np = np.array(seoul_elderly_pop_trend, dtype=float)
    seoul_elderly_ratio_trend = np.zeros_like(seoul_total_pop_np); mask = seoul_total_pop_np > 0
    seoul_elderly_ratio_trend[mask] = (seoul_elderly_pop_np[mask] / seoul_total_pop_np[mask]) * 100
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(goryeong_years_str_list, seoul_elderly_ratio_trend, marker='o', linestyle='-', color='green', label='서울시 고령화율')
    for i, year_str in enumerate(goryeong_years_str_list): ax2.text(year_str, seoul_elderly_ratio_trend[i] + 0.1, f'{seoul_elderly_ratio_trend[i]:.2f}%', ha='center')
    ax2.set_title('서울시 고령화율 변화', fontsize=15); ax2.set_xlabel('연도'); ax2.set_ylabel('고령화율 (%)')
    if len(seoul_elderly_ratio_trend) > 0 and np.nanmax(seoul_elderly_ratio_trend) > 0 : ax2.set_ylim(max(0, np.nanmin(seoul_elderly_ratio_trend) -1) , np.nanmax(seoul_elderly_ratio_trend) + 1)
    ax2.legend(); ax2.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout(); st.pyplot(fig2)

def plot_district_elderly_ratio_yearly(df_goryeong_districts, selected_goryeong_year_str):
    if df_goryeong_districts.empty: st.info(f"{selected_goryeong_year_str}년 구별 고령화율 데이터를 그릴 수 없습니다."); return
    try:
        total_pop = df_goryeong_districts[(selected_goryeong_year_str, '전체인구', '소계', '소계')]
        elderly_pop = df_goryeong_districts[(selected_goryeong_year_str, '65세이상 인구', '소계', '소계')]
    except KeyError: st.warning(f"{selected_goryeong_year_str}년 인구 데이터 컬럼을 고령자현황 데이터에서 찾을 수 없습니다."); return
    ratio = (elderly_pop / total_pop.replace(0, np.nan)) * 100; ratio_sorted = ratio.dropna().sort_values(ascending=False)
    if ratio_sorted.empty: st.info(f"{selected_goryeong_year_str}년 구별 고령화율 계산 결과가 없습니다."); return
    fig, ax = plt.subplots(figsize=(15, 7))
    bars = sns.barplot(x=ratio_sorted.index.get_level_values('구분_소'), y=ratio_sorted.values, color='mediumseagreen', ax=ax, label='고령화율')
    ax.set_title(f'{selected_goryeong_year_str}년 서울시 자치구별 고령화율', fontsize=15)
    ax.set_xlabel('자치구'); ax.set_ylabel('고령화율 (%)'); plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    for bar_obj in bars.patches:
        yval = bar_obj.get_height()
        if pd.notnull(yval): ax.text(bar_obj.get_x() + bar_obj.get_width()/2.0, yval + 0.1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=9)
    if not ratio_sorted.empty: ax.set_ylim(0, ratio_sorted.max() * 1.1)
    ax.legend(fontsize=10)
    plt.tight_layout(); st.pyplot(fig)

def plot_elderly_sex_ratio_pie_yearly(seoul_total_goryeong_data, selected_goryeong_year_str):
    if seoul_total_goryeong_data.empty: st.info(f"{selected_goryeong_year_str}년 노인 성별분포 데이터를 그릴 수 없습니다."); return
    try:
        male_pop = seoul_total_goryeong_data[(selected_goryeong_year_str, '65세이상 인구', '남자', '소계')]
        female_pop = seoul_total_goryeong_data[(selected_goryeong_year_str, '65세이상 인구', '여자', '소계')]
    except KeyError: st.warning(f"{selected_goryeong_year_str}년 노인 성별 데이터 컬럼을 고령자현황 데이터에서 찾을 수 없습니다."); return
    fig, ax = plt.subplots(figsize=(7, 7))
    pie_labels = [f'남자 ({male_pop:,}명)', f'여자 ({female_pop:,}명)']
    ax.pie([male_pop, female_pop], explode=(0, 0.05), labels=pie_labels,
            colors=['skyblue', 'lightcoral'], autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': 11})
    ax.set_title(f'{selected_goryeong_year_str}년 서울시 65세 이상 인구 성별 분포', fontsize=15); ax.axis('equal')
    ax.legend(pie_labels, title="성별", loc="best", fontsize=10)
    st.pyplot(fig)

def plot_dokgo_vs_total_elderly_ratio_gu_yearly(df_dokgo_gu, df_goryeong_districts, selected_year_dokgo_format):
    if df_dokgo_gu.empty or df_goryeong_districts.empty: st.info(f"{selected_year_dokgo_format} 구별 전체 노인 대비 독거노인 비율을 계산할 데이터가 부족합니다."); return
    if selected_year_dokgo_format not in df_dokgo_gu.columns: st.warning(f"독거노인 데이터에 {selected_year_dokgo_format} 컬럼이 없습니다."); return
    dokgo_count_series_gu = df_dokgo_gu[selected_year_dokgo_format]
    selected_year_goryeong_format = selected_year_dokgo_format.replace('년', '')
    try:
        total_elderly_series_gu = df_goryeong_districts[(selected_year_goryeong_format, '65세이상 인구', '소계', '소계')]
        total_elderly_series_gu.index = total_elderly_series_gu.index.get_level_values('구분_소')
    except KeyError: st.warning(f"고령자현황 데이터에서 {selected_year_goryeong_format}년 전체 65세 이상 인구 데이터를 찾을 수 없습니다."); return
    aligned_total_elderly_gu = total_elderly_series_gu.reindex(dokgo_count_series_gu.index)
    ratio_gu = (dokgo_count_series_gu / aligned_total_elderly_gu.replace(0, np.nan)) * 100
    ratio_gu_sorted = ratio_gu.dropna().sort_values(ascending=False)
    if ratio_gu_sorted.empty: st.info(f"{selected_year_dokgo_format} 구별 독거노인 비율 계산 결과가 없습니다."); return
    fig, ax = plt.subplots(figsize=(15, 7))
    bars = sns.barplot(x=ratio_gu_sorted.index, y=ratio_gu_sorted.values, color='lightcoral', ax=ax, label='독거노인 비율')
    ax.set_title(f'{selected_year_dokgo_format} 서울시 자치구별 65세 이상 인구 중 독거노인 비율', fontsize=15)
    ax.set_xlabel('자치구'); ax.set_ylabel('독거노인 비율 (%)'); plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    for bar_obj in bars.patches:
        yval = bar_obj.get_height()
        if pd.notnull(yval): ax.text(bar_obj.get_x() + bar_obj.get_width()/2.0, yval + 0.1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=9)
    if not ratio_gu_sorted.empty: ax.set_ylim(0, ratio_gu_sorted.max() * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    ax.legend(fontsize=10)
    plt.tight_layout(); st.pyplot(fig)


# --- Streamlit 페이지 레이아웃 ---
def run_elderly_population_page():
    st.title("노인 인구 및 독거노인 현황")
    set_korean_font()

    available_years_int = [2019, 2020, 2021, 2022, 2023]
    available_years_str = [str(y) for y in available_years_int] 
    available_years_str_dokgo = [f"{y}년" for y in available_years_int] 

    if "selected_year_elderly" not in st.session_state:
        st.session_state.selected_year_elderly = available_years_int[-1]

    selected_year_int = st.session_state.selected_year_elderly
    selected_year_dokgo_format = f"{selected_year_int}년"
    selected_year_goryeong_format = str(selected_year_int)

    df_dokgo_raw_s1923 = load_csv("data/Seoul1923.csv")
    df_goryeong_raw_page = load_csv("data/고령자현황_20250531210628.csv", header_config=[0,1,2,3]) 
    geojson_url = 'https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json'

    @st.cache_data
    def get_geojson_data_cached_elderly_page(url): return load_geojson(url)
    seoul_geo_data_elderly = get_geojson_data_cached_elderly_page(geojson_url)

    if df_dokgo_raw_s1923 is None or df_goryeong_raw_page is None:
        st.error("필수 데이터 파일(Seoul1923.csv 또는 고령자현황_20250531210628.csv)을 로드하지 못했습니다."); return

    df_dokgo_processed_s1923, year_cols_dokgo_s1923_from_data = preprocess_dokgo_data_original_cached(df_dokgo_raw_s1923)
    df_seoul_total_s1923, df_seoul_male_s1923, df_seoul_female_s1923, df_gu_dokgo_s1923 = filter_dokgo_data_cached(df_dokgo_processed_s1923, year_cols_dokgo_s1923_from_data if year_cols_dokgo_s1923_from_data else available_years_str_dokgo)
    df_goryeong_districts_page, seoul_total_goryeong_data_page = preprocess_goryeong_data_cached(df_goryeong_raw_page)

    main_tab1, main_tab2, main_tab3 = st.tabs([
        "서울시 전체 고령화 추세",
        "서울시 전체 독거노인 추세",
        "자치구별 현황 비교"
    ])

    with main_tab1:
        st.subheader("서울시 전체 인구 및 고령화율 변화")
        if not seoul_total_goryeong_data_page.empty:
            plot_seoul_population_trends(seoul_total_goryeong_data_page, available_years_str)
        else: st.warning("서울시 전체 고령자현황 데이터가 없어 추세를 표시할 수 없습니다.")

        st.subheader(f"{selected_year_goryeong_format}년 서울시 65세 이상 인구 성별 분포")
        if not seoul_total_goryeong_data_page.empty:
            plot_elderly_sex_ratio_pie_yearly(seoul_total_goryeong_data_page, selected_year_goryeong_format)
        else: st.warning(f"{selected_year_goryeong_format}년 서울시 전체 고령자현황 데이터가 없어 성별 분포를 표시할 수 없습니다.")

    with main_tab2:
        st.subheader("서울시 전체 독거노인 수 변화 (성별 구분)")
        plot_seoul_total_dokgo_trend(df_seoul_total_s1923, df_seoul_male_s1923, df_seoul_female_s1923, year_cols_dokgo_s1923_from_data if year_cols_dokgo_s1923_from_data else available_years_str_dokgo)

        st.subheader("서울시 상위 10개구 독거노인 수 변화")
        plot_top_gu_dokgo_trend(df_gu_dokgo_s1923, year_cols_dokgo_s1923_from_data if year_cols_dokgo_s1923_from_data else available_years_str_dokgo, N=10)

    with main_tab3:
        new_selected_year_from_slider = st.slider(
            "조회 연도 선택",
            min_value=available_years_int[0],
            max_value=available_years_int[-1],
            step=1,
            value=st.session_state.selected_year_elderly, 
            key="elderly_year_slider_main_tab3" 
        )

        if st.session_state.selected_year_elderly != new_selected_year_from_slider:
            st.session_state.selected_year_elderly = new_selected_year_from_slider
            st.rerun() 

        # "전체 대비 노인 비율" 탭 제거
        sub_tab_gu1, sub_tab_gu2, sub_tab_gu3, sub_tab_gu5 = st.tabs([
            "고령화율", "독거노인 수", "노인 중 독거노인 비율", "독거노인 지도"
        ])
        with sub_tab_gu1:
            plot_district_elderly_ratio_yearly(df_goryeong_districts_page, selected_year_goryeong_format)
        with sub_tab_gu2:
            plot_dokgo_by_gu_yearly(df_gu_dokgo_s1923, selected_year_dokgo_format)
        with sub_tab_gu3:
            plot_dokgo_vs_total_elderly_ratio_gu_yearly(df_gu_dokgo_s1923, df_goryeong_districts_page, selected_year_dokgo_format)
        # sub_tab_gu4 (전체 대비 노인 비율) 관련 코드 블록 전체 삭제
        with sub_tab_gu5:
            if seoul_geo_data_elderly:
                dokgo_map_gu = create_dokgo_map_yearly(df_gu_dokgo_s1923, selected_year_dokgo_format, seoul_geo_data_elderly)
                if dokgo_map_gu: st_folium(dokgo_map_gu, width=800, height=600)
                else: st.info(f"{selected_year_int}년 독거노인 현황 지도를 생성할 데이터가 없거나 GeoJSON 로드에 실패했습니다.")
            else: st.warning("GeoJSON 데이터가 로드되지 않아 지도를 표시할 수 없습니다.")

if __name__ == "__main__":
    run_elderly_population_page()
