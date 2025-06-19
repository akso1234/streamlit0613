# --- START OF 4_ElderlyPopulation.py (전체 코드) ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from utils import set_korean_font, load_csv, load_geojson
from shapely.geometry import shape, Polygon, MultiPolygon
import os
from matplotlib.ticker import FuncFormatter, PercentFormatter

# --- 데이터 전처리 및 추출 함수 (이전과 동일) ---
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
def preprocess_goryeong_data_cached(df_raw):
    if df_raw is None: return pd.DataFrame(), pd.Series(dtype='float64')
    df = df_raw.copy()
    try:
        if isinstance(df.columns, pd.MultiIndex) and len(df.columns.levels) >= 2:
            idx_col_name_tuple1 = df.columns[0]
            idx_col_name_tuple2 = df.columns[1]
            df = df.set_index([idx_col_name_tuple1, idx_col_name_tuple2])
            df.index.names = ['구분_대', '구분_소']
        elif len(df.columns) >=2:
            df = df.set_index([df.columns[0], df.columns[1]])
            df.index.names = ['구분_대', '구분_소']
        else:
            st.error("고령자현황 데이터의 인덱스 컬럼 구조가 예상과 다릅니다.")
            return pd.DataFrame(), pd.Series(dtype='float64')
    except Exception as e: st.error(f"고령자현황 데이터 인덱스 설정 중 오류: {e}"); return pd.DataFrame(), pd.Series(dtype='float64')
    for col_tuple in df.columns:
        if df[col_tuple].dtype == 'object':
            df[col_tuple] = df[col_tuple].astype(str).str.replace(',', '', regex=False)
        df[col_tuple] = pd.to_numeric(df[col_tuple], errors='coerce')
    df = df.fillna(0).astype(int)
    seoul_total_data = pd.Series(dtype='float64')
    if ('합계', '소계') in df.index: 
        seoul_total_data = df.loc[('합계', '소계')]
    df_districts = pd.DataFrame()
    if '합계' in df.index.get_level_values(0):
        df_districts_temp = df.loc['합계']
        if '소계' in df_districts_temp.index:
             df_districts = df_districts_temp[df_districts_temp.index != '소계']
        else:
            df_districts = df_districts_temp
    return df_districts, seoul_total_data

# --- 시각화 함수 ---
def plot_seoul_population_trends(seoul_total_goryeong_data, goryeong_years_str_list):
    if seoul_total_goryeong_data.empty: st.info("서울시 전체 인구 추이 데이터를 그릴 수 없습니다."); return
    seoul_total_pop_trend = [seoul_total_goryeong_data.get((year, '전체인구', '소계', '소계'), 0) for year in goryeong_years_str_list]
    seoul_elderly_pop_trend = [seoul_total_goryeong_data.get((year, '65세이상 인구', '소계', '소계'), 0) for year in goryeong_years_str_list]
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(goryeong_years_str_list, seoul_total_pop_trend, marker='o', linestyle='-', label='서울시 전체 인구')
    ax1.plot(goryeong_years_str_list, seoul_elderly_pop_trend, marker='s', linestyle='--', label='서울시 65세 이상 인구')
    ax1.set_title('서울시 전체 인구 및 65세 이상 인구 변화', fontsize=15)
    ax1.set_xlabel('연도'); ax1.set_ylabel('인구수 (명)'); ax1.legend(); ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.tight_layout(); st.pyplot(fig1)

    seoul_total_pop_np = np.array(seoul_total_pop_trend, dtype=float); seoul_elderly_pop_np = np.array(seoul_elderly_pop_trend, dtype=float)
    seoul_elderly_ratio_trend = np.zeros_like(seoul_total_pop_np); mask = seoul_total_pop_np > 0
    seoul_elderly_ratio_trend[mask] = (seoul_elderly_pop_np[mask] / seoul_total_pop_np[mask]) * 100
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(goryeong_years_str_list, seoul_elderly_ratio_trend, marker='o', linestyle='-', label='서울시 고령화율')
    ax2.set_title('서울시 고령화율 변화', fontsize=15)
    ax2.set_xlabel('연도'); ax2.set_ylabel('고령화율 (%)')
    if len(seoul_elderly_ratio_trend) > 0 and np.nanmax(seoul_elderly_ratio_trend) > 0 : ax2.set_ylim(max(0, np.nanmin(seoul_elderly_ratio_trend) -1) , np.nanmax(seoul_elderly_ratio_trend) + 1)
    ax2.legend(); ax2.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout(); st.pyplot(fig2)

def plot_elderly_sex_ratio_pie_yearly(seoul_total_goryeong_data, selected_goryeong_year_str):
    if seoul_total_goryeong_data.empty: st.info("노인 성별분포 데이터를 그릴 수 없습니다."); return
    try:
        male_pop = seoul_total_goryeong_data[(selected_goryeong_year_str, '65세이상 인구', '남자', '소계')]
        female_pop = seoul_total_goryeong_data[(selected_goryeong_year_str, '65세이상 인구', '여자', '소계')]
    except KeyError: st.warning(f"노인 성별 데이터 컬럼을 고령자현황 데이터에서 찾을 수 없습니다."); return
    fig, ax = plt.subplots(figsize=(7, 7))
    pie_labels_for_legend = ['남자', '여자']
    pie_data = [male_pop, female_pop]
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return f'{pct:.1f}%\n({val:,}명)' # 그래프 안에는 비율과 명수 표시
        return my_autopct
    wedges, texts, autotexts = ax.pie(pie_data, explode=(0, 0.05), 
            autopct=make_autopct(pie_data), 
            shadow=True, startangle=140, textprops={'fontsize': 9, 'color':'black'})
    ax.set_title('서울시 65세 이상 인구 성별 분포', fontsize=15); ax.axis('equal')
    ax.legend(wedges, pie_labels_for_legend, title="성별", loc="best", fontsize=10) # 범례에는 '남자', '여자'만
    st.pyplot(fig)

def plot_district_elderly_ratio_yearly(df_goryeong_districts, selected_goryeong_year_str):
    if df_goryeong_districts.empty: st.info("구별 고령화율 데이터를 그릴 수 없습니다."); return
    try:
        total_pop = df_goryeong_districts[(selected_goryeong_year_str, '전체인구', '소계', '소계')]
        elderly_pop = df_goryeong_districts[(selected_goryeong_year_str, '65세이상 인구', '소계', '소계')]
    except KeyError: st.warning(f"인구 데이터 컬럼을 고령자현황 데이터에서 찾을 수 없습니다."); return
    ratio = (elderly_pop / total_pop.replace(0, np.nan)) * 100; ratio_sorted = ratio.dropna().sort_values(ascending=False)
    if ratio_sorted.empty: st.info("구별 고령화율 계산 결과가 없습니다."); return
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.barplot(x=ratio_sorted.index.get_level_values('구분_소'), y=ratio_sorted.values, ax=ax, label='고령화율')
    ax.set_title('서울시 자치구별 고령화율', fontsize=15)
    ax.set_xlabel('자치구'); ax.set_ylabel('고령화율 (%)'); plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    if not ratio_sorted.empty: ax.set_ylim(0, ratio_sorted.max() * 1.1 if ratio_sorted.max() > 0 else 10)
    ax.legend(fontsize=10) # 범례 추가
    plt.tight_layout(); st.pyplot(fig)

def plot_district_elderly_ratio_change(df_districts, start_year_str, end_year_str):
    if df_districts.empty:
        st.info("자치구별 고령화율 변화량 데이터를 그릴 수 없습니다.")
        return

    start_total_col = (start_year_str, '전체인구', '소계', '소계')
    start_elderly_col = (start_year_str, '65세이상 인구', '소계', '소계')
    end_total_col = (end_year_str, '전체인구', '소계', '소계')
    end_elderly_col = (end_year_str, '65세이상 인구', '소계', '소계')

    if not all(col in df_districts.columns for col in [start_total_col, start_elderly_col, end_total_col, end_elderly_col]):
        st.warning("고령화율 변화량 계산에 필요한 연도별 인구 데이터 컬럼이 부족합니다.")
        return

    dist_total_pop_start = df_districts[start_total_col]
    dist_elderly_pop_start = df_districts[start_elderly_col]
    dist_elderly_ratio_start = (dist_elderly_pop_start / dist_total_pop_start.replace(0, np.nan)) * 100

    dist_total_pop_end = df_districts[end_total_col]
    dist_elderly_pop_end = df_districts[end_elderly_col]
    dist_elderly_ratio_end = (dist_elderly_pop_end / dist_total_pop_end.replace(0, np.nan)) * 100

    ratio_change_pp = dist_elderly_ratio_end - dist_elderly_ratio_start.reindex(dist_elderly_ratio_end.index)
    ratio_change_pp_sorted = ratio_change_pp.dropna().sort_values(ascending=False)

    if ratio_change_pp_sorted.empty:
        st.info("고령화율 변화량 계산 결과가 없습니다.")
        return

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.barplot(x=ratio_change_pp_sorted.index.get_level_values('구분_소'), 
                y=ratio_change_pp_sorted.values, ax=ax,
                label=f'{start_year_str}년 대비 {end_year_str}년 변화량 (p.p.)') # 범례 레이블
    ax.set_title(f'서울시 자치구별 고령화율 변화량 ({start_year_str}년 대비 {end_year_str}년)', fontsize=16)
    ax.set_xlabel('자치구', fontsize=12)
    ax.set_ylabel('고령화율 변화 (p.p.)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(fontsize=10) # 범례 표시
    plt.tight_layout()
    st.pyplot(fig)


def plot_dokgo_by_gu_yearly(df_gu_dokgo, selected_year):
    if df_gu_dokgo.empty or selected_year not in df_gu_dokgo.columns:
        st.info("구별 독거노인 데이터를 그릴 수 없습니다."); return
    df_gu_sorted = df_gu_dokgo.sort_values(by=selected_year, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(df_gu_sorted.index, df_gu_sorted[selected_year], label=f'독거노인 수')
    ax.set_title(f'서울시 자치구별 독거노인 수', fontsize=15)
    ax.set_xlabel('독거노인 수 (명)'); ax.set_ylabel('자치구')
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.legend(fontsize=10, loc="lower right") # 범례 추가
    plt.tight_layout(); st.pyplot(fig)

def plot_dokgo_vs_total_elderly_ratio_gu_yearly(df_dokgo_gu, df_goryeong_districts, selected_year_dokgo_format):
    if df_dokgo_gu.empty or df_goryeong_districts.empty: st.info("구별 전체 노인 대비 독거노인 비율을 계산할 데이터가 부족합니다."); return
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
    if ratio_gu_sorted.empty: st.info("구별 독거노인 비율 계산 결과가 없습니다."); return
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.barplot(x=ratio_gu_sorted.index, y=ratio_gu_sorted.values, ax=ax, label='65세 이상 인구 중 독거노인 비율') # color 지정 제거
    ax.set_title('서울시 자치구별 65세 이상 인구 중 독거노인 비율', fontsize=15)
    ax.set_xlabel('자치구'); ax.set_ylabel('독거노인 비율 (%)'); plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    if not ratio_gu_sorted.empty: ax.set_ylim(0, ratio_gu_sorted.max() * 1.15 if ratio_gu_sorted.max() > 0 else 10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    ax.legend(fontsize=10) # 범례 추가
    plt.tight_layout(); st.pyplot(fig)

def plot_seoul_total_dokgo_trend(df_seoul_total, df_seoul_male, df_seoul_female, year_data_cols):
    if df_seoul_male.empty or df_seoul_female.empty or not year_data_cols:
        st.info("서울시 전체 독거노인(성별) 추이 데이터를 그릴 수 없습니다."); return
    fig, ax = plt.subplots(figsize=(12, 6))
    if not df_seoul_male.empty and all(col in df_seoul_male.columns for col in year_data_cols):
        ax.plot(year_data_cols, df_seoul_male[year_data_cols].iloc[0], marker='o', linestyle='-', label='남성')
    if not df_seoul_female.empty and all(col in df_seoul_female.columns for col in year_data_cols):
        ax.plot(year_data_cols, df_seoul_female[year_data_cols].iloc[0], marker='s', linestyle='--', label='여성')
    if not df_seoul_total.empty and all(col in df_seoul_total.columns for col in year_data_cols):
        ax.plot(year_data_cols, df_seoul_total[year_data_cols].iloc[0], marker='^', linestyle=':', label='전체 (계)')
    ax.set_title('서울시 전체 연도별 독거노인 수 변화 (성별 구분)', fontsize=16)
    ax.set_xlabel('연도', fontsize=12); ax.set_ylabel('독거노인 수 (명)', fontsize=12)
    ax.set_xticks(year_data_cols); ax.set_xticklabels(year_data_cols)
    ax.legend(title='성별', fontsize=10); ax.grid(True, linestyle=':', alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.tight_layout(); st.pyplot(fig)

def plot_top_gu_dokgo_trend(df_gu_dokgo, year_data_cols, N=10):
    if df_gu_dokgo.empty or not year_data_cols:
        st.info("상위 자치구 독거노인 추이 데이터를 그릴 수 없습니다."); return
    latest_year = year_data_cols[-1] if year_data_cols else None
    if not latest_year or latest_year not in df_gu_dokgo.columns:
        st.info(f"최신 연도 데이터가 없어 상위 자치구를 선택할 수 없습니다."); return
    df_gu_sorted_latest = df_gu_dokgo.sort_values(by=latest_year, ascending=False)
    top_n_gu_names = df_gu_sorted_latest.head(N).index.tolist()
    if not top_n_gu_names: st.info("상위 자치구를 찾을 수 없습니다."); return
    df_top_trends = df_gu_dokgo.loc[top_n_gu_names, year_data_cols]
    fig, ax = plt.subplots(figsize=(14, 7))
    for gu_name in top_n_gu_names:
        if gu_name in df_top_trends.index:
            ax.plot(year_data_cols, df_top_trends.loc[gu_name, year_data_cols], marker='o', label=gu_name)
    ax.set_title(f'서울시 상위 {N}개 자치구 연도별 독거노인 변화', fontsize=15) 
    ax.set_xlabel('연도'); ax.set_ylabel('독거노인 수 (명)')
    ax.legend(title='자치구 이름', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True); ax.set_xticks(year_data_cols); ax.set_xticklabels(year_data_cols)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.tight_layout(rect=[0, 0, 0.85, 1]); st.pyplot(fig)

def create_dokgo_map_yearly(df_gu_dokgo, selected_year, geo_data):
    # ... (이전 코드 내용 유지) ...
    if df_gu_dokgo.empty or selected_year not in df_gu_dokgo.columns or not geo_data: return None
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=10.5, tiles='CartoDB positron')
    choropleth_data_series = df_gu_dokgo[selected_year]
    choropleth_df = choropleth_data_series.reset_index(); choropleth_df.columns = ['자치구', '독거노인수']
    folium.Choropleth(
        geo_data=geo_data, name='독거노인 수', data=choropleth_df,
        columns=['자치구', '독거노인수'], key_on='feature.properties.name',
        fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.3,
        legend_name='독거노인 수', highlight=True, tooltip=None
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
                icon=folium.DivIcon(html=f'<div style="font-size: 9pt; font-weight: bold; color: black; background-color: transparent; white-space: nowrap;">{gu_name_geojson}</div>')
            ).add_to(m)
    except Exception as e: st.warning(f"지도 라벨 생성 중 오류: {e}")
    return m

def plot_district_elderly_population_latest(df_districts, latest_year_str='2023'):
    if df_districts.empty:
        st.info(f"자치구별 65세 이상 인구 수 데이터를 그릴 수 없습니다.")
        return
    col_key = (latest_year_str, '65세이상 인구', '소계', '소계')
    if col_key not in df_districts.columns:
        st.warning(f"65세 이상 인구 데이터 컬럼('{col_key}')을 찾을 수 없습니다.")
        return
    elderly_pop_by_district = df_districts[col_key].sort_values(ascending=False)
    if elderly_pop_by_district.empty:
        st.info(f"자치구별 65세 이상 인구 수 데이터가 비어있습니다.")
        return
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.barplot(x=elderly_pop_by_district.index.get_level_values('구분_소'), 
                y=elderly_pop_by_district.values, ax=ax, label='65세 이상 인구 수') # 범례 추가
    ax.set_title(f'서울시 자치구별 65세 이상 인구 수', fontsize=16)
    ax.set_xlabel('자치구', fontsize=12)
    ax.set_ylabel('65세 이상 인구 수 (명)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.legend(fontsize=10) # 범례 표시
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

def plot_seoul_total_dokgo_ratio_pie(seoul_total_data, df_dokgo_seoul_total, year_str='2023'):
    if seoul_total_data.empty or df_dokgo_seoul_total.empty:
        st.info(f"서울시 전체 독거노인 비율 데이터를 표시할 수 없습니다.") # 연도 정보 제거
        return
    try:
        total_elderly_col = (year_str, '65세이상 인구', '소계', '소계')
        if total_elderly_col not in seoul_total_data.index:
            st.warning(f"서울시 전체 65세 이상 인구 데이터 ('{total_elderly_col}')를 seoul_total_data에서 찾을 수 없습니다.") # 연도 정보 제거
            return
        seoul_total_65plus_pop = seoul_total_data[total_elderly_col]

        dokgo_col_name_for_pie = f"{year_str}년"
        if dokgo_col_name_for_pie not in df_dokgo_seoul_total.columns:
            st.warning(f"독거노인 수 데이터 컬럼을 찾을 수 없습니다.") # 연도 정보 제거
            return
        
        seoul_total_dokgeo_pop = df_dokgo_seoul_total[dokgo_col_name_for_pie].iloc[0]
        other_elderly_pop = seoul_total_65plus_pop - seoul_total_dokgeo_pop

        if seoul_total_65plus_pop == 0:
            st.info(f"서울시 전체 65세 이상 인구가 0명입니다. 비율을 계산할 수 없습니다.") # 연도 정보 제거
            return
        if other_elderly_pop < 0:
            st.warning(f"경고: 데이터에서 독거노인 수({seoul_total_dokgeo_pop:,.0f})가 전체 65세 이상 인구({seoul_total_65plus_pop:,.0f})보다 많습니다. 데이터를 확인해주세요.") # 연도 정보 제거
            other_elderly_pop = 0

        labels_for_pie = [f'독거노인 ({seoul_total_dokgeo_pop:,.0f}명)',
                          f'기타 65세 이상 노인 ({other_elderly_pop:,.0f}명)']
        sizes = [seoul_total_dokgeo_pop, other_elderly_pop]
        explode = (0.05, 0)

        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=None,
                autopct='%1.1f%%', shadow=True, startangle=140,
                textprops={'fontsize': 12, 'color': 'black', 'weight': 'bold'})
        ax.set_title(f'서울시 전체 65세 이상 인구 중 독거노인 비율\n(총 65세 이상: {seoul_total_65plus_pop:,.0f}명)', fontsize=15) # 연도 정보 제거
        ax.axis('equal')
        ax.legend(wedges, labels_for_pie, title="구분", loc="best", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.tight_layout(rect=[0,0,0.75,1])
        st.pyplot(fig)
    except KeyError as e:
        st.error(f"데이터 처리 중 필요한 컬럼을 찾을 수 없습니다: {e}") # 연도 정보 제거
    except Exception as e:
        st.error(f"서울시 전체 독거노인 비율 그래프 생성 중 오류: {e}") # 연도 정보 제거


# --- Streamlit 페이지 레이아웃 ---
def run_elderly_population_page():
    st.title("노인 인구 및 독거노인 현황")
    set_korean_font()

    available_years_int = [2019, 2020, 2021, 2022, 2023]
    available_years_str = [str(y) for y in available_years_int] 
    available_years_str_dokgo = [f"{y}년" for y in available_years_int] 

    if "selected_year_elderly_tab3" not in st.session_state:
        st.session_state.selected_year_elderly_tab3 = available_years_int[-1]

    df_dokgo_raw_s1923 = load_csv("data/Seoul1923.csv")
    elderly_population_file_path = 'data/elderly_status_20250531210628.csv'
    if not os.path.exists(elderly_population_file_path):
        original_elderly_file_name = 'data/고령자현황_20250531210628.csv'
        if os.path.exists(original_elderly_file_name):
            st.warning(f"'{os.path.basename(elderly_population_file_path)}'를 찾지 못해 '{os.path.basename(original_elderly_file_name)}'로 대체합니다. 파일명을 'elderly_status_20250531210628.csv'로 변경하는 것을 권장합니다.")
            elderly_population_file_path = original_elderly_file_name
        else:
            st.error(f"고령자 현황 파일을 다음 경로들에서 찾을 수 없습니다: '{elderly_population_file_path}', '{original_elderly_file_name}'. 파일명을 확인해주세요.")
            return
    df_goryeong_raw_page = load_csv(elderly_population_file_path, header_config=[0,1,2,3]) 
    
    geojson_url = 'https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json'
    @st.cache_data
    def get_geojson_data_cached_elderly_page(url): return load_geojson(url)
    seoul_geo_data_elderly = get_geojson_data_cached_elderly_page(geojson_url)

    if df_dokgo_raw_s1923 is None or df_goryeong_raw_page is None:
        st.error("필수 데이터 파일(Seoul1923.csv 또는 고령자현황 파일)을 로드하지 못했습니다."); return

    df_dokgo_processed_s1923, year_cols_dokgo_s1923_from_data = preprocess_dokgo_data_original_cached(df_dokgo_raw_s1923)
    df_seoul_total_s1923, df_seoul_male_s1923, df_seoul_female_s1923, df_gu_dokgo_s1923 = filter_dokgo_data_cached(df_dokgo_processed_s1923, year_cols_dokgo_s1923_from_data if year_cols_dokgo_s1923_from_data else [f"{y}년" for y in available_years_int])
    df_goryeong_districts_page, seoul_total_goryeong_data_page = preprocess_goryeong_data_cached(df_goryeong_raw_page)

    # 탭 제목 수정 및 새 탭 추가
    tab_new_population, main_tab1, main_tab2, main_tab3 = st.tabs([
        "2023년 자치구별 65세 이상 인구", # 새 탭
        "서울시 전체 고령화 추세",
        "서울시 전체 독거노인 추세",
        "자치구별 현황 비교"
    ])

    with tab_new_population:
        st.subheader("2023년 서울시 자치구별 65세 이상 인구 수")
        if not df_goryeong_districts_page.empty:
            plot_district_elderly_population_latest(df_goryeong_districts_page, '2023')
        else:
            st.warning("자치구별 고령자 현황 데이터가 없어 그래프를 표시할 수 없습니다.")

    with main_tab1:
        st.subheader("서울시 전체 고령화 추세")
        if not seoul_total_goryeong_data_page.empty:
            plot_seoul_population_trends(seoul_total_goryeong_data_page, [str(y) for y in available_years_int])
            year_for_pie_main_tab1 = str(st.session_state.selected_year_elderly_tab3)
            plot_elderly_sex_ratio_pie_yearly(seoul_total_goryeong_data_page, year_for_pie_main_tab1)
        else: st.warning("서울시 전체 고령자현황 데이터가 없어 추세를 표시할 수 없습니다.")

    with main_tab2:
        st.subheader("서울시 전체 독거노인 추세")
        plot_seoul_total_dokgo_trend(df_seoul_total_s1923, df_seoul_male_s1923, df_seoul_female_s1923, year_cols_dokgo_s1923_from_data if year_cols_dokgo_s1923_from_data else [f"{y}년" for y in available_years_int])
        plot_top_gu_dokgo_trend(df_gu_dokgo_s1923, year_cols_dokgo_s1923_from_data if year_cols_dokgo_s1923_from_data else [f"{y}년" for y in available_years_int], N=10)

    with main_tab3:
        selected_year_tab3_val = st.slider(
            "조회 연도 선택", 
            min_value=available_years_int[0],
            max_value=available_years_int[-1],
            step=1,
            value=st.session_state.selected_year_elderly_tab3, 
            key="elderly_year_slider_tab3_specific_v7"
        )
        if st.session_state.selected_year_elderly_tab3 != selected_year_tab3_val:
            st.session_state.selected_year_elderly_tab3 = selected_year_tab3_val
            st.rerun()
        
        current_selected_year_dokgo_format_tab3 = f"{selected_year_tab3_val}년"
        current_selected_year_goryeong_format_tab3 = str(selected_year_tab3_val)

        st.subheader(f"{selected_year_tab3_val}년 자치구별 현황 비교")
        
        # 탭 순서 변경 및 새 탭("고령화율 변화량", "서울시 전체 독거노인 비율 (2023년)") 추가
        sub_tab_gu1, sub_tab_gu_change, sub_tab_gu2, sub_tab_new_dokgo_ratio, sub_tab_gu3, sub_tab_gu5 = st.tabs([
            "고령화율", "고령화율 변화량", "독거노인 수", "서울시 전체 독거노인 비율 (2023년)", "노인 중 독거노인 비율", "독거노인 지도"
        ])
        with sub_tab_gu1:
            plot_district_elderly_ratio_yearly(df_goryeong_districts_page, current_selected_year_goryeong_format_tab3)
        
        with sub_tab_gu_change:
            st.markdown(f"#### 자치구별 고령화율 변화량 (2019년 대비 {selected_year_tab3_val}년)")
            if not df_goryeong_districts_page.empty:
                plot_district_elderly_ratio_change(df_goryeong_districts_page, '2019', current_selected_year_goryeong_format_tab3)
            else:
                st.warning("자치구별 고령화율 변화량 데이터를 그릴 수 없습니다.")

        with sub_tab_gu2:
            plot_dokgo_by_gu_yearly(df_gu_dokgo_s1923, current_selected_year_dokgo_format_tab3)

        with sub_tab_new_dokgo_ratio: # <<< 새로운 탭
            st.markdown(f"#### 2023년 서울시 전체 65세 이상 인구 중 독거노인 비율") # 소주제 추가
            # df_seoul_total_s1923는 성별='계'인 서울시 전체 독거노인 데이터
            # seoul_total_goryeong_data_page는 성별='계'인 서울시 전체 고령자 데이터
            if not seoul_total_goryeong_data_page.empty and not df_seoul_total_s1923.empty:
                 plot_seoul_total_dokgo_ratio_pie(seoul_total_goryeong_data_page, df_seoul_total_s1923, '2023')
            else:
                st.info("서울시 전체 독거노인 비율 그래프를 표시하기 위한 데이터가 부족합니다.")


        with sub_tab_gu3:
            plot_dokgo_vs_total_elderly_ratio_gu_yearly(df_gu_dokgo_s1923, df_goryeong_districts_page, current_selected_year_dokgo_format_tab3)
        
        with sub_tab_gu5:
            st.markdown(f"#### {selected_year_tab3_val}년 독거노인 현황 지도")
            if seoul_geo_data_elderly:
                dokgo_map_gu = create_dokgo_map_yearly(df_gu_dokgo_s1923, current_selected_year_dokgo_format_tab3, seoul_geo_data_elderly)
                if dokgo_map_gu: st_folium(dokgo_map_gu, width=800, height=600)
                else: st.info("지도를 생성할 데이터가 없거나 GeoJSON 로드에 실패했습니다.")
            else: st.warning("GeoJSON 데이터가 로드되지 않아 지도를 표시할 수 없습니다.")

if __name__ == "__main__":
    run_elderly_population_page()
# --- END OF 4_ElderlyPopulation.py ---
