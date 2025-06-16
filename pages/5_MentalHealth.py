import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import set_korean_font, load_csv # load_geojson은 여기서는 불필요
import os

# --- 기존 데이터 전처리 및 분석 함수 ---
@st.cache_data
def preprocess_mental_health_data_cached(file_path, condition_name):
    try:
        temp_df_for_headers = pd.read_csv(file_path, encoding='utf-8-sig', header=None, nrows=5)
        if temp_df_for_headers.shape[0] < 5: return pd.DataFrame()
        header_year_row = temp_df_for_headers.iloc[3]; header_metric_row = temp_df_for_headers.iloc[4]
        df_data_part = pd.read_csv(file_path, encoding='utf-8-sig', header=None, skiprows=5)
        if df_data_part.empty: return pd.DataFrame()
    except Exception as e: st.error(f"'{os.path.basename(file_path)}' 파일 읽기 중 오류: {e}"); return pd.DataFrame()

    base_cols = ['시도', '시군구', '성별', '연령구분']; num_base_cols = len(base_cols)
    year_metric_cols = []; current_year_header = ""
    max_cols_to_process = min(len(header_year_row), len(header_metric_row), df_data_part.shape[1])
    for i in range(num_base_cols, max_cols_to_process):
        year_val_raw = header_year_row[i]; metric_val_raw = header_metric_row[i]
        year_val = str(year_val_raw).strip() if pd.notna(year_val_raw) else ""
        metric_val = str(metric_val_raw).strip() if pd.notna(metric_val_raw) else ""
        if '년' in year_val: current_year_header = year_val.replace(" ", "")
        if current_year_header and metric_val: year_metric_cols.append(f"{current_year_header}_{metric_val}")

    expected_metric_cols_count = df_data_part.shape[1] - num_base_cols
    if len(year_metric_cols) > expected_metric_cols_count: year_metric_cols = year_metric_cols[:expected_metric_cols_count]
    elif len(year_metric_cols) < expected_metric_cols_count: df_data_part = df_data_part.iloc[:, :num_base_cols + len(year_metric_cols)]
    if not year_metric_cols :
        st.warning(f"{condition_name} 파일에서 연도별 데이터 컬럼명을 생성하지 못했습니다.")
        return pd.DataFrame()
    df_data_part.columns = base_cols + year_metric_cols

    value_columns_pattern = ['환자수', '요양급여비용', '입내원일수']
    columns_to_clean = [col for col in df_data_part.columns if any(pat in col for pat in value_columns_pattern)]
    for col in columns_to_clean:
        if col in df_data_part.columns:
            if df_data_part[col].dtype == 'object':
                df_data_part[col] = df_data_part[col].astype(str).str.replace(',', '', regex=False).str.strip()
                df_data_part[col] = pd.to_numeric(df_data_part[col], errors='coerce')
            df_data_part[col] = df_data_part[col].fillna(0).astype(int)
    df_seoul = df_data_part[df_data_part['시도'] == '서울'].copy(); df_seoul['질환명'] = condition_name
    return df_seoul

@st.cache_data
def analyze_elderly_mental_condition_cached(df_seoul_condition, elderly_groups):
    if df_seoul_condition is None or df_seoul_condition.empty: return None, None, None
    df_elderly = df_seoul_condition[df_seoul_condition['연령구분'].isin(elderly_groups)].copy()
    if df_elderly.empty: return None, None, None
    id_vars_melt = ['시도', '시군구', '성별', '연령구분', '질환명']
    value_vars_melt = [col for col in df_elderly.columns if any(year_str in col for year_str in ['2019년','2020년','2021년','2022년','2023년'])]
    if not value_vars_melt: return None, None, None
    df_melted = df_elderly.melt(id_vars=id_vars_melt, value_vars=value_vars_melt, var_name='연도_항목', value_name='값')
    try:
        split_data = df_melted['연도_항목'].str.split('_', expand=True, n=1)
        if split_data.shape[1] < 2: return None, None, None
        df_melted[['연도', '항목']] = split_data
    except Exception: return None, None, None
    df_melted['연도'] = df_melted['연도'].str.replace('년', '').astype(int)
    df_patients_data = df_melted[df_melted['항목'] == '환자수'].copy()
    if df_patients_data.empty: return None, None, None

    total_patients_yearly = df_patients_data.groupby('연도')['값'].sum().reset_index()
    total_patients_yearly.rename(columns={'값': '총 노인 환자수'}, inplace=True)
    patients_gender_yearly = df_patients_data.groupby(['연도', '성별'])['값'].sum().unstack(fill_value=0).reset_index()
    if '남' not in patients_gender_yearly.columns: patients_gender_yearly['남'] = 0
    if '여' not in patients_gender_yearly.columns: patients_gender_yearly['여'] = 0
    elderly_subgroups_map = {'60~69세': '60대', '70~79세': '70대', '80~89세': '80세 이상', '90~99세': '80세 이상', '100세 이상': '80세 이상'}
    df_patients_data['세부연령그룹'] = df_patients_data['연령구분'].map(elderly_subgroups_map)
    df_patients_subgroup_data = df_patients_data.dropna(subset=['세부연령그룹'])
    patients_subgroup_gender_yearly = pd.DataFrame()
    if not df_patients_subgroup_data.empty:
        patients_subgroup_gender_yearly = df_patients_subgroup_data.groupby(['연도', '성별', '세부연령그룹'])['값'].sum().reset_index()
        patients_subgroup_gender_yearly.rename(columns={'값': '환자수'}, inplace=True)
    return total_patients_yearly, patients_gender_yearly, patients_subgroup_gender_yearly

# --- Colab Cell 15, 17, 20, 21의 로직을 Streamlit용 함수로 변환 ---
@st.cache_data
def preprocess_lonely_elderly_data_revised_cached(file_path):
    # '독거노인+현황(연령별_동별)_20250523002757.csv' 파일은 일반 CSV로 가정하고 header=0으로 로드
    df_raw = load_csv(file_path, header_config=0, encoding='utf-8-sig') # encoding 명시
    if df_raw is None or df_raw.empty:
        st.warning(f"독거노인 파일 '{os.path.basename(file_path)}'을 로드하지 못했거나 비어있습니다.")
        return pd.DataFrame()

    # 예상되는 컬럼명 (실제 파일에 따라 확인 및 조정 필요)
    # 예: '동별(1)', '독거노인별', '성별', '2019 년', '2020 년', ...
    # '동별(1)' 컬럼을 '시군구'로, '독거노인별' 컬럼을 사용해 필터링, '성별'로 필터링
    # 연도 컬럼명 정리
    year_cols_rename_map = {}
    for col in df_raw.columns:
        if " 년" in col and col[:4].isdigit(): # "2019 년" 형태의 컬럼
            year_cols_rename_map[col] = col.replace(" 년", "").strip()
        elif col.endswith("년") and col[:-1].isdigit(): # "2019년" 형태의 컬럼
            year_cols_rename_map[col] = col[:-1]

    df_raw.rename(columns=year_cols_rename_map, inplace=True)
    processed_year_cols = [str(year) for year in range(2019, 2024) if str(year) in df_raw.columns]

    # 컬럼명 확인 (파일의 실제 컬럼명에 맞게 조정)
    sigungu_col_name = '동별(1)' # 또는 '동별', '자치구' 등 파일의 실제 컬럼명
    dokgo_type_col_name = '독거노인별' # 파일의 실제 컬럼명
    gender_col_name = '성별' # 파일의 실제 컬럼명

    if not all(c in df_raw.columns for c in [sigungu_col_name, dokgo_type_col_name, gender_col_name]):
        st.error(f"독거노인 파일에 필요한 컬럼({sigungu_col_name}, {dokgo_type_col_name}, {gender_col_name})이 없습니다. 파일 내용을 확인해주세요.")
        # print("사용 가능한 컬럼:", df_raw.columns.tolist()) # 디버깅용
        return pd.DataFrame()

    df_filtered = df_raw[
        (df_raw[dokgo_type_col_name] == '합계') & # 이 부분은 '독거노인현황(성별)' 파일과 다를 수 있음. 파일 확인 필요.
        (df_raw[gender_col_name].isin(['남', '여']))
    ].copy()

    df_filtered.rename(columns={sigungu_col_name: '시군구'}, inplace=True)
    df_sigungu_data = df_filtered[df_filtered['시군구'].str.endswith('구', na=False)].copy()

    if not processed_year_cols or df_sigungu_data.empty: return pd.DataFrame()

    df_sigungu_melted = df_sigungu_data.melt(
        id_vars=['시군구', gender_col_name], # 성별 컬럼명 사용
        value_vars=processed_year_cols,
        var_name='연도', value_name='독거노인수'
    )
    # melt 후 성별 컬럼명을 일관되게 '성별'로 변경
    df_sigungu_melted.rename(columns={gender_col_name: '성별'}, inplace=True)


    df_sigungu_melted['연도'] = pd.to_numeric(df_sigungu_melted['연도'], errors='coerce').dropna().astype(int)
    df_sigungu_melted['독거노인수'] = pd.to_numeric(df_sigungu_melted['독거노인수'].astype(str).str.replace(',', '', regex=False), errors='coerce').fillna(0).astype(int)
    return df_sigungu_melted

@st.cache_data
def preprocess_elderly_population_total_cached(file_path):
    # 이 함수는 '노인 인구 및 독거노인 현황' 페이지의 preprocess_goryeong_data_cached 와 유사한 기능을 수행해야 함
    # 해당 함수를 직접 사용하거나, 그 로직을 여기에 맞게 적용하는 것이 좋음.
    # 여기서는 이전 코드의 구조를 유지하며, MultiIndex를 가정하고 로드 시도
    df_goryeong_raw = load_csv(file_path, header_config=[0,1,2,3])
    if df_goryeong_raw is None or df_goryeong_raw.empty:
        st.warning(f"고령자현황 파일 '{os.path.basename(file_path)}'을 로드하지 못했거나 비어있습니다.")
        return pd.DataFrame()

    try:
        # MultiIndex 컬럼을 가진다고 가정
        df_goryeong_raw.columns = pd.MultiIndex.from_tuples(df_goryeong_raw.columns)
        # 필요한 데이터 추출 로직 (Colab Cell 20 기반)
        # '합계' 행 및 '소계'가 아닌 구 데이터 필터링
        df_filtered = df_goryeong_raw[df_goryeong_raw.iloc[:,0] == '합계']
        df_sigungu_rows = df_filtered[df_filtered.iloc[:,1] != '소계']

        df_sigungu_elderly_pop_list_total = []
        for index, row_data in df_sigungu_rows.iterrows():
            sigungu_name = row_data.iloc[1] # 두 번째 열이 구 이름
            if pd.isna(sigungu_name) or not str(sigungu_name).strip().endswith('구'):
                continue

            for year_int in range(2019, 2024):
                year_str = str(year_int)
                # 컬럼명 (Year, '65세이상 인구', '소계', '소계') 또는 유사한 형태
                # 실제 컬럼명 확인 후 정확한 키 사용 필요
                pop_col_key = None
                for col in row_data.index: # MultiIndex 순회
                    if isinstance(col, tuple) and len(col) == 4:
                        if col[0] == year_str and '65세이상' in col[1] and col[2] == '소계' and col[3] == '소계':
                            pop_col_key = col
                            break
                
                total_elderly_val = 0
                if pop_col_key and pop_col_key in row_data.index:
                    value_raw = row_data[pop_col_key]
                    if pd.notna(value_raw) and str(value_raw).strip() not in ['', '-']:
                       total_elderly_val = int(str(value_raw).replace(',', '').strip())
                
                df_sigungu_elderly_pop_list_total.append({
                    '시군구': sigungu_name.strip(), '연도': year_int, '총_65세_이상_노인인구': total_elderly_val
                })
        if df_sigungu_elderly_pop_list_total:
            return pd.DataFrame(df_sigungu_elderly_pop_list_total)
        return pd.DataFrame()

    except Exception as e:
        st.error(f"고령자 현황 파일 처리 중 오류(preprocess_elderly_population_total_cached): {e}")
        # print("사용 가능한 컬럼:", df_goryeong_raw.columns.tolist()) # 디버깅용
        return pd.DataFrame()


@st.cache_data
def aggregate_mental_patients_sigungu_gender_cached(dataframes_by_condition, elderly_age_groups):
    all_sigungu_mental_patients_gender_list = []
    if not dataframes_by_condition: return pd.DataFrame()

    for condition_name, df_condition in dataframes_by_condition.items():
        if df_condition is None or df_condition.empty: continue
        df_elderly = df_condition[df_condition['연령구분'].isin(elderly_age_groups)].copy()
        if df_elderly.empty: continue

        id_vars = ['시군구', '성별', '질환명']
        value_vars = [col for col in df_elderly.columns if any(y_str in col for y_str in ['2019년','2020년','2021년','2022년','2023년']) and '환자수' in col]
        if not value_vars: continue

        df_melted = df_elderly.melt(id_vars=id_vars, value_vars=value_vars, var_name='연도_항목', value_name='정신질환_환자수')
        df_melted['연도'] = df_melted['연도_항목'].str.extract(r'(\d{4})년')[0].astype(int)
        df_melted_gender_specific = df_melted[df_melted['성별'].isin(['남', '여'])].copy()

        sigungu_year_gender_condition_patients = df_melted_gender_specific.groupby(
            ['연도', '시군구', '성별', '질환명']
        )['정신질환_환자수'].sum().reset_index()
        all_sigungu_mental_patients_gender_list.append(sigungu_year_gender_condition_patients)

    if not all_sigungu_mental_patients_gender_list: return pd.DataFrame()
    return pd.concat(all_sigungu_mental_patients_gender_list).reset_index(drop=True)

@st.cache_data
def aggregate_mental_patients_sigungu_total_cached(dataframes_by_condition, elderly_age_groups):
    all_mental_patients_list_for_ratio = []
    if not dataframes_by_condition: return pd.DataFrame()

    for condition_name, df_condition in dataframes_by_condition.items():
        if df_condition is None or df_condition.empty: continue
        df_elderly = df_condition[df_condition['연령구분'].isin(elderly_age_groups)].copy()
        if df_elderly.empty: continue

        id_vars = ['시군구', '질환명']
        value_vars = [col for col in df_elderly.columns if any(y_str in col for y_str in ['2019년','2020년','2021년','2022년','2023년']) and '환자수' in col]
        if not value_vars: continue

        df_melted = df_elderly.melt(id_vars=id_vars, value_vars=value_vars, var_name='연도_항목', value_name='값')
        df_melted['연도'] = df_melted['연도_항목'].str.extract(r'(\d{4})년')[0].astype(int)

        sigungu_year_condition_total_patients = df_melted.groupby(
            ['연도', '시군구', '질환명']
        )['값'].sum().reset_index()
        sigungu_year_condition_total_patients.rename(columns={'값': '질환별_노인_환자수_총합'}, inplace=True)
        all_mental_patients_list_for_ratio.append(sigungu_year_condition_total_patients)

    if not all_mental_patients_list_for_ratio: return pd.DataFrame()
    return pd.concat(all_mental_patients_list_for_ratio).reset_index(drop=True)


# --- 기존 시각화 함수 ---
def plot_total_elderly_trend(total_patients_df, condition_name):
    if total_patients_df is None or total_patients_df.empty: st.info(f"노인 {condition_name} 환자수 총계 추이 데이터를 그릴 수 없습니다."); return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=total_patients_df, x='연도', y='총 노인 환자수', marker='o', ax=ax, color='steelblue')
    ax.set_title(f'서울시 노인 {condition_name} 환자수 추이', fontsize=15)
    ax.set_xlabel('연도'); ax.set_ylabel('총 노인 환자수'); ax.grid(True)
    if not total_patients_df.empty: ax.set_xticks(total_patients_df['연도'].unique())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}')); st.pyplot(fig)

def plot_gender_elderly_trend(patients_gender_df, condition_name):
    if patients_gender_df is None or patients_gender_df.empty: st.info(f"노인 {condition_name} 성별 환자수 추이 데이터를 그릴 수 없습니다."); return
    fig, ax = plt.subplots(figsize=(10, 5))
    if '남' in patients_gender_df.columns: ax.plot(patients_gender_df['연도'], patients_gender_df['남'], marker='o', label='남성', color='dodgerblue')
    if '여' in patients_gender_df.columns: ax.plot(patients_gender_df['연도'], patients_gender_df['여'], marker='s', label='여성', color='hotpink')
    ax.set_title(f'서울시 노인 {condition_name} 환자수 추이 (성별)', fontsize=15)
    ax.set_xlabel('연도'); ax.set_ylabel('노인 환자수'); ax.legend(); ax.grid(True)
    if not patients_gender_df.empty: ax.set_xticks(patients_gender_df['연도'].unique())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}')); st.pyplot(fig)

def plot_subgroup_gender_elderly_trend(patients_subgroup_df, condition_name):
    if patients_subgroup_df is None or patients_subgroup_df.empty: st.info(f"노인 {condition_name} 세부 연령대 및 성별 환자수 추이 데이터를 그릴 수 없습니다."); return
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=patients_subgroup_df, x='연도', y='환자수', hue='세부연령그룹', style='성별', marker='o', markersize=7, ax=ax)
    ax.set_title(f'서울시 {condition_name} 노인 환자수 추이 (세부 연령대 및 성별)', fontsize=15)
    ax.set_xlabel('연도'); ax.set_ylabel('환자 수')
    if not patients_subgroup_df.empty: ax.set_xticks(sorted(patients_subgroup_df['연도'].unique()))
    ax.legend(title='구분', bbox_to_anchor=(1.02, 1), loc='upper left', title_fontsize='11', fontsize='10')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}')); plt.tight_layout(rect=[0,0,0.85,1]); st.pyplot(fig)

def plot_all_conditions_yearly_comparison(all_conditions_summary_df, selected_year_int):
    if all_conditions_summary_df.empty: st.info("종합 비교를 위한 데이터가 없습니다."); return
    year_df_to_plot = all_conditions_summary_df[all_conditions_summary_df['연도'] == selected_year_int].copy()
    if year_df_to_plot.empty: st.info(f"{selected_year_int}년 데이터가 없어 종합 비교 그래프를 생성할 수 없습니다."); return
    year_df_to_plot = year_df_to_plot.sort_values(by='총 노인 환자수', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=year_df_to_plot, x='질환명', y='총 노인 환자수', color='teal', ax=ax)
    ax.set_title(f'서울시 {selected_year_int}년 노인 정신질환별 환자수 비교', fontsize=15)
    ax.set_xlabel('질환명'); ax.set_ylabel('총 노인 환자수'); plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10); ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}')); plt.tight_layout(); st.pyplot(fig)

def plot_pie_chart_by_year(all_conditions_summary_df, selected_year_int):
    if all_conditions_summary_df.empty: st.info("파이차트 생성을 위한 데이터가 없습니다."); return
    year_data_for_pie = all_conditions_summary_df[all_conditions_summary_df['연도'] == selected_year_int].copy()
    if year_data_for_pie.empty or year_data_for_pie['총 노인 환자수'].sum() == 0: st.info(f"{selected_year_int}년 질환별 환자 수 비율을 계산할 데이터가 없습니다."); return
    year_data_for_pie = year_data_for_pie.sort_values(by='총 노인 환자수', ascending=False)
    fig, ax = plt.subplots(figsize=(9, 9))
    wedges, texts, autotexts = ax.pie(
        year_data_for_pie['총 노인 환자수'], labels=year_data_for_pie['질환명'],
        autopct=lambda p: '{:.1f}% ({:,.0f}명)'.format(p, p * np.sum(year_data_for_pie['총 노인 환자수']) / 100),
        startangle=140, pctdistance=0.75
    )
    for text in texts: text.set_fontsize(10)
    for autotext in autotexts: autotext.set_fontsize(9); autotext.set_color('black')
    centre_circle = plt.Circle((0,0),0.60,fc='white'); fig.gca().add_artist(centre_circle)
    ax.set_title(f'서울시 {selected_year_int}년 전체 노인 정신질환별 환자 수 비율', fontsize=16)
    ax.axis('equal'); plt.tight_layout(); st.pyplot(fig)

# --- 새로운 시각화 함수 (Colab Cell 19, 22 기반) ---
def plot_sigungu_lonely_vs_mental_gender_ratio(df_merged, selected_condition, selected_year):
    if df_merged.empty:
        st.info(f"{selected_year}년 {selected_condition}에 대한 구별 독거노인 성별 대비 정신질환자 비율 데이터가 없습니다.")
        return

    df_plot = df_merged[
        (df_merged['질환명'] == selected_condition) &
        (df_merged['연도'] == selected_year)
    ].copy()

    if df_plot.empty or ('독거노인_대비_질환자_비율(%)' in df_plot.columns and df_plot['독거노인_대비_질환자_비율(%)'].sum() == 0):
        st.info(f"{selected_year}년 {selected_condition}에 대한 유의미한 구별 비율 데이터가 없습니다.")
        return
    if '독거노인_대비_질환자_비율(%)' not in df_plot.columns:
        st.warning("`독거노인_대비_질환자_비율(%)` 컬럼이 없어 그래프를 그릴 수 없습니다.")
        return


    order_sigungu = sorted(df_plot['시군구'].unique())

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.barplot(
        data=df_plot, x='시군구', y='독거노인_대비_질환자_비율(%)', hue='성별',
        order=order_sigungu, palette={'남': 'cornflowerblue', '여': 'salmon'}, ax=ax
    )
    ax.set_title(f'서울시 구별 <{selected_condition}> 독거노인 성별 대비 환자 비율 ({selected_year}년)', fontsize=18)
    ax.set_xlabel('시군구', fontsize=14); ax.set_ylabel(f'{selected_condition} 환자 비율 (%)', fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center', fontsize=9)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(title='성별', fontsize=12, title_fontsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for p in ax.patches:
        if p.get_height() > 0.001:
            ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontsize=8)
    plt.tight_layout(); st.pyplot(fig)

def plot_sigungu_total_elderly_vs_mental_ratio(df_merged, selected_condition, selected_year):
    if df_merged.empty:
        st.info(f"{selected_year}년 {selected_condition}에 대한 구별 전체 노인 대비 정신질환자 비율 데이터가 없습니다.")
        return

    df_plot = df_merged[
        (df_merged['질환명'] == selected_condition) &
        (df_merged['연도'] == selected_year)
    ].copy()

    if df_plot.empty or ('전체노인인구_대비_질환자_비율(%)' in df_plot.columns and df_plot['전체노인인구_대비_질환자_비율(%)'].sum() == 0):
        st.info(f"{selected_year}년 {selected_condition}에 대한 유의미한 구별 비율 데이터가 없습니다.")
        return
    if '전체노인인구_대비_질환자_비율(%)' not in df_plot.columns:
        st.warning("`전체노인인구_대비_질환자_비율(%)` 컬럼이 없어 그래프를 그릴 수 없습니다.")
        return

    df_plot = df_plot.sort_values(by='전체노인인구_대비_질환자_비율(%)', ascending=False)

    fig, ax = plt.subplots(figsize=(20, 10))
    bars = sns.barplot(data=df_plot, x='시군구', y='전체노인인구_대비_질환자_비율(%)', palette='viridis_r', ax=ax)
    ax.set_title(f'서울시 구별 <{selected_condition}> 전체 65세 이상 노인 인구 대비 환자 비율 ({selected_year}년)', fontsize=16)
    ax.set_xlabel('시군구', fontsize=14); ax.set_ylabel(f'{selected_condition} 환자 비율 (%)', fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for bar_patch in bars.patches:
        if bar_patch.get_height() > 0.001:
            ax.text(bar_patch.get_x() + bar_patch.get_width() / 2, bar_patch.get_height(),
                    f"{bar_patch.get_height():.2f}%", ha='center', va='bottom', fontsize=8)
    plt.tight_layout(); st.pyplot(fig)


# --- Streamlit 페이지 레이아웃 ---
def run_mental_health_page():
    st.title("서울시 노인 정신질환 현황")
    set_korean_font()

    file_paths_mental_health = {
        "불면증": "data/지역별 불면증 진료현황(2019년~2023년).csv", "우울증": "data/지역별 우울증 진료현황(2019년~2023년).csv",
        "불안장애": "data/지역별 불안장애 진료현황(2019년~2023년).csv", "조울증": "data/지역별 조울증 진료현황(2019년~2023년).csv",
        "조현병": "data/지역별 조현병 진료현황(2019년~2023년).csv"
    }
    # 독거노인 파일명 변경
    lonely_elderly_file_path_st = 'data/독거노인+현황(연령별_동별)_20250523002757.csv'
    elderly_population_total_file_path_st = 'data/고령자현황_20250531210628.csv'

    elderly_age_groups_mental = ['60~69세', '70~79세', '80~89세', '90~99세', '100세 이상']
    available_years_int = [2019, 2020, 2021, 2022, 2023]

    if "selected_year_mental" not in st.session_state:
        st.session_state.selected_year_mental = available_years_int[-1]

    # 데이터 로드
    dataframes_by_condition = {}
    all_conditions_summaries_list = []
    for condition_key, file_path_value in file_paths_mental_health.items():
        df_condition_raw = preprocess_mental_health_data_cached(file_path_value, condition_key)
        if not df_condition_raw.empty:
            dataframes_by_condition[condition_key] = df_condition_raw
            total_df, _, _ = analyze_elderly_mental_condition_cached(df_condition_raw, elderly_age_groups_mental)
            if total_df is not None and not total_df.empty:
                temp_summary = total_df.copy(); temp_summary['질환명'] = condition_key
                all_conditions_summaries_list.append(temp_summary)
    
    df_sigungu_lonely_elderly_gender_st = preprocess_lonely_elderly_data_revised_cached(lonely_elderly_file_path_st)
    df_sigungu_total_elderly_population_st = preprocess_elderly_population_total_cached(elderly_population_total_file_path_st)
    df_sigungu_mental_patients_gender_all_conditions_st = aggregate_mental_patients_sigungu_gender_cached(dataframes_by_condition, elderly_age_groups_mental)
    df_sigungu_mental_total_patients_for_ratio_st = aggregate_mental_patients_sigungu_total_cached(dataframes_by_condition, elderly_age_groups_mental)


    if not dataframes_by_condition:
        st.error("정신질환 데이터를 하나도 로드하지 못했습니다."); return

    main_tab1, main_tab2, main_tab3 = st.tabs(["개별 질환 분석 (서울시 전체)", "정신질환 종합 비교 (서울시 전체)", "구별 상세 분석"])

    with main_tab1:
        st.subheader("개별 정신질환 분석 (서울시 전체)")
        selected_condition_name_tab1 = st.selectbox(
            "분석할 질환을 선택하세요:", list(dataframes_by_condition.keys()), key="mental_condition_select_tab1"
        )
        if selected_condition_name_tab1 and selected_condition_name_tab1 in dataframes_by_condition:
            df_to_analyze = dataframes_by_condition[selected_condition_name_tab1]
            total_res, gender_res, subgroup_res = analyze_elderly_mental_condition_cached(df_to_analyze, elderly_age_groups_mental)
            st.markdown(f"#### {selected_condition_name_tab1} 분석 결과")
            plot_type_tab1, plot_type_tab2, plot_type_tab3 = st.tabs(["연도별 총계", "연도별 성별", "세부 연령대 및 성별"])
            with plot_type_tab1: plot_total_elderly_trend(total_res, selected_condition_name_tab1)
            with plot_type_tab2: plot_gender_elderly_trend(gender_res, selected_condition_name_tab1)
            with plot_type_tab3: plot_subgroup_gender_elderly_trend(subgroup_res, selected_condition_name_tab1)

    with main_tab2:
        st.subheader("정신질환 종합 비교 (서울시 전체)")
        selected_year_tab2 = st.slider(
            "조회 연도 선택 (종합 비교용)", available_years_int[0], available_years_int[-1],
            st.session_state.selected_year_mental, key="mental_year_slider_tab2"
        )
        if st.session_state.selected_year_mental != selected_year_tab2:
            st.session_state.selected_year_mental = selected_year_tab2
            st.rerun()
        
        current_selected_year_for_plots = st.session_state.selected_year_mental

        if all_conditions_summaries_list:
            all_conditions_summary_df_final = pd.concat(all_conditions_summaries_list).reset_index(drop=True)
            st.markdown(f"##### {current_selected_year_for_plots}년 질환별 환자수 비교 (막대)")
            plot_all_conditions_yearly_comparison(all_conditions_summary_df_final, current_selected_year_for_plots)
            st.markdown(f"##### {current_selected_year_for_plots}년 질환별 환자수 비율 (파이)")
            plot_pie_chart_by_year(all_conditions_summary_df_final, current_selected_year_for_plots)
        else: st.info("정신질환 종합 비교를 위한 데이터가 충분하지 않습니다.")

    with main_tab3:
        st.subheader("구별 상세 분석")
        
        # main_tab2의 슬라이더 값과 연동하기 위해 session_state 사용
        # 또는 이 탭에서 독립적인 연도 선택기 사용 가능
        # 여기서는 main_tab2와 연동되도록 session_state.selected_year_mental을 사용
        selected_year_tab3 = st.session_state.selected_year_mental 
        
        # 이 탭에서 연도/질환을 독립적으로 선택하고 싶다면 아래 주석 해제
        # selected_year_tab3 = st.selectbox(
        #     "조회 연도 선택 (구별 분석용)", available_years_int, 
        #     index=available_years_int.index(st.session_state.selected_year_mental),
        #     key="mental_year_select_tab3_independent"
        # )

        selected_condition_tab3 = st.selectbox(
            "분석할 질환 선택 (구별 분석용)", list(dataframes_by_condition.keys()),
            key="mental_condition_select_tab3"
        )
        
        df_sigungu_lonely_vs_mental_merged = pd.DataFrame()
        if not df_sigungu_mental_patients_gender_all_conditions_st.empty and not df_sigungu_lonely_elderly_gender_st.empty:
            df_sigungu_lonely_vs_mental_merged = pd.merge(
                df_sigungu_mental_patients_gender_all_conditions_st,
                df_sigungu_lonely_elderly_gender_st,
                on=['연도', '시군구', '성별'], how='left'
            )
            df_sigungu_lonely_vs_mental_merged['독거노인수'] = pd.to_numeric(df_sigungu_lonely_vs_mental_merged['독거노인수'], errors='coerce').fillna(0)
            df_sigungu_lonely_vs_mental_merged['정신질환_환자수'] = pd.to_numeric(df_sigungu_lonely_vs_mental_merged['정신질환_환자수'], errors='coerce').fillna(0)
            df_sigungu_lonely_vs_mental_merged['독거노인_대비_질환자_비율(%)'] = np.where(
                df_sigungu_lonely_vs_mental_merged['독거노인수'] > 0,
                (df_sigungu_lonely_vs_mental_merged['정신질환_환자수'] / df_sigungu_lonely_vs_mental_merged['독거노인수']) * 100, 0
            )
        
        df_sigungu_total_elderly_vs_mental_merged = pd.DataFrame()
        if not df_sigungu_mental_total_patients_for_ratio_st.empty and not df_sigungu_total_elderly_population_st.empty:
             df_sigungu_total_elderly_vs_mental_merged = pd.merge(
                df_sigungu_mental_total_patients_for_ratio_st,
                df_sigungu_total_elderly_population_st,
                on=['연도', '시군구'], how='left'
            )
             df_sigungu_total_elderly_vs_mental_merged.rename(columns={'총_65세_이상_노인인구': '분모_전체_노인인구'}, inplace=True, errors='ignore')
             # '분모_전체_노인인구' 컬럼이 없는 경우를 대비한 get 사용 및 기본값 0 처리
             df_sigungu_total_elderly_vs_mental_merged['분모_전체_노인인구'] = pd.to_numeric(df_sigungu_total_elderly_vs_mental_merged.get('분모_전체_노인인구'), errors='coerce').fillna(0)
             df_sigungu_total_elderly_vs_mental_merged['질환별_노인_환자수_총합'] = pd.to_numeric(df_sigungu_total_elderly_vs_mental_merged['질환별_노인_환자수_총합'], errors='coerce').fillna(0)
             
             if '분모_전체_노인인구' in df_sigungu_total_elderly_vs_mental_merged.columns:
                 df_sigungu_total_elderly_vs_mental_merged['전체노인인구_대비_질환자_비율(%)'] = np.where(
                    df_sigungu_total_elderly_vs_mental_merged['분모_전체_노인인구'] > 0,
                    (df_sigungu_total_elderly_vs_mental_merged['질환별_노인_환자수_총합'] / df_sigungu_total_elderly_vs_mental_merged['분모_전체_노인인구']) * 100, 0
                )
             else:
                 st.warning("구별 전체 65세 이상 노인 인구 데이터('분모_전체_노인인구')를 찾을 수 없어 해당 비율을 계산할 수 없습니다.")


        sub_tab_sigungu1, sub_tab_sigungu2 = st.tabs([
            "독거노인 성별 대비 비율", "전체 65세+ 노인 대비 비율"
        ])

        with sub_tab_sigungu1:
            st.markdown(f"##### {selected_year_tab3}년 {selected_condition_tab3} - 구별 독거노인 성별 대비 환자 비율")
            if not df_sigungu_lonely_vs_mental_merged.empty:
                plot_sigungu_lonely_vs_mental_gender_ratio(df_sigungu_lonely_vs_mental_merged, selected_condition_tab3, selected_year_tab3)
            else:
                st.info("독거노인 성별 대비 정신질환자 비율을 계산할 데이터가 부족합니다.")

        with sub_tab_sigungu2:
            st.markdown(f"##### {selected_year_tab3}년 {selected_condition_tab3} - 구별 전체 65세+ 노인 대비 환자 비율")
            if not df_sigungu_total_elderly_vs_mental_merged.empty:
                 plot_sigungu_total_elderly_vs_mental_ratio(df_sigungu_total_elderly_vs_mental_merged, selected_condition_tab3, selected_year_tab3)
            else:
                st.info("전체 65세 이상 노인 대비 정신질환자 비율을 계산할 데이터가 부족합니다.")


if __name__ == "__main__":
    run_mental_health_page()
