import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import set_korean_font, load_csv
import os
import numpy as np

# --- 데이터 전처리 및 분석 함수 (이전과 동일) ---
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
    if not year_metric_cols : # 추가된 방어 코드
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

# --- 시각화 함수 (색상 통일) ---
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
    sns.lineplot(data=patients_subgroup_df, x='연도', y='환자수', hue='세부연령그룹', style='성별', marker='o', markersize=7, ax=ax) # palette, dashes 제거
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
        startangle=140, pctdistance=0.75 # colors 제거
    )
    for text in texts: text.set_fontsize(10)
    for autotext in autotexts: autotext.set_fontsize(9); autotext.set_color('black')
    centre_circle = plt.Circle((0,0),0.60,fc='white'); fig.gca().add_artist(centre_circle)
    ax.set_title(f'서울시 {selected_year_int}년 전체 노인 정신질환별 환자 수 비율', fontsize=16)
    ax.axis('equal'); plt.tight_layout(); st.pyplot(fig)

# --- Streamlit 페이지 레이아웃 ---
def run_mental_health_page():
    st.title("🧠 서울시 연령별 정신질환 현황")
    set_korean_font()

    file_paths_mental_health = {
        "불면증": "data/지역별 불면증 진료현황(2019년~2023년).csv", "우울증": "data/지역별 우울증 진료현황(2019년~2023년).csv",
        "불안장애": "data/지역별 불안장애 진료현황(2019년~2023년).csv", "조울증": "data/지역별 조울증 진료현황(2019년~2023년).csv",
        "조현병": "data/지역별 조현병 진료현황(2019년~2023년).csv"
    }
    elderly_age_groups_mental = ['60~69세', '70~79세', '80~89세', '90~99세', '100세 이상']
    available_years_int = [2019, 2020, 2021, 2022, 2023]

    if "selected_year_mental" not in st.session_state:
        st.session_state.selected_year_mental = available_years_int[-1]

    selected_year_int = st.slider(
        "조회 연도 선택 (종합 비교용)",
        min_value=available_years_int[0],
        max_value=available_years_int[-1],
        step=1,
        value=st.session_state.selected_year_mental,
        key="mental_year_slider_main" # 키 변경
    )
    st.session_state.selected_year_mental = selected_year_int

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
        else: st.warning(f"{condition_key} 데이터를 로드하거나 전처리하는 데 실패했습니다.")

    if not dataframes_by_condition:
        st.error("정신질환 데이터를 하나도 로드하지 못했습니다."); return

    main_tab1, main_tab2 = st.tabs(["개별 질환 분석", "정신질환 종합 비교"])

    with main_tab1:
        st.subheader("개별 정신질환 분석")
        selected_condition_name = st.selectbox( # 질환 선택은 selectbox 유지
            "분석할 질환을 선택하세요:",
            list(dataframes_by_condition.keys()),
            key="mental_condition_select_main_tab_selectbox"
        )

        if selected_condition_name and selected_condition_name in dataframes_by_condition:
            df_to_analyze = dataframes_by_condition[selected_condition_name]
            total_res, gender_res, subgroup_res = analyze_elderly_mental_condition_cached(df_to_analyze, elderly_age_groups_mental)

            st.markdown(f"#### 📊 {selected_condition_name} 분석 결과")
            # 그래프 유형 선택 라디오 버튼 -> 탭으로 변경
            plot_type_tab1, plot_type_tab2, plot_type_tab3 = st.tabs([
                "연도별 총계", "연도별 성별", "세부 연령대 및 성별"
            ])
            with plot_type_tab1:
                plot_total_elderly_trend(total_res, selected_condition_name)
            with plot_type_tab2:
                plot_gender_elderly_trend(gender_res, selected_condition_name)
            with plot_type_tab3:
                plot_subgroup_gender_elderly_trend(subgroup_res, selected_condition_name)

    with main_tab2:
        st.subheader("정신질환 종합 비교")
        if all_conditions_summaries_list:
            all_conditions_summary_df_final = pd.concat(all_conditions_summaries_list).reset_index(drop=True)

            # 종합 비교 유형 선택 selectbox -> 탭으로 변경
            overall_plot_tab1, overall_plot_tab2 = st.tabs([
                "질환별 환자수 (막대)", "질환별 환자수 비율 (파이)"
            ])
            with overall_plot_tab1:
                plot_all_conditions_yearly_comparison(all_conditions_summary_df_final, selected_year_int)
            with overall_plot_tab2:
                plot_pie_chart_by_year(all_conditions_summary_df_final, selected_year_int)
        else: st.info("정신질환 종합 비교를 위한 데이터가 충분하지 않습니다.")

if __name__ == "__main__":
    run_mental_health_page()