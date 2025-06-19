# --- START OF 5_MentalHealth.py ---
import streamlit as st
import pandas as pd
from utils import set_korean_font, load_csv
from chart_utils import (
    plot_total_elderly_trend,
    plot_gender_elderly_trend,
    plot_subgroup_gender_elderly_trend,
    plot_all_conditions_yearly_comparison,
    plot_pie_chart_by_year,
    plot_sigungu_mental_patients_by_condition_year
)
import os

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


def run_mental_health_page():
    set_korean_font()
    st.title("서울시 노인 정신질환 현황")

    file_paths_mental_health = {
        "불면증": "data/지역별 불면증 진료현황(2019년~2023년).csv",
        "우울증": "data/지역별 우울증 진료현황(2019년~2023년).csv",
        "불안장애": "data/지역별 불안장애 진료현황(2019년~2023년).csv",
        "조울증": "data/지역별 조울증 진료현황(2019년~2023년).csv",
        "조현병": "data/지역별 조현병 진료현황(2019년~2023년).csv"
    }
    
    elderly_age_groups_mental = ['60~69세', '70~79세', '80~89세', '90~99세', '100세 이상']
    available_years_for_mental_analysis = [2019, 2020, 2021, 2022, 2023]

    if "selected_year_mental_tab2" not in st.session_state:
        st.session_state.selected_year_mental_tab2 = available_years_for_mental_analysis[-1]
    if "selected_year_mental_tab3" not in st.session_state:
        st.session_state.selected_year_mental_tab3 = available_years_for_mental_analysis[-1]

    dataframes_by_condition = {}
    all_conditions_summaries_list = []
    at_least_one_file_loaded = False

    for condition_key, file_path_value in file_paths_mental_health.items():
        if not os.path.exists(file_path_value):
            continue 
        df_condition_raw = preprocess_mental_health_data_cached(file_path_value, condition_key)
        if df_condition_raw is not None and not df_condition_raw.empty:
            dataframes_by_condition[condition_key] = df_condition_raw
            at_least_one_file_loaded = True
            total_df, _, _ = analyze_elderly_mental_condition_cached(df_condition_raw, elderly_age_groups_mental)
            if total_df is not None and not total_df.empty:
                temp_summary = total_df.copy()
                temp_summary['질환명'] = condition_key
                all_conditions_summaries_list.append(temp_summary)

    if not at_least_one_file_loaded:
        st.error("필수 정신질환 데이터 파일을 하나도 로드하지 못했습니다. 'data' 폴더 내용을 확인해주세요.")
        st.info("다음 파일들이 필요합니다: " + ", ".join([os.path.basename(fp) for fp in file_paths_mental_health.values()]))
        return

    df_sigungu_mental_total_patients_for_tab3 = aggregate_mental_patients_sigungu_total_cached(dataframes_by_condition, elderly_age_groups_mental)

    tab1_title = "개별 질환 분석"
    tab2_title = "정신질환 종합 비교"
    tab3_title = "구별 정신질환자 수"

    main_tab1, main_tab2, main_tab3 = st.tabs([tab1_title, tab2_title, tab3_title])

    with main_tab1:
        st.subheader(tab1_title) # 요청사항 1: (서울시 전체) 제거됨
        
        selected_condition_name_tab1 = st.selectbox(
            "분석할 질환을 선택하세요:", 
            list(dataframes_by_condition.keys()) if dataframes_by_condition else ["데이터 없음"], 
            key="mental_condition_select_tab1_final_v2" # 키 변경
        )
        if dataframes_by_condition and selected_condition_name_tab1 in dataframes_by_condition:
            df_to_analyze = dataframes_by_condition[selected_condition_name_tab1]
            total_res, gender_res, subgroup_res = analyze_elderly_mental_condition_cached(df_to_analyze, elderly_age_groups_mental)
            
            plot_type_tabs_main1 = st.tabs(["연도별 총계", "연도별 성별", "세부 연령대 및 성별"])
            with plot_type_tabs_main1[0]: 
                plot_total_elderly_trend(total_res, selected_condition_name_tab1)
            with plot_type_tabs_main1[1]: 
                plot_gender_elderly_trend(gender_res, selected_condition_name_tab1)
            with plot_type_tabs_main1[2]: 
                plot_subgroup_gender_elderly_trend(subgroup_res, selected_condition_name_tab1)
        elif not dataframes_by_condition:
            st.info("로드된 질환 데이터가 없습니다.")
        else:
            st.info(f"'{selected_condition_name_tab1}'에 대한 데이터를 찾을 수 없습니다.")


    with main_tab2:
        # 요청사항 1 (종합 비교 탭): 주제를 슬라이더 위로
        current_selected_year_tab2 = st.session_state.selected_year_mental_tab2
        st.subheader(f"{current_selected_year_tab2}년 정신질환별 환자수 비교") # 요청사항 5

        # 요청사항 3 (종합 비교 탭): 슬라이더 레이블에서 괄호 제거
        selected_year_val_tab2 = st.slider(
            "조회 연도 선택", 
            min_value=min(available_years_for_mental_analysis), 
            max_value=max(available_years_for_mental_analysis),
            value=st.session_state.selected_year_mental_tab2, 
            step=1,
            key="mental_year_slider_tab2_main_v3" # 키 변경
        )
        if st.session_state.selected_year_mental_tab2 != selected_year_val_tab2:
            st.session_state.selected_year_mental_tab2 = selected_year_val_tab2
            st.rerun()
        
        if all_conditions_summaries_list:
            all_conditions_summary_df_final = pd.concat(all_conditions_summaries_list).reset_index(drop=True)
            plot_all_conditions_yearly_comparison(all_conditions_summary_df_final, selected_year_val_tab2)
            # 요청사항 2 (종합 비교 탭): 그래프 사이 선 제거
            plot_pie_chart_by_year(all_conditions_summary_df_final, selected_year_val_tab2)
        else: 
            st.info("정신질환 종합 비교를 위한 데이터가 충분하지 않습니다.")

    with main_tab3:
        # 요청사항 4 (구별 분석 탭): 주제를 슬라이더 위로
        current_selected_year_tab3 = st.session_state.selected_year_mental_tab3
        st.subheader(f"{current_selected_year_tab3}년 {tab3_title}") 

        selected_year_val_tab3 = st.slider(
            "조회 연도 선택",
            min_value=min(available_years_for_mental_analysis),
            max_value=max(available_years_for_mental_analysis),
            value=st.session_state.selected_year_mental_tab3,
            step=1,
            key="mental_year_slider_tab3_main_v3" # 키 변경
        )
        if st.session_state.selected_year_mental_tab3 != selected_year_val_tab3:
            st.session_state.selected_year_mental_tab3 = selected_year_val_tab3
            st.rerun()
        
        selected_condition_tab3 = st.selectbox(
            "분석할 질환 선택:",
            list(dataframes_by_condition.keys()) if dataframes_by_condition else ["데이터 없음"],
            key="mental_condition_select_tab3_final_v3" # 키 변경
        )
        
        if dataframes_by_condition and selected_condition_tab3 in dataframes_by_condition:
            if not df_sigungu_mental_total_patients_for_tab3.empty:
                plot_sigungu_mental_patients_by_condition_year(
                    df_sigungu_mental_total_patients_for_tab3, 
                    selected_condition_tab3, 
                    selected_year_val_tab3 
                )
            else:
                st.info("구별 정신질환자 수 집계 데이터를 생성하지 못했습니다.")
        elif not dataframes_by_condition:
            st.info("로드된 질환 데이터가 없습니다.")
        else:
            st.info(f"'{selected_condition_tab3}'에 대한 데이터를 찾을 수 없습니다.")

if __name__ == "__main__":
    run_mental_health_page()
# --- END OF 5_MentalHealth.py ---
