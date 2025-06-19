# --- START OF 5_MentalHealth.py (sys.path 수정 제거, chart_utils의 최소 함수 호출) ---
import streamlit as st
import pandas as pd
import os
import numpy as np

from utils import set_korean_font, load_csv
from chart_utils import ( # chart_utils.py에 정의된 (최소화된) 함수들을 임포트
    plot_total_elderly_trend,
    plot_gender_elderly_trend,
    plot_subgroup_gender_elderly_trend,
    plot_all_conditions_yearly_comparison,
    plot_pie_chart_by_year,
    plot_sigungu_mental_patients_by_condition_year
)

# ... (이하 @st.cache_data 데코레이터가 붙은 함수 정의 및 run_mental_health_page 함수는
#      이전 답변에서 제공된 "전체 5_MentalHealth.py와 또 다른 수정해야 하는 코드가 있으면 그것도 줘"에 대한
#      답변의 5_MentalHealth.py 코드 내용을 참고하되, chart_utils.py에서 비활성화한 함수 호출 부분은
#      st.info 메시지만 뜨도록 하거나 주석 처리해야 합니다.)

# 예시: run_mental_health_page 내에서 그래프 호출 부분을 아래처럼 수정
# with plot_type_tabs_main1[0]:
#     # plot_total_elderly_trend(total_res, selected_condition_name_tab1) # 실제 그래프 대신 info
#     st.info(f"개별 질환 - 연도별 총계 그래프는 chart_utils.py의 해당 함수 내용 복구 필요 ({selected_condition_name_tab1})")

# (모든 정신질환 그래프 호출 부분을 위와 같이 임시로 수정)

# 나머지 run_mental_health_page 로직은 이전 답변 참고

@st.cache_data
def preprocess_mental_health_data_cached(file_path, condition_name):
    try:
        df_header_info = pd.read_csv(file_path, encoding='utf-8-sig', header=None, nrows=5)
        if df_header_info.shape[0] < 5:
            # st.warning(f"{condition_name} 파일에서 헤더 정보를 충분히 읽을 수 없습니다.") # 디버깅 시 주석 해제
            return pd.DataFrame()
        
        header_year_row = df_header_info.iloc[3]
        header_metric_row = df_header_info.iloc[4]
        df_data_part = pd.read_csv(file_path, encoding='utf-8-sig', header=None, skiprows=5)
        if df_data_part.empty:
            return pd.DataFrame()
    except FileNotFoundError:
        st.error(f"파일을 찾을 수 없습니다: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"'{os.path.basename(file_path)}' 파일 읽기 중 오류: {e}")
        return pd.DataFrame()

    base_cols = ['시도', '시군구', '성별', '연령구분']
    num_base_cols = len(base_cols)
    year_metric_cols = []
    current_year_header = ""
    
    num_header_cols = min(len(header_year_row), len(header_metric_row))
    actual_data_metric_cols = df_data_part.shape[1] - num_base_cols
    cols_to_generate_names_for = min(num_header_cols - num_base_cols, actual_data_metric_cols)

    for i_offset in range(cols_to_generate_names_for):
        i = num_base_cols + i_offset
        year_val_raw = header_year_row[i]
        metric_val_raw = header_metric_row[i]
        
        year_val = str(year_val_raw).strip() if pd.notna(year_val_raw) else ""
        metric_val = str(metric_val_raw).strip() if pd.notna(metric_val_raw) else ""

        if '년' in year_val:
            current_year_header = year_val.replace(" ", "")
        
        if current_year_header and metric_val:
            year_metric_cols.append(f"{current_year_header}_{metric_val}")
        elif metric_val : 
            year_metric_cols.append(f"_{metric_val}") 
        else:
            year_metric_cols.append(f"Unnamed_col_{i}")

    if len(year_metric_cols) != actual_data_metric_cols:
        if len(year_metric_cols) > actual_data_metric_cols:
            year_metric_cols = year_metric_cols[:actual_data_metric_cols]
        else: 
             year_metric_cols.extend([f"Unnamed_ext_{j}" for j in range(actual_data_metric_cols - len(year_metric_cols))])

    df_data_part = df_data_part.iloc[:, :num_base_cols + len(year_metric_cols)]
    df_data_part.columns = base_cols + year_metric_cols
    
    value_columns_pattern = ['환자수', '요양급여비용', '입내원일수']
    columns_to_clean = [col for col in df_data_part.columns if any(pat in str(col) for pat in value_columns_pattern)]

    for col in columns_to_clean:
        if col in df_data_part.columns:
            if df_data_part[col].dtype == 'object':
                df_data_part[col] = df_data_part[col].astype(str).str.replace(',', '', regex=False).str.strip()
                df_data_part[col] = pd.to_numeric(df_data_part[col], errors='coerce')
            df_data_part[col] = df_data_part[col].fillna(0).astype(int)

    df_seoul = df_data_part[df_data_part['시도'] == '서울'].copy()
    df_seoul['질환명'] = condition_name
    return df_seoul

@st.cache_data
def analyze_elderly_mental_condition_cached(df_seoul_condition, elderly_groups_param):
    if df_seoul_condition is None or df_seoul_condition.empty: return None, None, None
    df_elderly = df_seoul_condition[df_seoul_condition['연령구분'].isin(elderly_groups_param)].copy()
    if df_elderly.empty: return None, None, None
    id_vars_melt = ['시도', '시군구', '성별', '연령구분', '질환명']
    
    available_years_in_df = sorted(list(set([col.split('_')[0] for col in df_elderly.columns if '_환자수' in col and '년' in col])))
    value_vars_melt = [f"{year}_환자수" for year in available_years_in_df if f"{year}_환자수" in df_elderly.columns]

    if not value_vars_melt:
        condition_name_for_log = df_seoul_condition['질환명'].iloc[0] if not df_seoul_condition.empty and '질환명' in df_seoul_condition.columns else '알 수 없는 질환'
        # st.warning(f"{condition_name_for_log} 데이터에서 연도별 환자수 컬럼을 찾을 수 없습니다.") # UI 경고 최소화
        return None, None, None

    df_melted = df_elderly.melt(id_vars=id_vars_melt, value_vars=value_vars_melt, var_name='연도_항목', value_name='값')
    
    df_melted['연도'] = df_melted['연도_항목'].str.extract(r'(\d{4}년)')[0].str.replace('년', '').astype(int)
    df_melted['항목'] = '환자수'

    df_patients_data = df_melted

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
def aggregate_mental_patients_sigungu_total_cached(dataframes_by_condition, elderly_age_groups_param):
    all_mental_patients_list = []
    if not dataframes_by_condition: return pd.DataFrame()
    for condition_name, df_condition in dataframes_by_condition.items():
        if df_condition is None or df_condition.empty: continue
        df_elderly = df_condition[df_condition['연령구분'].isin(elderly_age_groups_param)].copy()
        if df_elderly.empty: continue
        
        id_vars = ['시군구', '질환명']
        available_years_in_df = sorted(list(set([col.split('_')[0] for col in df_elderly.columns if '_환자수' in col and '년' in col])))
        value_vars = [f"{year}_환자수" for year in available_years_in_df if f"{year}_환자수" in df_elderly.columns]

        if not value_vars: continue
            
        df_melted = df_elderly.melt(id_vars=id_vars, value_vars=value_vars, var_name='연도_항목', value_name='값')
        df_melted['연도'] = df_melted['연도_항목'].str.extract(r'(\d{4}년)')[0].str.replace('년', '').astype(int)
        
        sigungu_year_condition_total_patients = df_melted.groupby(
            ['연도', '시군구', '질환명']
        )['값'].sum().reset_index()
        sigungu_year_condition_total_patients.rename(columns={'값': '질환별_노인_환자수_총합'}, inplace=True)
        all_mental_patients_list.append(sigungu_year_condition_total_patients)
        
    if not all_mental_patients_list: return pd.DataFrame()
    return pd.concat(all_mental_patients_list).reset_index(drop=True)


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

    df_sigungu_mental_total_patients_for_tab3 = aggregate_mental_patients_sigungu_total_cached(
        dataframes_by_condition, 
        elderly_age_groups_mental
    )

    tab1_title = "개별 질환 분석"
    tab2_title = "정신질환 종합 비교"
    tab3_title = "구별 정신질환자 수"

    main_tab1, main_tab2, main_tab3 = st.tabs([tab1_title, tab2_title, tab3_title])

    with main_tab1:
        st.subheader(tab1_title)
        
        selectbox_options_tab1 = list(dataframes_by_condition.keys())
        if not selectbox_options_tab1: selectbox_options_tab1 = ["데이터 없음"]
        
        selected_condition_name_tab1 = st.selectbox(
            "분석할 질환을 선택하세요:", 
            selectbox_options_tab1, 
            key="mental_condition_select_tab1_page5_v4"
        )
        if selected_condition_name_tab1 != "데이터 없음" and dataframes_by_condition and selected_condition_name_tab1 in dataframes_by_condition:
            df_to_analyze = dataframes_by_condition[selected_condition_name_tab1]
            total_res, gender_res, subgroup_res = analyze_elderly_mental_condition_cached(
                df_to_analyze, 
                elderly_age_groups_mental
            )
            
            plot_type_tabs_main1 = st.tabs(["연도별 총계", "연도별 성별", "세부 연령대 및 성별"])
            with plot_type_tabs_main1[0]: 
                plot_total_elderly_trend(total_res, selected_condition_name_tab1)
            with plot_type_tabs_main1[1]: 
                plot_gender_elderly_trend(gender_res, selected_condition_name_tab1)
            with plot_type_tabs_main1[2]: 
                plot_subgroup_gender_elderly_trend(subgroup_res, selected_condition_name_tab1)
        elif not dataframes_by_condition:
            st.info("로드된 질환 데이터가 없습니다.")
        elif selected_condition_name_tab1 != "데이터 없음":
            st.info(f"'{selected_condition_name_tab1}'에 대한 데이터를 찾을 수 없습니다.")


    with main_tab2:
        current_selected_year_tab2 = st.session_state.selected_year_mental_tab2
        st.subheader(f"{current_selected_year_tab2}년 정신질환별 환자수 비교")

        selected_year_val_tab2 = st.slider(
            "조회 연도 선택", 
            min_value=min(available_years_for_mental_analysis), 
            max_value=max(available_years_for_mental_analysis),
            value=st.session_state.selected_year_mental_tab2, 
            step=1,
            key="mental_year_slider_tab2_page5_v4"
        )
        if st.session_state.selected_year_mental_tab2 != selected_year_val_tab2:
            st.session_state.selected_year_mental_tab2 = selected_year_val_tab2
            st.rerun()
        
        if all_conditions_summaries_list:
            all_conditions_summary_df_final = pd.concat(all_conditions_summaries_list).reset_index(drop=True)
            plot_all_conditions_yearly_comparison(all_conditions_summary_df_final, selected_year_val_tab2)
            plot_pie_chart_by_year(all_conditions_summary_df_final, selected_year_val_tab2)
        else: 
            st.info("정신질환 종합 비교를 위한 데이터가 충분하지 않습니다.")

    with main_tab3:
        current_selected_year_tab3 = st.session_state.selected_year_mental_tab3
        st.subheader(f"{current_selected_year_tab3}년 {tab3_title}") 

        selected_year_val_tab3 = st.slider(
            "조회 연도 선택",
            min_value=min(available_years_for_mental_analysis),
            max_value=max(available_years_for_mental_analysis),
            value=st.session_state.selected_year_mental_tab3,
            step=1,
            key="mental_year_slider_tab3_page5_v4"
        )
        if st.session_state.selected_year_mental_tab3 != selected_year_val_tab3:
            st.session_state.selected_year_mental_tab3 = selected_year_val_tab3
            st.rerun()
        
        selectbox_options_tab3 = list(dataframes_by_condition.keys())
        if not selectbox_options_tab3: selectbox_options_tab3 = ["데이터 없음"]
        
        selected_condition_tab3 = st.selectbox(
            "분석할 질환 선택:",
            selectbox_options_tab3,
            key="mental_condition_select_tab3_page5_v4"
        )
        
        if selected_condition_tab3 != "데이터 없음" and dataframes_by_condition and selected_condition_tab3 in dataframes_by_condition:
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
        elif selected_condition_tab3 != "데이터 없음":
             st.info(f"'{selected_condition_tab3}'에 대한 데이터를 찾을 수 없습니다.")

if __name__ == "__main__":
    run_mental_health_page()
# --- END OF 5_MentalHealth.py ---
