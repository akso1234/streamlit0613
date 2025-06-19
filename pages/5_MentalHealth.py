# --- START OF 5_MentalHealth.py ---
import streamlit as st
import pandas as pd
import os
import numpy as np

# sys.path 수정 코드를 제거합니다.
from utils import set_korean_font, load_csv
from chart_utils import (
    plot_total_elderly_trend,
    plot_gender_elderly_trend,
    plot_subgroup_gender_elderly_trend,
    plot_all_conditions_yearly_comparison,
    plot_pie_chart_by_year,
    plot_sigungu_mental_patients_by_condition_year, # 구별 '환자 수' 그래프
    plot_all_conditions_trend_lineplot,             # 서울시 전체 질환별 환자수 추이 (꺾은선)
    plot_elderly_population_ratio_trend_lineplot,   # 서울시 전체 질환별 환자비율 추이 (꺾은선)
    plot_sigungu_mental_patients_ratio_by_condition_year # 구별 질환별 환자비율 (막대)
)

@st.cache_data
def preprocess_mental_health_data_cached(file_path, condition_name):
    try:
        df_header_info = pd.read_csv(file_path, encoding='utf-8-sig', header=None, nrows=5)
        if df_header_info.shape[0] < 5:
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
        if '년' in year_val: current_year_header = year_val.replace(" ", "")
        if current_year_header and metric_val: year_metric_cols.append(f"{current_year_header}_{metric_val}")
        elif metric_val : year_metric_cols.append(f"_{metric_val}") 
        else: year_metric_cols.append(f"Unnamed_col_{i}")

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
    df_seoul = df_data_part[df_data_part['시도'] == '서울'].copy(); df_seoul['질환명'] = condition_name
    return df_seoul

@st.cache_data
def analyze_elderly_mental_condition_cached(df_seoul_condition, elderly_groups_param):
    if df_seoul_condition is None or df_seoul_condition.empty: return None, None, None
    df_elderly = df_seoul_condition[df_seoul_condition['연령구분'].isin(elderly_groups_param)].copy()
    if df_elderly.empty: return None, None, None
    id_vars_melt = ['시도', '시군구', '성별', '연령구분', '질환명']
    available_years_in_df = sorted(list(set([col.split('_')[0] for col in df_elderly.columns if '_환자수' in col and '년' in col])))
    value_vars_melt = [f"{year}_환자수" for year in available_years_in_df if f"{year}_환자수" in df_elderly.columns]
    if not value_vars_melt: return None, None, None
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

@st.cache_data
def load_and_preprocess_elderly_population_data(file_path):
    try:
        header_line1_raw = pd.read_csv(file_path, skiprows=0, nrows=1, header=None, encoding='utf-8-sig').iloc[0]
        header_line2_raw = pd.read_csv(file_path, skiprows=1, nrows=1, header=None, encoding='utf-8-sig').iloc[0]
        header_line3_raw = pd.read_csv(file_path, skiprows=2, nrows=1, header=None, encoding='utf-8-sig').iloc[0]
        header_line4_raw = pd.read_csv(file_path, skiprows=3, nrows=1, header=None, encoding='utf-8-sig').iloc[0]
        df_population_data_rows = pd.read_csv(file_path, skiprows=4, header=None, encoding='utf-8-sig', dtype=str)
    except FileNotFoundError:
        st.error(f"고령자 현황 파일을 찾을 수 없습니다: {file_path}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"고령자 현황 파일 로드 중 오류: {e}")
        return pd.DataFrame(), pd.DataFrame()

    multi_index_cols = []
    current_h1_year = ""
    current_h2_category = ""
    for i in range(df_population_data_rows.shape[1]):
        h1_val = str(header_line1_raw[i]).replace(" 년", "").strip() if i < len(header_line1_raw) and pd.notna(header_line1_raw[i]) else ''
        h2_val = str(header_line2_raw[i]).replace("(1)", "").replace("(2)", "").strip() if i < len(header_line2_raw) and pd.notna(header_line2_raw[i]) else ''
        h3_val = str(header_line3_raw[i]).strip() if i < len(header_line3_raw) and pd.notna(header_line3_raw[i]) else ''
        h4_val = str(header_line4_raw[i]).strip() if i < len(header_line4_raw) and pd.notna(header_line4_raw[i]) else ''
        if i == 0: multi_index_cols.append((h1_val, '_H2_', '_H3_', '_H4_')); current_h1_year = h1_val
        elif i == 1: multi_index_cols.append((current_h1_year, h2_val, '_H3_', '_H4_'))
        else:
            if h1_val and "Unnamed" not in h1_val and "동별" not in h1_val: current_h1_year = h1_val
            if h2_val and "Unnamed" not in h2_val and "동별" not in h2_val: current_h2_category = h2_val.replace(" ", "")
            multi_index_cols.append((current_h1_year, current_h2_category, h3_val, h4_val))
    df_multi_temp = pd.DataFrame(df_population_data_rows.values, columns=pd.MultiIndex.from_tuples(multi_index_cols))

    seoul_total_pop_data_list = []
    key_level0_col0 = df_multi_temp.columns[0]
    key_level0_col1 = df_multi_temp.columns[1]
    seoul_overall_row = df_multi_temp[(df_multi_temp[key_level0_col0].astype(str).str.strip() == '합계') & (df_multi_temp[key_level0_col1].astype(str).str.strip() == '소계')]
    if not seoul_overall_row.empty:
        for year_int in range(2019, 2024):
            year_str = str(year_int); pop_col_key = (year_str, '65세이상인구', '소계', '소계')
            elderly_pop_value = 0
            if pop_col_key in seoul_overall_row.columns:
                try:
                    value_raw = seoul_overall_row[pop_col_key].iloc[0]
                    if pd.notna(value_raw) and str(value_raw).strip() not in ['', '-']: elderly_pop_value = int(str(value_raw).replace(',', '').strip())
                except (ValueError, TypeError): pass
            seoul_total_pop_data_list.append({'연도': year_int, '총_65세_이상_노인인구': elderly_pop_value})
    df_seoul_total_elderly_pop = pd.DataFrame(seoul_total_pop_data_list) if seoul_total_pop_data_list else pd.DataFrame(columns=['연도', '총_65세_이상_노인인구'])

    df_sigungu_elderly_pop_list_total = []
    sigungu_data_rows = df_multi_temp[(df_multi_temp[key_level0_col0].astype(str).str.strip() == '합계') & (df_multi_temp[key_level0_col1].astype(str).str.strip() != '소계')].copy()
    if not sigungu_data_rows.empty:
        for index, row_data in sigungu_data_rows.iterrows():
            sigungu_name = row_data[key_level0_col1]
            if pd.isna(sigungu_name) or not str(sigungu_name).endswith('구'): continue
            for year_int in range(2019, 2024):
                year_str = str(year_int); pop_col_key = (year_str, '65세이상인구', '소계', '소계')
                total_elderly_val = 0
                if pop_col_key in row_data.index:
                    try:
                        value_raw = row_data[pop_col_key]
                        if pd.notna(value_raw) and str(value_raw).strip() not in ['', '-']: total_elderly_val = int(str(value_raw).replace(',', '').strip())
                    except (ValueError, TypeError): pass
                df_sigungu_elderly_pop_list_total.append({'시군구': sigungu_name, '연도': year_int, '총_65세_이상_노인인구': total_elderly_val})
    df_sigungu_total_elderly_pop = pd.DataFrame(df_sigungu_elderly_pop_list_total) if df_sigungu_elderly_pop_list_total else pd.DataFrame(columns=['시군구', '연도', '총_65세_이상_노인인구'])
    
    return df_seoul_total_elderly_pop, df_sigungu_total_elderly_pop

@st.cache_data
def calculate_seoul_total_mental_health_ratios(combined_year_df, seoul_total_elderly_pop_final):
    if combined_year_df.empty or seoul_total_elderly_pop_final.empty:
        return pd.DataFrame()
    df_merged_ratio_seoul_total = pd.merge(combined_year_df, seoul_total_elderly_pop_final, on='연도', how='left')
    if '총_65세_이상_노인인구' not in df_merged_ratio_seoul_total.columns:
        st.warning("서울시 전체 노인 인구 데이터 병합에 실패했습니다. ('총_65세_이상_노인인구' 컬럼 없음)")
        return pd.DataFrame()
        
    df_merged_ratio_seoul_total['총 서울 노인 인구'] = pd.to_numeric(df_merged_ratio_seoul_total['총_65세_이상_노인인구'], errors='coerce').fillna(0)
    df_merged_ratio_seoul_total['총 노인 환자수'] = pd.to_numeric(df_merged_ratio_seoul_total['총 노인 환자수'], errors='coerce').fillna(0)
    df_merged_ratio_seoul_total['노인 인구 대비 환자 비율 (%)'] = np.where(
        df_merged_ratio_seoul_total['총 서울 노인 인구'] > 0,
        (df_merged_ratio_seoul_total['총 노인 환자수'] / df_merged_ratio_seoul_total['총 서울 노인 인구']) * 100,
        0
    )
    return df_merged_ratio_seoul_total

@st.cache_data
def calculate_sigungu_mental_health_ratios(df_sigungu_mental_total_patients, df_sigungu_total_elderly_population):
    if df_sigungu_mental_total_patients.empty or df_sigungu_total_elderly_population.empty:
        return pd.DataFrame()
    df_sigungu_ratio_final = pd.merge(
        df_sigungu_mental_total_patients,
        df_sigungu_total_elderly_population,
        on=['연도', '시군구'],
        how='left'
    )
    if '총_65세_이상_노인인구' not in df_sigungu_ratio_final.columns:
        st.warning("구별 전체 노인 인구 데이터 병합에 실패했습니다. ('총_65세_이상_노인인구' 컬럼 없음)")
        df_sigungu_ratio_final['분모_전체_노인인구'] = 0 
    else:
        df_sigungu_ratio_final.rename(columns={'총_65세_이상_노인인구': '분모_전체_노인인구'}, inplace=True)
        df_sigungu_ratio_final['분모_전체_노인인구'] = pd.to_numeric(df_sigungu_ratio_final['분모_전체_노인인구'], errors='coerce').fillna(0)

    df_sigungu_ratio_final['질환별_노인_환자수_총합'] = pd.to_numeric(df_sigungu_ratio_final['질환별_노인_환자수_총합'], errors='coerce').fillna(0)
    
    if '분모_전체_노인인구' in df_sigungu_ratio_final.columns:
        df_sigungu_ratio_final['전체노인인구_대비_질환자_비율(%)'] = np.where(
            df_sigungu_ratio_final['분모_전체_노인인구'] > 0,
            (df_sigungu_ratio_final['질환별_노인_환자수_총합'] / df_sigungu_ratio_final['분모_전체_노인인구']) * 100,
            0
        )
    else:
        df_sigungu_ratio_final['전체노인인구_대비_질환자_비율(%)'] = 0

    return df_sigungu_ratio_final


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
    elderly_population_file_path = 'data/elderly_status_20250531210628.csv'
    
    elderly_age_groups_mental = ['60~69세', '70~79세', '80~89세', '90~99세', '100세 이상']
    available_years_for_mental_analysis = [2019, 2020, 2021, 2022, 2023]

    if "selected_year_mental_overall_v2" not in st.session_state:
        st.session_state.selected_year_mental_overall_v2 = available_years_for_mental_analysis[-1]
    if "selected_year_mental_sigungu_count" not in st.session_state:
        st.session_state.selected_year_mental_sigungu_count = available_years_for_mental_analysis[-1]
    if "selected_year_mental_sigungu_ratio_v2" not in st.session_state:
        st.session_state.selected_year_mental_sigungu_ratio_v2 = available_years_for_mental_analysis[-1]

    dataframes_by_condition = {}
    all_conditions_summaries_list = []
    at_least_one_mental_file_loaded = False

    for condition_key, file_path_value in file_paths_mental_health.items():
        if not os.path.exists(file_path_value):
            continue 
        df_condition_raw = preprocess_mental_health_data_cached(file_path_value, condition_key)
        if df_condition_raw is not None and not df_condition_raw.empty:
            dataframes_by_condition[condition_key] = df_condition_raw
            at_least_one_mental_file_loaded = True
            total_df, _, _ = analyze_elderly_mental_condition_cached(df_condition_raw, elderly_age_groups_mental)
            if total_df is not None and not total_df.empty:
                temp_summary = total_df.copy()
                temp_summary['질환명'] = condition_key
                all_conditions_summaries_list.append(temp_summary)

    if not at_least_one_mental_file_loaded:
        st.error("필수 정신질환 데이터 파일을 하나도 로드하지 못했습니다. 'data' 폴더 내용을 확인해주세요.")
        return

    df_seoul_total_elderly_pop, df_sigungu_total_elderly_pop = load_and_preprocess_elderly_population_data(elderly_population_file_path)

    combined_year_df_for_seoul = pd.DataFrame()
    if all_conditions_summaries_list:
        combined_year_df_for_seoul = pd.concat(all_conditions_summaries_list).reset_index(drop=True)
    
    df_merged_ratio_seoul_total_final = pd.DataFrame()
    if not combined_year_df_for_seoul.empty and not df_seoul_total_elderly_pop.empty:
        df_merged_ratio_seoul_total_final = calculate_seoul_total_mental_health_ratios(combined_year_df_for_seoul, df_seoul_total_elderly_pop)

    df_sigungu_mental_total_patients = aggregate_mental_patients_sigungu_total_cached(
        dataframes_by_condition, 
        elderly_age_groups_mental
    )
    df_sigungu_ratio_final_data = pd.DataFrame()
    if not df_sigungu_mental_total_patients.empty and not df_sigungu_total_elderly_pop.empty:
        df_sigungu_ratio_final_data = calculate_sigungu_mental_health_ratios(
            df_sigungu_mental_total_patients, 
            df_sigungu_total_elderly_pop
        )

    # --- Streamlit 페이지 탭 구성 ---
    tab_titles = [
        "개별 질환 분석 (서울시 전체)", 
        "서울시 전체 종합 비교",
        "구별 환자 수",
        "서울시 전체 비율 분석", 
        "구별 환자 비율 분석"
    ]
    main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs(tab_titles)

    with main_tab1:
        st.subheader(tab_titles[0])
        selectbox_options_tab1 = list(dataframes_by_condition.keys())
        if not selectbox_options_tab1: selectbox_options_tab1 = ["데이터 없음"]
        
        selected_condition_name_tab1 = st.selectbox(
            "분석할 질환을 선택하세요:", 
            selectbox_options_tab1, 
            key="mental_condition_select_tab1_page5_v6"
        )
        if selected_condition_name_tab1 != "데이터 없음" and dataframes_by_condition and selected_condition_name_tab1 in dataframes_by_condition:
            df_to_analyze = dataframes_by_condition[selected_condition_name_tab1]
            total_res, gender_res, subgroup_res = analyze_elderly_mental_condition_cached(
                df_to_analyze, 
                elderly_age_groups_mental
            )
            plot_type_tabs_main1 = st.tabs(["연도별 총계", "연도별 성별", "세부 연령대 및 성별"])
            with plot_type_tabs_main1[0]: plot_total_elderly_trend(total_res, selected_condition_name_tab1)
            with plot_type_tabs_main1[1]: plot_gender_elderly_trend(gender_res, selected_condition_name_tab1)
            with plot_type_tabs_main1[2]: plot_subgroup_gender_elderly_trend(subgroup_res, selected_condition_name_tab1)
        elif not dataframes_by_condition: st.info("로드된 질환 데이터가 없습니다.")
        elif selected_condition_name_tab1 != "데이터 없음": st.info(f"'{selected_condition_name_tab1}'에 대한 데이터를 찾을 수 없습니다.")

    with main_tab2: 
        st.subheader(tab_titles[1]) 
        selected_year_val_tab2 = st.slider(
            "조회 연도 선택 ", 
            min_value=min(available_years_for_mental_analysis), 
            max_value=max(available_years_for_mental_analysis),
            value=st.session_state.selected_year_mental_overall_v2, 
            step=1,
            key="mental_year_slider_overall_page5_v4"
        )
        if st.session_state.selected_year_mental_overall_v2 != selected_year_val_tab2:
            st.session_state.selected_year_mental_overall_v2 = selected_year_val_tab2
            st.rerun()
        
        st.markdown(f"#### {selected_year_val_tab2}년 정신질환별 환자수 비교") # Markdown으로 변경
        if not combined_year_df_for_seoul.empty:
            plot_all_conditions_yearly_comparison(combined_year_df_for_seoul, selected_year_val_tab2)
            plot_pie_chart_by_year(combined_year_df_for_seoul, selected_year_val_tab2)
        else: 
            st.info("정신질환 종합 비교를 위한 데이터가 충분하지 않습니다.")

    with main_tab3:
        current_selected_year_tab3 = st.session_state.selected_year_mental_sigungu_count
        st.subheader(f"{current_selected_year_tab3}년 {tab_titles[2]}") 

        selected_year_val_tab3 = st.slider(
            "조회 연도 선택  ", 
            min_value=min(available_years_for_mental_analysis),
            max_value=max(available_years_for_mental_analysis),
            value=st.session_state.selected_year_mental_sigungu_count,
            step=1,
            key="mental_year_slider_sigungu_count_page5_v4"
        )
        if st.session_state.selected_year_mental_sigungu_count != selected_year_val_tab3:
            st.session_state.selected_year_mental_sigungu_count = selected_year_val_tab3
            st.rerun()
        
        selectbox_options_tab3 = list(dataframes_by_condition.keys())
        if not selectbox_options_tab3: selectbox_options_tab3 = ["데이터 없음"]
        
        selected_condition_tab3 = st.selectbox(
            "분석할 질환 선택 ", 
            selectbox_options_tab3,
            key="mental_condition_select_sigungu_count_page5_v4"
        )
        
        if selected_condition_tab3 != "데이터 없음" and dataframes_by_condition and selected_condition_tab3 in dataframes_by_condition:
            if not df_sigungu_mental_total_patients.empty:
                plot_sigungu_mental_patients_by_condition_year(
                    df_sigungu_mental_total_patients, 
                    selected_condition_tab3, 
                    selected_year_val_tab3 
                )
            else:
                st.info("구별 정신질환자 수 집계 데이터를 생성하지 못했습니다.")
        elif not dataframes_by_condition: st.info("로드된 질환 데이터가 없습니다.")
        elif selected_condition_tab3 != "데이터 없음": st.info(f"'{selected_condition_tab3}'에 대한 데이터를 찾을 수 없습니다.")

    with main_tab4: # 서울시 전체 비율 분석 (신규 탭)
        st.subheader(tab_titles[3])
        if not combined_year_df_for_seoul.empty:
            st.markdown(f"#### 5대 정신질환별 노인 환자수 연도별 추이")
            plot_all_conditions_trend_lineplot(combined_year_df_for_seoul)
        else:
            st.info("서울시 전체 환자수 추이 데이터를 표시할 수 없습니다.")
        st.markdown("---")
        if not df_merged_ratio_seoul_total_final.empty:
            st.markdown(f"#### 연도별 정신질환별 노인 환자 비율 (전체 노인 인구 대비)")
            plot_elderly_population_ratio_trend_lineplot(df_merged_ratio_seoul_total_final)
        else:
            st.info("서울시 전체 노인 인구 대비 환자 비율 추이 데이터를 계산하거나 표시할 수 없습니다.")


    with main_tab5: # 구별 환자 비율 분석 (신규 탭)
        current_selected_year_sigungu_ratio = st.session_state.selected_year_mental_sigungu_ratio_v2
        st.subheader(f"{current_selected_year_sigungu_ratio}년 {tab_titles[4]}")

        selected_year_sigungu_ratio_val = st.slider(
            "조회 연도 선택   ", 
            min_value=min(available_years_for_mental_analysis),
            max_value=max(available_years_for_mental_analysis),
            value=st.session_state.selected_year_mental_sigungu_ratio_v2,
            step=1,
            key="mental_year_slider_sigungu_ratio_page5_v5"
        )
        if st.session_state.selected_year_mental_sigungu_ratio_v2 != selected_year_sigungu_ratio_val:
            st.session_state.selected_year_mental_sigungu_ratio_v2 = selected_year_sigungu_ratio_val
            st.rerun()

        selectbox_options_tab5 = list(dataframes_by_condition.keys())
        if not selectbox_options_tab5: selectbox_options_tab5 = ["데이터 없음"]

        selected_condition_sigungu_ratio = st.selectbox(
            "분석할 질환 선택   ", 
            selectbox_options_tab5,
            key="mental_condition_select_sigungu_ratio_page5_v5"
        )

        if selected_condition_sigungu_ratio != "데이터 없음" and dataframes_by_condition and selected_condition_sigungu_ratio in dataframes_by_condition:
            if not df_sigungu_ratio_final_data.empty:
                plot_sigungu_mental_patients_ratio_by_condition_year( # 이 함수 호출
                    df_sigungu_ratio_final_data,
                    selected_condition_sigungu_ratio,
                    selected_year_sigungu_ratio_val
                )
            else:
                st.info("구별 정신질환자 비율 데이터를 생성하지 못했습니다.")
        elif not dataframes_by_condition:
            st.info("로드된 질환 데이터가 없습니다.")
        elif selected_condition_sigungu_ratio != "데이터 없음":
            st.info(f"'{selected_condition_sigungu_ratio}'에 대한 데이터를 찾을 수 없습니다.")

if __name__ == "__main__":
    run_mental_health_page()
# --- END OF 5_MentalHealth.py ---
