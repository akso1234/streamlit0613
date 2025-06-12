# --- START OF pages/1_Hospital.py ---
import streamlit as st
import sys
import os

# 1. st.set_page_config()를 스크립트의 가장 먼저 실행되는 Streamlit 명령으로 설정합니다.
st.set_page_config(
    page_title="서울시 병원 대시보드",
    page_icon="🏥",
    layout="wide"
)

# 2. 디버깅 정보 출력 (ImportError 원인 파악용)
# Streamlit Cloud 로그에서 이 print문들의 출력을 확인하세요.
# print(f"--- DEBUG INFO from 1_Hospital.py (START) ---")
# print(f"Current Working Directory (from 1_Hospital.py): {os.getcwd()}")

# 현재 파일(1_Hospital.py)의 디렉토리: /mount/src/streamlit0613/pages
# 상위 디렉토리(프로젝트 루트): /mount/src/streamlit0613
# 이 경로가 sys.path에 있어야 utils.py 등을 찾을 수 있습니다.
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_path = os.path.abspath(os.path.join(current_file_dir, '..'))

# print(f"Calculated Project Root (from 1_Hospital.py): {project_root_path}")

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path) # sys.path의 가장 앞에 추가
    # print(f"Added to sys.path (from 1_Hospital.py): {project_root_path}")
# else:
    # print(f"Project root '{project_root_path}' is already in sys.path (from 1_Hospital.py).")
# print(f"Current sys.path (from 1_Hospital.py): {sys.path}")

# 프로젝트 루트에 utils.py가 있는지 확인
# utils_py_path_check = os.path.join(project_root_path, 'utils.py')
# if os.path.exists(utils_py_path_check):
#     print(f"'utils.py' FOUND at: {utils_py_path_check}")
# else:
#     print(f"'utils.py' NOT FOUND at: {utils_py_path_check}")
#     print(f"Files in project root ({project_root_path}): {os.listdir(project_root_path) if os.path.exists(project_root_path) else 'Cannot list files'}")
# print(f"--- DEBUG INFO from 1_Hospital.py (END) ---")


# 3. 이제 모듈 임포트를 시도합니다.
try:
    from utils import set_korean_font
    # print("DEBUG: Successfully imported set_korean_font from utils.") # 성공 시 터미널 로그
except ImportError as e:
    # 이 에러가 발생하면, 위의 디버깅 print문들의 출력을 Streamlit Cloud 로그에서 확인해야 합니다.
    # print(f"ERROR: Failed to import set_korean_font from utils: {e}") # 실패 시 터미널 로그
    st.error(f"모듈 임포트 오류: {e}. 'utils.py'의 위치를 확인하거나 Streamlit Cloud 로그를 확인하세요.")
    st.stop() # 임포트 실패 시 앱 실행 중지
except Exception as general_e:
    # print(f"ERROR: An unexpected error occurred during utils import: {general_e}") # 실패 시 터미널 로그
    st.error(f"예상치 못한 임포트 오류: {general_e}")
    st.stop()

from data_loader import load_raw_data
from map_utils import (
    make_merged_counts,
    make_merged_avg_beds,
    draw_hospital_count_choropleth,
    draw_avg_beds_choropleth,
)
from chart_utils import (
    draw_hospital_count_bar_charts,
    draw_aggregate_hospital_bed_charts,
    draw_avg_beds_heatmap
)

# 4. 폰트 설정을 수행합니다. (st.set_page_config 이후, 다른 st 명령어 사용 전에 적절)
# set_korean_font 함수 내에서 st.sidebar.warning 등이 호출될 수 있으므로,
# 다른 st 요소가 렌더링되기 전에 호출하는 것이 좋습니다.
set_korean_font()


def run_hospital_page():
    st.title("🏥 병원 관련 대시보드")

    if "selected_year_hospital" not in st.session_state:
        st.session_state.selected_year_hospital = 2023

    selected_year = st.slider(
        label="조회 연도 선택",
        min_value=2019,
        max_value=2023,
        step=1,
        value=st.session_state.selected_year_hospital,
        key="hospital_year_slider"
    )
    if selected_year != st.session_state.selected_year_hospital:
        st.session_state.selected_year_hospital = selected_year

    st.write(f"### 현재 선택된 연도: {selected_year}년")

    df_hosp, df_beds, gdf_gu = load_raw_data(selected_year)
    
    if df_hosp is None or df_beds is None or gdf_gu is None:
        # load_raw_data 함수 내에서 st.error를 이미 호출했을 수 있음
        # 여기서는 추가 메시지나 중복 st.error를 피하기 위해 간단히 return
        # st.error("필수 데이터 로드에 실패하여 페이지를 표시할 수 없습니다.") # data_loader에서 이미 처리
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Choropleth", 
        "📊 막대그래프", 
        "🛏️ 침상 그래프", 
        "🌡️ 히트맵"
    ])

    with tab1:
        st.subheader(f"{selected_year}년 구별 의료기관 수 & 평균 병상 수 Choropleth")
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**1) 구별 의료기관 수**")
            merged_counts = make_merged_counts(df_hosp, gdf_gu)
            if not merged_counts.empty:
                m1 = draw_hospital_count_choropleth(merged_counts)
                if m1:
                    from streamlit_folium import st_folium
                    st_folium(m1, width=700, height=500, returned_objects=[]) # returned_objects 추가
            else:
                st.info("구별 의료기관 수 데이터가 없어 지도를 표시할 수 없습니다.")

        with col2:
            st.markdown("**2) 구별 평균 병상 수**")
            merged_avg_beds_data = make_merged_avg_beds(df_hosp, df_beds, gdf_gu) # 변수명 수정
            if not merged_avg_beds_data.empty:
                m2 = draw_avg_beds_choropleth(merged_avg_beds_data)
                if m2:
                    from streamlit_folium import st_folium
                    st_folium(m2, width=700, height=500, returned_objects=[]) # returned_objects 추가
            else:
                st.info("구별 평균 병상 수 데이터가 없어 지도를 표시할 수 없습니다.")

    with tab2:
        st.subheader(f"🏥 {selected_year}년 구별 의료기관 수 막대그래프")
        if df_hosp is not None and not df_hosp.empty:
            draw_hospital_count_bar_charts(df_hosp)
        else:
            st.info("의료기관 수 데이터가 없어 막대 그래프를 그릴 수 없습니다.")

    with tab3:
        st.subheader(f"🏥 {selected_year}년 침상 수 그래프")
        if df_hosp is not None and not df_hosp.empty and \
           df_beds is not None and not df_beds.empty:
            draw_aggregate_hospital_bed_charts(df_hosp, df_beds)
        else:
            st.info("침상 관련 데이터가 없어 그래프를 그릴 수 없습니다.")

    with tab4:
        st.subheader(f"🏥 {selected_year}년 평균 병상 수 히트맵")
        if df_hosp is not None and not df_hosp.empty and \
           df_beds is not None and not df_beds.empty:
            pivot_table_data = draw_avg_beds_heatmap(df_hosp, df_beds) # 변수명 수정
            # if pivot_table_data is not None and not pivot_table_data.empty:
            #     st.dataframe(pivot_table_data.style.format("{:.1f}", na_rep="-").background_gradient(cmap='Blues'))
        else:
            st.info("평균 병상 수 관련 데이터가 없어 히트맵을 그릴 수 없습니다.")

if __name__ == "__main__":
    run_hospital_page()
# --- END OF pages/1_Hospital.py ---
