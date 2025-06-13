# pages/1_Hospital.py
import streamlit as st

# 1. st.set_page_config()를 스크립트의 가장 먼저 실행되는 Streamlit 명령으로 설정합니다.
st.set_page_config(
    page_title="서울시 병원 대시보드",
    page_icon="🏥",
    layout="wide"
)

# 2. 그 다음에 다른 모듈들을 임포트합니다.
# sys, os 임포트는 utils.py로 옮겨졌거나, 직접적인 경로 조작이 필요 없을 경우 제거 가능
# import sys
# import os
from utils import set_korean_font # utils.py 임포트
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

# 3. 폰트 설정을 수행합니다.
# 이 함수는 Matplotlib의 전역 폰트 설정을 처리하며,
# 내부적으로 st.sidebar.warning 등을 사용할 수 있으므로 set_page_config 이후에 호출합니다.
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
        key="hospital_year_slider" # 다른 페이지 슬라이더와 키 중복 방지
    )
    if selected_year != st.session_state.selected_year_hospital:
        st.session_state.selected_year_hospital = selected_year

    st.write(f"### 현재 선택된 연도: {selected_year}년")

    # geojson_path의 기본값을 data_loader.load_raw_data 함수 내에서 처리하도록 함
    df_hosp, df_beds, gdf_gu = load_raw_data(selected_year) 
    
    if df_hosp is None or df_beds is None or gdf_gu is None:
        st.error("필수 데이터(병원, 병상, 지리 정보) 로드에 실패하여 페이지를 표시할 수 없습니다.")
        return # 필수 데이터 없으면 실행 중단

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Choropleth (지도)", 
        "📊 자치구별 병원 수", 
        "🌡️ 평균 병상 수 히트맵",
        "🛏️ 전체 병원/병상 집계"
    ])

    with tab1:
        st.subheader(f"{selected_year}년 구별 의료기관 수 & 평균 병상 수 지도")
        
        # col1, col2 = st.columns(2) # 지도를 한 줄에 하나씩 표시하도록 변경 (가로 공간 확보)

        # with col1: # 첫 번째 지도
        st.markdown("##### **1) 구별 의료기관 수**")
        merged_counts = make_merged_counts(df_hosp, gdf_gu)
        if merged_counts is not None and not merged_counts.empty: # None 체크 추가
            m1 = draw_hospital_count_choropleth(merged_counts) 
            if m1: # 지도 객체가 정상적으로 생성되었는지 확인
                from streamlit_folium import st_folium 
                st_folium(m1, width=900, height=650, returned_objects=[]) # returned_objects 추가
        else:
            st.info("구별 의료기관 수 데이터가 없어 지도를 표시할 수 없습니다.")
        
        st.markdown("---") # 구분선

        # with col2: # 두 번째 지도
        st.markdown("##### **2) 구별 평균 병상 수**")
        merged_avg_beds_data = make_merged_avg_beds(df_hosp, df_beds, gdf_gu)
        if merged_avg_beds_data is not None and not merged_avg_beds_data.empty: # None 체크 추가
            m2 = draw_avg_beds_choropleth(merged_avg_beds_data)
            if m2: # 지도 객체가 정상적으로 생성되었는지 확인
                from streamlit_folium import st_folium
                st_folium(m2, width=900, height=650, returned_objects=[]) # returned_objects 추가
        else:
            st.info("구별 평균 병상 수 데이터가 없어 지도를 표시할 수 없습니다.")

    with tab2:
        st.subheader(f"🏥 {selected_year}년 구별 의료기관 수 막대그래프")
        if df_hosp is not None and not df_hosp.empty: # None 및 empty 체크
            draw_hospital_count_bar_charts(df_hosp)
        else:
            st.info("의료기관 수 데이터가 없어 막대 그래프를 그릴 수 없습니다.")

    with tab3: # 히트맵 탭
        st.subheader(f"🌡️ {selected_year}년 기관 유형별 평균 병상 수 히트맵") # 이모티콘 변경
        if df_hosp is not None and not df_hosp.empty and \
           df_beds is not None and not df_beds.empty:
            pivot_table_data = draw_avg_beds_heatmap(df_hosp, df_beds)
            if pivot_table_data is None or pivot_table_data.empty:
                 st.info("평균 병상 수 히트맵을 생성할 데이터가 부족합니다.")
            # else:
                 # st.dataframe(pivot_table_data.style.format("{:.1f}", na_rep="-").background_gradient(cmap='viridis_r'))
        else:
            st.info("평균 병상 수 관련 데이터가 없어 히트맵을 그릴 수 없습니다.")
    
    with tab4: # 전체 병원/병상 집계 탭
        st.subheader(f"🛏️ {selected_year}년 의료기관 유형별 전체 병원 및 병상 수") # 이모티콘 변경
        if df_hosp is not None and not df_hosp.empty and \
           df_beds is not None and not df_beds.empty:
            draw_aggregate_hospital_bed_charts(df_hosp, df_beds)
        else:
            st.info("전체 병원 또는 병상 수 데이터가 없어 집계 그래프를 그릴 수 없습니다.")


if __name__ == "__main__":
    run_hospital_page()
