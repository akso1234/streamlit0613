# pages/1_Hospital.py
import streamlit as st

# 1. st.set_page_config()를 스크립트의 가장 먼저 실행되는 Streamlit 명령으로 설정합니다.
st.set_page_config(
    page_title="서울시 병원 대시보드", # 앱 탭에 표시될 제목 (Home.py에서 가져옴)
    page_icon="🏥",                 # 앱 탭 아이콘 (선택 사항)
    layout="wide"                  # 페이지 레이아웃 (Home.py에서 가져옴)
)

# 2. 그 다음에 다른 모듈들을 임포트합니다.
from utils import set_korean_font # 한글 폰트 설정을 위해 utils.py 임포트
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

# 3. 폰트 설정을 수행합니다. (st.set_page_config 이후)
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
        key="hospital_year_slider" # 원본 파일에 없었으나, 페이지별 슬라이더 구분을 위해 추가 권장
    )
    if selected_year != st.session_state.selected_year_hospital:
        st.session_state.selected_year_hospital = selected_year

    st.write(f"### 현재 선택된 연도: {selected_year}년")

    # geojson_path의 기본값은 data_loader.load_raw_data 함수 내에서 처리
    df_hosp, df_beds, gdf_gu = load_raw_data(selected_year) 
    
    if df_hosp is None or df_beds is None or gdf_gu is None :
        st.error("필수 데이터(병원, 병상, 지리 정보) 로드에 실패하여 페이지를 표시할 수 없습니다.")
        return

    # 원본 탭 구성 유지
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Choropleth", 
        "📊  지역구별 병원수 막대그래프",  # 원본 탭 이름 유지
        "🌡️ 평균 병상수 히트맵",       # 원본 탭 이름 유지
        "🛏️ 전체 병원, 병상 데이터 막대그래프" # 원본 탭 이름 유지
    ])

    with tab1:
        st.subheader(f"{selected_year}년 구별 의료기관 수 & 평균 병상 수 Choropleth") # 원본 제목 유지

        # 지도 표시 방식을 원본대로 한 줄에 하나씩으로 복원
        st.markdown("**1) 구별 의료기관 수**")
        merged_counts = make_merged_counts(df_hosp, gdf_gu)
        if merged_counts is not None and not merged_counts.empty:
            # draw_hospital_count_choropleth 함수는 width, height 인자를 받음 (원본 유지)
            m1 = draw_hospital_count_choropleth(merged_counts, width=800, height=600) 
            if m1:
                from streamlit_folium import st_folium # folium_static 대신 st_folium 권장
                st_folium(m1, width=900, height=650, returned_objects=[]) # returned_objects 추가
        else:
            st.info("구별 의료기관 수 데이터가 없어 지도를 표시할 수 없습니다.")
        
        st.markdown("---")

        st.markdown("**2) 구별 평균 병상 수**")
        merged_avg = make_merged_avg_beds(df_hosp, df_beds, gdf_gu) # 변수명 원본과 유사하게
        if merged_avg is not None and not merged_avg.empty:
            m2 = draw_avg_beds_choropleth(merged_avg, width=800, height=600)
            if m2:
                from streamlit_folium import st_folium
                st_folium(m2, width=900, height=650, returned_objects=[])
        else:
            st.info("구별 평균 병상 수 데이터가 없어 지도를 표시할 수 없습니다.")

    with tab2: # 원본 "📊 막대그래프" 탭
        st.subheader(f"🏥 {selected_year}년 구별 의료기관 수 막대그래프") # 원본 제목 유지
        if df_hosp is not None and not df_hosp.empty:
            draw_hospital_count_bar_charts(df_hosp)
        else:
            st.info("의료기관 수 데이터가 없어 막대 그래프를 그릴 수 없습니다.")

    with tab3: # 원본 "🌡️ 히트맵" 탭
        st.subheader(f"🏥 {selected_year}년 평균 병상 수 히트맵") # 원본 제목 유지
        if df_hosp is not None and not df_hosp.empty and \
           df_beds is not None and not df_beds.empty:
            draw_avg_beds_heatmap(df_hosp, df_beds) # 반환값 _ 사용은 원본 유지
        else:
            st.info("평균 병상 수 관련 데이터가 없어 히트맵을 그릴 수 없습니다.")
    
    with tab4: # 원본 "🛏️ 침상 그래프" 탭 (실제로는 전체 병원/병상 데이터)
        st.subheader(f"🏥 {selected_year}년 구별 전체 병원, 병상 그래프") # 원본 제목 유지
        if df_hosp is not None and not df_hosp.empty and \
           df_beds is not None and not df_beds.empty:
            draw_aggregate_hospital_bed_charts(df_hosp, df_beds)
        else:
            st.info("전체 병원 또는 병상 수 데이터가 없어 집계 그래프를 그릴 수 없습니다.")


if __name__ == "__main__":
    run_hospital_page()
