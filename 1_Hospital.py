# pages/1_Hospital.py
import streamlit as st
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

# Home.py에서 st.set_page_config를 이곳으로 옮깁니다.
# 앱 전체에 대한 설정을 정의합니다.
st.set_page_config(
    page_title="서울시 병원 대시보드", # 앱 탭에 표시될 제목
    page_icon="🏥",                 # 앱 탭에 표시될 아이콘
    layout="wide"                  # 페이지 레이아웃을 'wide'로 설정
)

def run_hospital_page():
    st.title("🏥 병원 관련 대시보드")

    # ---------------------------------------------------
    # 1) 세션 상태에 'selected_year' 초기값 설정
    # ---------------------------------------------------
    if "selected_year_hospital" not in st.session_state:
        st.session_state.selected_year_hospital = 2023  # 기본 연도
    # ---------------------------------------------------
    # 2) 메인 화면 상단에 '연도 선택' 슬라이더 배치
    # ---------------------------------------------------
    selected_year = st.slider(
        label="조회 연도 선택",
        min_value=2019,
        max_value=2023,
        step=1,
        value=st.session_state.selected_year_hospital,
        key="hospital_year_slider" # 다른 페이지 슬라이더와 키 중복 방지
    )
    # 슬라이더를 움직일 때마다 세션 상태 갱신
    if selected_year != st.session_state.selected_year_hospital:
        st.session_state.selected_year_hospital = selected_year

    st.write(f"### 현재 선택된 연도: {selected_year}년")

    # ---------------------------------------------------
    # 3) 선택된 연도 출력 및 데이터 로드
    # ---------------------------------------------------

    df_hosp, df_beds, gdf_gu = load_raw_data(selected_year)
    if df_hosp is None or gdf_gu is None:
        # CSV 또는 GeoJSON이 없으면 이후 코드 중단
        return

    # ---------------------------------------------------
    # 4) 탭(Tab) 구성: Choropleth / 막대그래프 / 침상 그래프 / 히트맵
    # ---------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Choropleth",
        "📊 막대그래프",
        "🛏️ 침상 그래프",
        "🌡️ 히트맵"
    ])

    # ----------------------
    # 탭 1: Choropleth
    # ----------------------
    with tab1:
        st.subheader(f"{selected_year}년 구별 의료기관 수 & 평균 병상 수 Choropleth")

        # (1) 구별 의료기관 수 Choropleth
        st.markdown("**1) 구별 의료기관 수**")
        merged_counts = make_merged_counts(df_hosp, gdf_gu)
        m1 = draw_hospital_count_choropleth(
            merged_counts,
            width=800,
            height=600
        )
        from streamlit_folium import folium_static # st_folium으로 대체 가능하면 그게 더 나을 수 있습니다.
        folium_static(m1, width=900, height=650)

        st.markdown("---")

        # (2) 구별 평균 병상 수 Choropleth
        st.markdown("**2) 구별 평균 병상 수**")
        merged_avg = make_merged_avg_beds(df_hosp, df_beds, gdf_gu)
        m2 = draw_avg_beds_choropleth(
            merged_avg,
            width=800,
            height=600
        )
        folium_static(m2, width=900, height=650)

    # ----------------------
    # 탭 2: 막대그래프
    # ----------------------
    with tab2:
        st.subheader(f"🏥 {selected_year}년 구별 의료기관 수 막대그래프")
        draw_hospital_count_bar_charts(df_hosp)

    # ----------------------
    # 탭 3: 침상 그래프
    # ----------------------
    with tab3:
        st.subheader(f"🏥 {selected_year}년 구별 침상 수 그래프")
        draw_aggregate_hospital_bed_charts(df_hosp, df_beds)

    # ----------------------
    # 탭 4: 히트맵
    # ----------------------
    with tab4:
        st.subheader(f"🏥 {selected_year}년 평균 병상 수 히트맵")
        _ = draw_avg_beds_heatmap(df_hosp, df_beds)


# Streamlit 멀티페이지 사용 시 이 파일을 pages 폴더에 배치합니다.
if __name__ == "__main__":
    run_hospital_page()