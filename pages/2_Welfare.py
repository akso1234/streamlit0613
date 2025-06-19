import os
import streamlit as st
from data_loader import (
    load_nursing_sheet0,
    load_nursing_sheet1,
    load_nursing_csv,
    load_nursing_sheet3,
    load_nursing_sheet4,
    load_nursing_sheet5
)
from chart_utils import (
    draw_sheet0_charts,
    draw_sheet1_charts,
    draw_nursing_csv_charts,
    draw_sheet3_charts,
    draw_sheet4_charts,
    draw_sheet5_charts
)

from utils import set_korean_font

set_korean_font()

# 복지시설 데이터를 읽을 때 필요한 'districts' 리스트
districts = [
    "종로구","중구","용산구","성동구","광진구","동대문구","중랑구","성북구",
    "강북구","도봉구","노원구","은평구","서대문구","마포구","양천구","강서구",
    "구로구","금천구","영등포구","동작구","관악구","서초구","강남구","송파구","강동구"
]

def run_welfare_page():
    st.title("복지시설 관련 대시보드")

    # ---------------------------------------------------
    # 1) 세션 상태에 'selected_year' 초기 값 설정
    # ---------------------------------------------------
    if "selected_year_welfare" not in st.session_state:
        st.session_state.selected_year_welfare = 2023  # 기본 연도

    # ---------------------------------------------------
    # 2) 메인 화면 상단에 '연도 선택' 슬라이더 배치
    # ---------------------------------------------------
    selected_year = st.slider(
        label="조회 연도 선택",
        min_value=2019,
        max_value=2023,
        step=1,
        value=st.session_state.selected_year_welfare
    )
    # 슬라이더를 움직일 때마다 세션 갱신
    if selected_year != st.session_state.selected_year_welfare:
        st.session_state.selected_year_welfare = selected_year

    # ---------------------------------------------------
    # 3) 연도별 파일 경로 생성 및 존재 여부 체크
    # ---------------------------------------------------
    excel_path = f"data/{selected_year}nursing.xlsx"
    csv_path   = f"data/{selected_year}nursing.csv"

    missing = False
    if not os.path.isfile(excel_path):
        st.error(f"❌ 엑셀 파일을 찾을 수 없습니다: {excel_path}")
        missing = True
    if not os.path.isfile(csv_path):
        st.error(f"❌ CSV 파일을 찾을 수 없습니다: {csv_path}")
        missing = True

    if missing:
        st.warning("올바른 연도별 파일을 `data/` 폴더에 배치했는지 확인하세요.")
        return

    # ---------------------------------------------------
    # 4) 탭(Tab) 구성: 총 6개 탭 생성
    # ---------------------------------------------------
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "주거복지시설",
        "의료복지시설",
        "여가복지시설",
        "재가노인복지시설",
        "노인일자리지원기관",
        "치매전담형장기요양"
    ])

    # ----------------------
    # Tab 0: 주거복지시설 (Sheet0)
    # ----------------------
    with tab0:
        df0 = load_nursing_sheet0(excel_path, districts)
        draw_sheet0_charts(df0)

    # ----------------------
    # Tab 1: 의료복지시설 (Sheet1)
    # ----------------------
    with tab1:
        df1 = load_nursing_sheet1(excel_path, districts)
        draw_sheet1_charts(df1)

    # ----------------------
    # Tab 2: 여가복지시설 (CSV)
    # ----------------------
    with tab2:
        df_welf, df_centers = load_nursing_csv(csv_path, districts)
        draw_nursing_csv_charts(df_welf, df_centers)

    # ----------------------
    # Tab 3: 재가노인복지시설 (Sheet3)
    # ----------------------
    with tab3:
        df3 = load_nursing_sheet3(excel_path, districts)
        draw_sheet3_charts(df3)

    # ----------------------
    # Tab 4: 노인일자리지원기관 (Sheet4)
    # ----------------------
    with tab4:
        df4 = load_nursing_sheet4(excel_path, districts)
        draw_sheet4_charts(df4)

    # ----------------------
    # Tab 5: 치매전담형장기요양 (Sheet5)
    # ----------------------
    with tab5:
        df5 = load_nursing_sheet5(excel_path, districts)
        draw_sheet5_charts(df5)


# Streamlit 멀티페이지 사용 시, 이 파일을 pages 폴더에 넣으면
# 사이드바에 “Welfare” 메뉴가 자동 생성됩니다.
if __name__ == "__main__":
    run_welfare_page()
