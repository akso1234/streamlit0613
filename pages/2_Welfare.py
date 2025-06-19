# --- START OF 2_Welfare.py ---
import os
import streamlit as st
# data_loader.py, chart_utils.py, utils.py가 프로젝트 루트에 있고,
# 이 파일(2_Welfare.py)이 pages 폴더에 있다고 가정합니다.
# ImportError 발생 시 이전 답변의 sys.path.append() 코드 고려
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
import pandas as pd # 혹시 모를 경우를 위해 유지

set_korean_font()

districts = [
    "종로구","중구","용산구","성동구","광진구","동대문구","중랑구","성북구",
    "강북구","도봉구","노원구","은평구","서대문구","마포구","양천구","강서구",
    "구로구","금천구","영등포구","동작구","관악구","서초구","강남구","송파구","강동구"
]

def run_welfare_page():
    st.title("복지시설 관련 대시보드")

    if "selected_year_welfare" not in st.session_state:
        st.session_state.selected_year_welfare = 2023

    selected_year = st.slider( # 슬라이더를 페이지 상단으로 이동
        label="조회 연도 선택",
        min_value=2019,
        max_value=2023,
        step=1,
        value=st.session_state.selected_year_welfare,
        key="welfare_year_slider_top" # 키 변경
    )
    st.session_state.selected_year_welfare = selected_year # 슬라이더 값으로 세션 상태 업데이트

    excel_path = f"data/{selected_year}nursing.xlsx"
    csv_path   = f"data/{selected_year}nursing.csv"

    missing_excel = not os.path.isfile(excel_path)
    missing_csv = not os.path.isfile(csv_path)

    # 탭 구성 먼저
    tab_titles = [
        "주거복지시설", "의료복지시설", "여가복지시설",
        "재가노인복지시설", "노인일자리지원기관", "치매전담형장기요양"
    ]
    tabs = st.tabs(tab_titles)

    with tabs[0]: # 주거복지시설
        st.subheader(f"{selected_year}년 주거복지시설 현황")
        if missing_excel:
            st.error(f"데이터 파일을 찾을 수 없습니다: {excel_path}")
        else:
            df0 = load_nursing_sheet0(excel_path, districts)
            if df0 is not None and not df0.empty:
                draw_sheet0_charts(df0, selected_year)
            else:
                st.info(f"{selected_year}년 주거복지시설 데이터를 불러오지 못했거나 데이터가 없습니다.")

    with tabs[1]: # 의료복지시설
        st.subheader(f"{selected_year}년 의료복지시설 현황")
        if missing_excel:
            st.error(f"데이터 파일을 찾을 수 없습니다: {excel_path}")
        else:
            df1 = load_nursing_sheet1(excel_path, districts)
            if df1 is not None and not df1.empty:
                draw_sheet1_charts(df1, selected_year)
            else:
                st.info(f"{selected_year}년 의료복지시설 데이터를 불러오지 못했거나 데이터가 없습니다.")

    with tabs[2]: # 여가복지시설
        st.subheader(f"{selected_year}년 여가복지시설 현황")
        if missing_csv:
            st.error(f"데이터 파일을 찾을 수 없습니다: {csv_path}")
        else:
            df_welf, df_centers = load_nursing_csv(csv_path, districts)
            if (df_welf is not None and not df_welf.empty) or \
               (df_centers is not None and not df_centers.empty):
                draw_nursing_csv_charts(df_welf, df_centers, selected_year)
            else:
                st.info(f"{selected_year}년 여가복지시설 데이터를 불러오지 못했거나 데이터가 없습니다.")

    with tabs[3]: # 재가노인복지시설
        st.subheader(f"{selected_year}년 재가노인복지시설 현황")
        if missing_excel:
            st.error(f"데이터 파일을 찾을 수 없습니다: {excel_path}")
        else:
            df3 = load_nursing_sheet3(excel_path, districts)
            if df3 is not None and not df3.empty:
                draw_sheet3_charts(df3, selected_year)
            else:
                st.info(f"{selected_year}년 재가노인복지시설 데이터를 불러오지 못했거나 데이터가 없습니다.")

    with tabs[4]: # 노인일자리지원기관
        st.subheader(f"{selected_year}년 노인일자리지원기관 현황")
        if missing_excel:
            st.error(f"데이터 파일을 찾을 수 없습니다: {excel_path}")
        else:
            df4 = load_nursing_sheet4(excel_path, districts)
            if df4 is not None and not df4.empty:
                draw_sheet4_charts(df4, selected_year)
            else:
                st.info(f"{selected_year}년 노인일자리지원기관 데이터를 불러오지 못했거나 데이터가 없습니다.")

    with tabs[5]: # 치매전담형장기요양
        st.subheader(f"{selected_year}년 치매전담형장기요양기관 현황")
        if missing_excel:
            st.error(f"데이터 파일을 찾을 수 없습니다: {excel_path}")
        else:
            df5 = load_nursing_sheet5(excel_path, districts)
            if df5 is not None and not df5.empty:
                draw_sheet5_charts(df5, selected_year)
            else:
                st.info(f"{selected_year}년 치매전담형장기요양기관 데이터를 불러오지 못했거나 데이터가 없습니다.")


if __name__ == "__main__":
    run_welfare_page()
# --- END OF 2_Welfare.py ---
