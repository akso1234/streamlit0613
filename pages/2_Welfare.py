# --- START OF 2_Welfare.py ---
import os
import streamlit as st
# data_loader, chart_utils, utils가 프로젝트 루트에 있다고 가정
# 만약 pages 폴더 밖에 있다면, 이전 답변처럼 sys.path.append()가 필요할 수 있으나,
# 원래 잘 작동했다면 이대로 두는 것이 맞습니다.
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
# import pandas as pd # pandas 직접 사용이 없다면 제거 가능

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

    selected_year = st.slider(
        label="조회 연도 선택",
        min_value=2019,
        max_value=2023,
        step=1,
        value=st.session_state.selected_year_welfare,
        key="welfare_slider_main_page_v2" # 슬라이더 키 변경
    )
    st.session_state.selected_year_welfare = selected_year

    excel_path = f"data/{selected_year}nursing.xlsx"
    csv_path   = f"data/{selected_year}nursing.csv"

    missing_excel = not os.path.isfile(excel_path)
    missing_csv = not os.path.isfile(csv_path)

    tab_titles = [
        "주거복지시설", "의료복지시설", "여가복지시설",
        "재가노인복지시설", "노인일자리지원기관", "치매전담형장기요양"
    ]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.subheader(f"{selected_year}년 주거복지시설 현황")
        if missing_excel:
            st.error(f"데이터 파일을 찾을 수 없습니다: {excel_path}")
        else:
            df0 = load_nursing_sheet0(excel_path, districts)
            if df0 is not None and not df0.empty:
                draw_sheet0_charts(df0, selected_year) # selected_year 전달
            else:
                st.info(f"{selected_year}년 주거복지시설 데이터를 불러오지 못했거나 데이터가 없습니다.")
        # 데이터 테이블 제거

    with tabs[1]:
        st.subheader(f"{selected_year}년 의료복지시설 현황")
        if missing_excel:
            st.error(f"데이터 파일을 찾을 수 없습니다: {excel_path}")
        else:
            df1 = load_nursing_sheet1(excel_path, districts)
            if df1 is not None and not df1.empty:
                draw_sheet1_charts(df1, selected_year) # selected_year 전달
            else:
                st.info(f"{selected_year}년 의료복지시설 데이터를 불러오지 못했거나 데이터가 없습니다.")
        # 데이터 테이블 제거

    with tabs[2]:
        st.subheader(f"{selected_year}년 여가복지시설 현황")
        if missing_csv:
            st.error(f"데이터 파일을 찾을 수 없습니다: {csv_path}")
        else:
            df_welf, df_centers = load_nursing_csv(csv_path, districts)
            # draw_nursing_csv_charts는 내부적으로 데이터 유무를 확인하고,
            # df_welf나 df_centers 중 하나라도 데이터가 있으면 해당 그래프를 그림
            draw_nursing_csv_charts(df_welf, df_centers, selected_year) # selected_year 전달
        # 데이터 테이블 제거

    with tabs[3]:
        st.subheader(f"{selected_year}년 재가노인복지시설 현황")
        if missing_excel:
            st.error(f"데이터 파일을 찾을 수 없습니다: {excel_path}")
        else:
            df3 = load_nursing_sheet3(excel_path, districts)
            if df3 is not None and not df3.empty:
                draw_sheet3_charts(df3, selected_year) # selected_year 전달
            else:
                st.info(f"{selected_year}년 재가노인복지시설 데이터를 불러오지 못했거나 데이터가 없습니다.")
        # 데이터 테이블 제거

    with tabs[4]:
        st.subheader(f"{selected_year}년 노인일자리지원기관 현황")
        if missing_excel:
            st.error(f"데이터 파일을 찾을 수 없습니다: {excel_path}")
        else:
            df4 = load_nursing_sheet4(excel_path, districts)
            if df4 is not None and not df4.empty:
                draw_sheet4_charts(df4, selected_year) # selected_year 전달
            else:
                st.info(f"{selected_year}년 노인일자리지원기관 데이터를 불러오지 못했거나 데이터가 없습니다.")
        # 데이터 테이블 제거

    with tabs[5]:
        st.subheader(f"{selected_year}년 치매전담형장기요양기관 현황")
        if missing_excel:
            st.error(f"데이터 파일을 찾을 수 없습니다: {excel_path}")
        else:
            df5 = load_nursing_sheet5(excel_path, districts)
            if df5 is not None and not df5.empty:
                draw_sheet5_charts(df5, selected_year) # selected_year 전달
            else:
                st.info(f"{selected_year}년 치매전담형장기요양기관 데이터를 불러오지 못했거나 데이터가 없습니다.")
        # 데이터 테이블 제거


if __name__ == "__main__":
    run_welfare_page()
# --- END OF 2_Welfare.py ---
