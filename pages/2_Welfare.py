import streamlit as st
# 만약 utils, data_processing, chart_utils가 pages 폴더 밖에 있다면, 
# 이전 답변에서 제안한 sys.path 수정 코드가 여기에 필요합니다.
# 예시:
# import sys
# import os
# current_file_path = os.path.abspath(__file__)
# current_dir = os.path.dirname(current_file_path)
# parent_dir = os.path.dirname(current_dir)
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

from utils import load_csv, load_excel_sheets, set_korean_font
from data_processing import (
    extract_sheet0_metrics, extract_sheet1_metrics,
    extract_nursing_csv_metrics, extract_sheet3_metrics,
    extract_sheet4_metrics, extract_sheet5_metrics
)
from chart_utils import (
    draw_sheet0_charts, draw_sheet1_charts,
    draw_nursing_csv_charts, draw_sheet3_charts,
    draw_sheet4_charts, draw_sheet5_charts
)
import pandas as pd

def run_welfare_facilities_page():
    set_korean_font()
    st.title("🧓 서울시 노인 복지시설 현황")

    excel_file_path = "data/서울시_노인복지시설.xlsx"
    csv_file_path = "data/서울시_노인여가복지시설(경로당, 노인교실, 노인복지관)_현황.csv"

    all_sheets_data = load_excel_sheets(excel_file_path)
    csv_data = load_csv(csv_file_path)

    available_years = [2020, 2021, 2022, 2023]
    if "selected_year_welfare" not in st.session_state:
        st.session_state.selected_year_welfare = available_years[-1]

    selected_year = st.sidebar.slider(
        "조회 연도 선택",
        min_value=min(available_years),
        max_value=max(available_years),
        value=st.session_state.selected_year_welfare,
        step=1,
        key="welfare_year_slider"
    )
    st.session_state.selected_year_welfare = selected_year
    st.sidebar.info(f"선택된 연도: **{selected_year}년**")

    df_sheet0 = extract_sheet0_metrics(all_sheets_data.get('0.노인주거복지시설'), selected_year)
    df_sheet1 = extract_sheet1_metrics(all_sheets_data.get('1.노인의료복지시설'), selected_year)
    df_welf_csv, df_centers_csv = extract_nursing_csv_metrics(csv_data, selected_year)
    df_sheet3 = extract_sheet3_metrics(all_sheets_data.get('3.재가노인복지시설'), selected_year)
    df_sheet4 = extract_sheet4_metrics(all_sheets_data.get('4.노인일자리지원기관'), selected_year)
    df_sheet5 = extract_sheet5_metrics(all_sheets_data.get('5.치매전담형 장기요양기관'), selected_year)

    tab_titles = [
        "주거복지시설", "의료복지시설", "여가복지시설(CSV)",
        "재가노인복지시설", "노인일자리지원기관", "치매전담형장기요양"
    ]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.subheader(f"{selected_year}년 노인주거복지시설 현황")
        if df_sheet0 is not None and not df_sheet0.empty:
            # chart_utils의 draw_sheet0_charts 함수가 내부적으로 정렬 및 여러 그래프를 그림
            draw_sheet0_charts(df_sheet0)
            st.markdown(f"---")
            st.markdown(f"#### {selected_year}년 상세 데이터 테이블")
            st.dataframe(df_sheet0.style.format("{:,.0f}", subset=pd.IndexSlice[:, df_sheet0.columns.difference(['cap_per_staff', 'occ_per_staff'])])
                                     .format("{:,.1f}", subset=['cap_per_staff', 'occ_per_staff'])
                                     .set_properties(**{'text-align': 'right'}),
                         use_container_width=True)
        else:
            st.info(f"{selected_year}년 데이터를 찾을 수 없거나, 시트 '0.노인주거복지시설'이 없습니다.")

    with tabs[1]:
        st.subheader(f"{selected_year}년 노인의료복지시설 현황")
        if df_sheet1 is not None and not df_sheet1.empty:
            draw_sheet1_charts(df_sheet1)
            st.markdown(f"---")
            st.markdown(f"#### {selected_year}년 상세 데이터 테이블")
            st.dataframe(df_sheet1.style.format("{:,.0f}", subset=pd.IndexSlice[:, df_sheet1.columns.difference(['cap_per_staff', 'occ_per_staff'])])
                                     .format("{:,.1f}", subset=['cap_per_staff', 'occ_per_staff'])
                                     .set_properties(**{'text-align': 'right'}),
                         use_container_width=True)
        else:
            st.info(f"{selected_year}년 데이터를 찾을 수 없거나, 시트 '1.노인의료복지시설'이 없습니다.")

    with tabs[2]:
        st.subheader(f"{selected_year}년 노인여가복지시설(CSV) 현황")
        display_welf = False
        if df_welf_csv is not None and not df_welf_csv.empty:
            # 부제목은 chart_utils 함수 내부에서 처리하거나, 여기서 추가적인 st.markdown을 사용할 수 있음
            # draw_nursing_csv_charts 함수가 첫 번째 그래프에 대한 제목을 이미 포함하고 있다고 가정
            display_welf = True
        else:
            st.info(f"{selected_year}년 노인복지관(CSV) 데이터를 찾을 수 없습니다.")

        display_centers = False
        if df_centers_csv is not None and not df_centers_csv.empty:
            # 두 번째 그래프에 대한 제목도 chart_utils에서 처리한다고 가정
            display_centers = True
        else:
            st.info(f"{selected_year}년 경로당 및 노인교실(CSV) 데이터를 찾을 수 없습니다.")
        
        if display_welf or display_centers:
            # draw_nursing_csv_charts 함수가 각 그래프 위에 적절한 연도별 제목을 표시하도록 수정되었다고 가정
            draw_nursing_csv_charts(df_welf_csv, df_centers_csv)
        
        st.markdown(f"---")
        if display_welf:
            st.markdown(f"#### {selected_year}년 노인복지관 상세 데이터 테이블")
            st.dataframe(df_welf_csv.style.format("{:,.0f}").set_properties(**{'text-align': 'right'}), use_container_width=True)
        if display_centers:
            st.markdown(f"#### {selected_year}년 경로당 및 노인교실 상세 데이터 테이블")
            st.dataframe(df_centers_csv.style.format("{:,.0f}").set_properties(**{'text-align': 'right'}), use_container_width=True)

    with tabs[3]:
        st.subheader(f"{selected_year}년 재가노인복지시설 현황")
        if df_sheet3 is not None and not df_sheet3.empty:
            draw_sheet3_charts(df_sheet3)
            st.markdown(f"---")
            st.markdown(f"#### {selected_year}년 상세 데이터 테이블")
            st.dataframe(df_sheet3.style.format("{:,.0f}", subset=pd.IndexSlice[:, df_sheet3.columns.difference(['cap_per_staff', 'occ_per_staff'])])
                                     .format("{:,.1f}", subset=['cap_per_staff', 'occ_per_staff'])
                                     .set_properties(**{'text-align': 'right'}),
                         use_container_width=True)
        else:
            st.info(f"{selected_year}년 데이터를 찾을 수 없거나, 시트 '3.재가노인복지시설'이 없습니다.")

    with tabs[4]:
        st.subheader(f"{selected_year}년 노인일자리지원기관 현황")
        if df_sheet4 is not None and not df_sheet4.empty:
            draw_sheet4_charts(df_sheet4)
            st.markdown(f"---")
            st.markdown(f"#### {selected_year}년 상세 데이터 테이블")
            st.dataframe(df_sheet4.style.format("{:,.0f}").set_properties(**{'text-align': 'right'}), use_container_width=True)
        else:
            st.info(f"{selected_year}년 데이터를 찾을 수 없거나, 시트 '4.노인일자리지원기관'이 없습니다.")

    with tabs[5]:
        st.subheader(f"{selected_year}년 치매전담형 장기요양기관 현황")
        if df_sheet5 is not None and not df_sheet5.empty:
            draw_sheet5_charts(df_sheet5)
            st.markdown(f"---")
            st.markdown(f"#### {selected_year}년 상세 데이터 테이블")
            st.dataframe(df_sheet5.style.format("{:,.0f}", subset=pd.IndexSlice[:, df_sheet5.columns.difference(['cap_per_staff', 'occ_per_staff'])])
                                     .format("{:,.1f}", subset=['cap_per_staff', 'occ_per_staff'])
                                     .set_properties(**{'text-align': 'right'}),
                         use_container_width=True)
        else:
            st.info(f"{selected_year}년 데이터를 찾을 수 없거나, 시트 '5.치매전담형 장기요양기관'이 없습니다.")


if __name__ == "__main__":
    run_welfare_facilities_page()
