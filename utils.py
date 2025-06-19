# 2_WelfareFacilities.py (또는 2_Welfare.py) 파일 상단
import streamlit as st
import sys # sys 모듈 임포트
import os # os 모듈 임포트

# 현재 파일의 디렉토리 기준으로 부모 디렉토리를 sys.path에 추가
# __file__은 현재 스크립트의 경로를 나타냅니다.
current_file_path = os.path.abspath(__file__)
# 현재 파일이 있는 디렉토리 (예: /mount/src/streamlit0613/pages)
current_dir = os.path.dirname(current_file_path)
# 부모 디렉토리 (예: /mount/src/streamlit0613)
parent_dir = os.path.dirname(current_dir)

# sys.path에 부모 디렉토리가 없다면 추가
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 이제 부모 디렉토리에 있는 모듈들을 임포트할 수 있습니다.
from utils import load_csv, load_excel_sheets, set_korean_font # 유틸리티 함수 로드
from data_processing import ( # 데이터 처리 함수 로드
    extract_sheet0_metrics, extract_sheet1_metrics,
    extract_nursing_csv_metrics, extract_sheet3_metrics,
    extract_sheet4_metrics, extract_sheet5_metrics
)
from chart_utils import ( # 차트 생성 함수 로드
    draw_sheet0_charts, draw_sheet1_charts,
    draw_nursing_csv_charts, draw_sheet3_charts,
    draw_sheet4_charts, draw_sheet5_charts
)
import pandas as pd # pandas 직접 사용이 필요한 경우

# ... 나머지 코드는 동일 ...

def run_welfare_facilities_page():
    set_korean_font() # 한글 폰트 설정
    st.title("🧓 서울시 노인 복지시설 현황")

    # --- 데이터 로드 ---
    excel_file_path = "data/서울시_노인복지시설.xlsx"
    csv_file_path = "data/서울시_노인여가복지시설(경로당, 노인교실, 노인복지관)_현황.csv"

    # 엑셀 파일 로드 (모든 시트)
    all_sheets_data = load_excel_sheets(excel_file_path)
    # CSV 파일 로드
    csv_data = load_csv(csv_file_path)

    # --- 연도 선택 슬라이더 ---
    available_years = [2020, 2021, 2022, 2023] # 사용 가능한 연도 목록
    if "selected_year_welfare" not in st.session_state:
        st.session_state.selected_year_welfare = available_years[-1] # 기본값: 가장 최근 연도

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


    # --- 데이터 처리 ---
    # 각 시트 및 CSV 데이터에 대해 연도별 데이터 추출
    df_sheet0 = extract_sheet0_metrics(all_sheets_data.get('0.노인주거복지시설'), selected_year)
    df_sheet1 = extract_sheet1_metrics(all_sheets_data.get('1.노인의료복지시설'), selected_year)
    
    # 여가복지시설(CSV) 데이터 처리
    df_welf_csv, df_centers_csv = extract_nursing_csv_metrics(csv_data, selected_year)

    df_sheet3 = extract_sheet3_metrics(all_sheets_data.get('3.재가노인복지시설'), selected_year)
    df_sheet4 = extract_sheet4_metrics(all_sheets_data.get('4.노인일자리지원기관'), selected_year)
    df_sheet5 = extract_sheet5_metrics(all_sheets_data.get('5.치매전담형 장기요양기관'), selected_year)


    # --- 탭 구성 ---
    tab_titles = [
        "주거복지시설", "의료복지시설", "여가복지시설(CSV)",
        "재가노인복지시설", "노인일자리지원기관", "치매전담형장기요양"
    ]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.subheader(f"{selected_year}년 노인주거복지시설 현황")
        if df_sheet0 is not None and not df_sheet0.empty:
            st.markdown(f"#### {selected_year}년 자치구별 정원·현원·추가 수용 가능 인원")
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
            st.markdown(f"#### {selected_year}년 자치구별 정원·현원·추가 수용 가능 인원")
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
            st.markdown(f"#### {selected_year}년 자치구별 노인복지관 현황")
            # draw_nursing_csv_charts 핸들러는 내부적으로 df_welf_csv와 df_centers_csv를 모두 받음
            # 여기서는 노인복지관 부분만 명시적으로 언급
            display_welf = True
        else:
            st.info(f"{selected_year}년 노인복지관(CSV) 데이터를 찾을 수 없습니다.")

        display_centers = False
        if df_centers_csv is not None and not df_centers_csv.empty:
            st.markdown(f"#### {selected_year}년 자치구별 경로당 및 노인교실 현황")
            # draw_nursing_csv_charts 핸들러는 내부적으로 df_welf_csv와 df_centers_csv를 모두 받음
            # 여기서는 경로당/노인교실 부분만 명시적으로 언급
            display_centers = True
        else:
            st.info(f"{selected_year}년 경로당 및 노인교실(CSV) 데이터를 찾을 수 없습니다.")
        
        if display_welf or display_centers:
            draw_nursing_csv_charts(df_welf_csv, df_centers_csv) # 차트 함수는 두 DF를 모두 받아 알아서 처리
        
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
            st.markdown(f"#### {selected_year}년 자치구별 정원·현원 인원수")
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
            st.markdown(f"#### {selected_year}년 자치구별 시설수 및 종사자수")
            draw_sheet4_charts(df_sheet4)
            st.markdown(f"---")
            st.markdown(f"#### {selected_year}년 상세 데이터 테이블")
            st.dataframe(df_sheet4.style.format("{:,.0f}").set_properties(**{'text-align': 'right'}), use_container_width=True)
        else:
            st.info(f"{selected_year}년 데이터를 찾을 수 없거나, 시트 '4.노인일자리지원기관'이 없습니다.")

    with tabs[5]:
        st.subheader(f"{selected_year}년 치매전담형 장기요양기관 현황")
        if df_sheet5 is not None and not df_sheet5.empty:
            st.markdown(f"#### {selected_year}년 자치구별 정원·현원·추가 수용 가능 인원")
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
