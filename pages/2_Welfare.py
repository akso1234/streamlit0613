# --- START OF 2_Welfare.py ---
import os # os 모듈은 파일 경로 확인에 필요하므로 유지
import streamlit as st
# data_loader.py가 프로젝트 루트에 있고, 이 파일이 pages 폴더에 있다면 아래와 같이 임포트합니다.
# 만약 data_loader.py도 pages 폴더에 있다면 from .data_loader import ...
# 현재 제공된 코드에서는 data_loader가 루트에 있다고 가정합니다.
from data_loader import (
    load_nursing_sheet0,
    load_nursing_sheet1,
    load_nursing_csv,
    load_nursing_sheet3,
    load_nursing_sheet4,
    load_nursing_sheet5
)
# chart_utils.py도 프로젝트 루트에 있다고 가정합니다.
from chart_utils import (
    draw_sheet0_charts,
    draw_sheet1_charts,
    draw_nursing_csv_charts,
    draw_sheet3_charts,
    draw_sheet4_charts,
    draw_sheet5_charts
)
# utils.py도 프로젝트 루트에 있다고 가정합니다.
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

    if "selected_year_welfare" not in st.session_state:
        st.session_state.selected_year_welfare = 2023

    selected_year = st.slider(
        label="조회 연도 선택",
        min_value=2019,
        max_value=2023,
        step=1,
        value=st.session_state.selected_year_welfare,
        key="welfare_slider_main_page" # 슬라이더 키를 명확히 구분
    )
    # 슬라이더 값이 변경되면 세션 상태를 업데이트하고 rerun을 유도할 수 있습니다.
    # Streamlit은 위젯 값 변경 시 자동으로 스크립트를 재실행하므로 명시적 rerun은 필요 없을 수 있습니다.
    st.session_state.selected_year_welfare = selected_year # 슬라이더 값으로 세션 상태 업데이트

    excel_path = f"data/{selected_year}nursing.xlsx"
    csv_path   = f"data/{selected_year}nursing.csv"

    missing = False
    if not os.path.isfile(excel_path):
        st.error(f"❌ {selected_year}년 엑셀 파일을 찾을 수 없습니다: {excel_path}")
        missing = True
    if not os.path.isfile(csv_path):
        st.error(f"❌ {selected_year}년 CSV 파일을 찾을 수 없습니다: {csv_path}")
        missing = True

    if missing:
        st.warning(f"올바른 {selected_year}년 파일을 `data/` 폴더에 배치했는지 확인하세요.")
        return

    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "주거복지시설",
        "의료복지시설",
        "여가복지시설",
        "재가노인복지시설",
        "노인일자리지원기관",
        "치매전담형장기요양"
    ])

    with tab0:
        st.subheader(f"{selected_year}년 주거복지시설 현황") # 탭 전체에 대한 연도별 부제목
        df0 = load_nursing_sheet0(excel_path, districts)
        if df0 is not None and not df0.empty:
            draw_sheet0_charts(df0, selected_year) # selected_year 전달
            # 상세 데이터 테이블 표시는 그대로 유지 가능
            st.markdown(f"--- \n #### {selected_year}년 상세 데이터 테이블")
            st.dataframe(df0.style.format("{:,.0f}", subset=df0.columns.difference(['cap_per_staff', 'occ_per_staff']))
                                 .format("{:,.1f}", subset=['cap_per_staff', 'occ_per_staff'])
                                 .set_properties(**{'text-align': 'right'}), 
                         use_container_width=True)
        else:
            st.info(f"{selected_year}년 주거복지시설 데이터를 불러오지 못했거나 데이터가 없습니다.")


    with tab1:
        st.subheader(f"{selected_year}년 의료복지시설 현황")
        df1 = load_nursing_sheet1(excel_path, districts)
        if df1 is not None and not df1.empty:
            draw_sheet1_charts(df1, selected_year) # selected_year 전달
            st.markdown(f"--- \n #### {selected_year}년 상세 데이터 테이블")
            st.dataframe(df1.style.format("{:,.0f}", subset=df1.columns.difference(['cap_per_staff', 'occ_per_staff']))
                                 .format("{:,.1f}", subset=['cap_per_staff', 'occ_per_staff'])
                                 .set_properties(**{'text-align': 'right'}), 
                         use_container_width=True)
        else:
            st.info(f"{selected_year}년 의료복지시설 데이터를 불러오지 못했거나 데이터가 없습니다.")


    with tab2:
        st.subheader(f"{selected_year}년 여가복지시설 현황")
        df_welf, df_centers = load_nursing_csv(csv_path, districts)
        # draw_nursing_csv_charts 함수가 내부적으로 df_welf와 df_centers의 None 또는 empty 여부를 확인하고,
        # 각 그래프에 대한 연도별 제목을 표시하도록 수정되었다고 가정합니다.
        draw_nursing_csv_charts(df_welf, df_centers, selected_year) # selected_year 전달

        st.markdown(f"---")
        if df_welf is not None and not df_welf.empty:
            st.markdown(f"#### {selected_year}년 노인복지관 상세 데이터 테이블")
            st.dataframe(df_welf.style.format("{:,.0f}").set_properties(**{'text-align': 'right'}), use_container_width=True)
        if df_centers is not None and not df_centers.empty:
            st.markdown(f"#### {selected_year}년 경로당 및 노인교실 상세 데이터 테이블")
            st.dataframe(df_centers.style.format("{:,.0f}").set_properties(**{'text-align': 'right'}), use_container_width=True)


    with tab3:
        st.subheader(f"{selected_year}년 재가노인복지시설 현황")
        df3 = load_nursing_sheet3(excel_path, districts)
        if df3 is not None and not df3.empty:
            draw_sheet3_charts(df3, selected_year) # selected_year 전달
            st.markdown(f"--- \n #### {selected_year}년 상세 데이터 테이블")
            st.dataframe(df3.style.format("{:,.0f}", subset=df3.columns.difference(['cap_per_staff', 'occ_per_staff']))
                                 .format("{:,.1f}", subset=['cap_per_staff', 'occ_per_staff'])
                                 .set_properties(**{'text-align': 'right'}), 
                         use_container_width=True)
        else:
            st.info(f"{selected_year}년 재가노인복지시설 데이터를 불러오지 못했거나 데이터가 없습니다.")

    with tab4:
        st.subheader(f"{selected_year}년 노인일자리지원기관 현황")
        df4 = load_nursing_sheet4(excel_path, districts)
        if df4 is not None and not df4.empty:
            draw_sheet4_charts(df4, selected_year) # selected_year 전달
            st.markdown(f"--- \n #### {selected_year}년 상세 데이터 테이블")
            st.dataframe(df4.style.format("{:,.0f}").set_properties(**{'text-align': 'right'}), use_container_width=True)
        else:
            st.info(f"{selected_year}년 노인일자리지원기관 데이터를 불러오지 못했거나 데이터가 없습니다.")

    with tab5:
        st.subheader(f"{selected_year}년 치매전담형장기요양기관 현황")
        df5 = load_nursing_sheet5(excel_path, districts)
        if df5 is not None and not df5.empty:
            draw_sheet5_charts(df5, selected_year) # selected_year 전달
            st.markdown(f"--- \n #### {selected_year}년 상세 데이터 테이블")
            st.dataframe(df5.style.format("{:,.0f}", subset=df5.columns.difference(['cap_per_staff', 'occ_per_staff']))
                                 .format("{:,.1f}", subset=['cap_per_staff', 'occ_per_staff'])
                                 .set_properties(**{'text-align': 'right'}), 
                         use_container_width=True)
        else:
            st.info(f"{selected_year}년 치매전담형장기요양기관 데이터를 불러오지 못했거나 데이터가 없습니다.")


if __name__ == "__main__":
    run_welfare_page()
# --- END OF 2_Welfare.py ---
