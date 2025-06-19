# --- START OF 5_MentalHealth.py (단순화된 테스트 버전) ---
import streamlit as st
import os
import sys

# 경로 설정 코드 (이전과 동일)
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 여기까지만 두고, 아래 chart_utils 임포트는 잠시 주석 처리
# from chart_utils import (
#     plot_total_elderly_trend,
#     plot_gender_elderly_trend,
#     plot_subgroup_gender_elderly_trend,
#     plot_all_conditions_yearly_comparison,
#     plot_pie_chart_by_year,
#     plot_sigungu_mental_patients_by_condition_year
# )
# from utils import set_korean_font, load_csv # 이것도 잠시 주석 처리

st.write(f"Current working directory: {os.getcwd()}")
st.write(f"Parent directory added to sys.path: {parent_dir}")
st.write("sys.path entries:")
for p in sys.path:
    st.write(p)

# 여기서 chart_utils.py 파일이 실제로 parent_dir에 있는지 확인
chart_utils_path = os.path.join(parent_dir, "chart_utils.py")
st.write(f"Expected chart_utils.py path: {chart_utils_path}")
st.write(f"Does chart_utils.py exist at expected path? {os.path.exists(chart_utils_path)}")

# dummy 함수 정의 (실제 실행은 안 함)
def run_mental_health_page():
    st.title("Test Page")
    st.write("If you see this, basic Streamlit and path setup might be okay.")

if __name__ == "__main__":
    run_mental_health_page()

# --- END OF 5_MentalHealth.py (단순화된 테스트 버전) ---
