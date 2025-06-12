import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

st.title("한글 폰트 테스트")

# 폰트 설정 시도
font_found = False
font_path_linux = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(font_path_linux):
    try:
        font_name = fm.FontProperties(fname=font_path_linux).get_name()
        plt.rc('font', family=font_name)
        plt.rcParams['font.family'] = font_name
        font_found = True
        st.write(f"경로에서 폰트 설정 성공: {font_name}")
    except Exception as e:
        st.write(f"경로에서 폰트 설정 실패: {e}")

if not font_found:
    try:
        plt.rc('font', family='NanumGothic') # 이름으로 시도
        plt.rcParams['font.family'] = 'NanumGothic'
        font_found = True
        st.write("이름으로 'NanumGothic' 설정 성공")
    except Exception as e:
        st.write(f"이름으로 'NanumGothic' 설정 실패: {e}")

if not font_found:
    st.warning("한글 폰트를 찾지 못했습니다.")

plt.rcParams['axes.unicode_minus'] = False

# 그래프 그리기
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 1])
ax.set_title("한글 제목 테스트")
ax.set_xlabel("엑스축 라벨")
ax.set_ylabel("와이축 라벨")
st.pyplot(fig)

st.write(f"현재 Matplotlib 폰트: {plt.rcParams['font.family']}")

# 사용 가능한 폰트 목록 (일부) - 디버깅용
# try:
#     st.write("사용 가능한 시스템 폰트 (일부):")
#     fonts = [f.name for f in fm.fontManager.ttflist if 'nanum' in f.name.lower()]
#     st.write(fonts if fonts else "나눔 계열 폰트 없음")
# except Exception as e:
#     st.write(f"폰트 목록 가져오기 실패: {e}")
