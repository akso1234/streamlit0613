# --- START OF utils.py ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import json
import requests
import io

def set_korean_font():
    """Matplotlib에서 한글을 사용하기 위한 폰트 설정을 수행합니다."""
    font_found = False
    
    # print("DEBUG (utils.py): set_korean_font() called.") # 함수 호출 확인 로그

    # Streamlit Cloud 환경을 위한 시도 (packages.txt로 fonts-nanum* 설치 가정)
    # 1. 알려진 나눔 폰트 이름으로 직접 설정 시도
    nanum_font_names = ['NanumGothic', 'NanumBarunGothic', 'NanumSquare'] 
    
    try:
        # fm._rebuild() # 캐시 재빌드는 일단 주석 처리하고 테스트
        # print("DEBUG (utils.py): Font cache rebuild SKIPPED for now.")
        
        all_system_fonts = [f.name for f in fm.fontManager.ttflist]
        # print(f"DEBUG (utils.py): All system fonts found by fontManager: {len(all_system_fonts)}")
        # for i, f_name in enumerate(all_system_fonts): # 너무 많으면 일부만 출력
        #     if i < 20 or 'Nanum' in f_name:
        #         print(f"DEBUG (utils.py): Font {i}: {f_name}")

        for font_name in nanum_font_names:
            if font_name in all_system_fonts:
                plt.rc('font', family=font_name)
                plt.rcParams['font.family'] = font_name
                font_found = True
                # print(f"DEBUG (utils.py): Successfully set font to '{font_name}' by name.")
                break
            # else:
                # print(f"DEBUG (utils.py): Font '{font_name}' not found in system font list.")
                
    except Exception as e:
        # print(f"DEBUG (utils.py): Error during font name check or rc setting: {e}")
        pass

    # 2. 경로 기반 시도 (만약 이름으로 못 찾았을 경우, 또는 특정 경로 보장 시)
    if not font_found:
        nanum_font_paths_linux = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
        ]
        for font_path in nanum_font_paths_linux:
            if os.path.exists(font_path):
                try:
                    # 폰트 매니저에 직접 추가하는 것이 도움이 될 수 있음
                    font_entry = fm.FontEntry(fname=font_path, name=os.path.splitext(os.path.basename(font_path))[0]) # 이름도 지정
                    fm.fontManager.ttflist.append(font_entry) # 주의: ttflist 직접 조작
                    # fm.fontManager.addfont(font_path) # 더 안전한 방법일 수 있으나, rebuild 필요할 수 있음

                    font_prop_name = fm.FontProperties(fname=font_path).get_name()
                    plt.rc('font', family=font_prop_name)
                    plt.rcParams['font.family'] = font_prop_name
                    font_found = True
                    # print(f"DEBUG (utils.py): Successfully set font to '{font_prop_name}' from path: {font_path}")
                    break
                except Exception as e:
                    # print(f"DEBUG (utils.py): Failed to set font from path '{font_path}': {e}")
                    pass
    
    # 로컬 환경에 대한 폴백 (Streamlit Cloud에서는 위의 로직이 성공해야 함)
    if not font_found:
        # print("DEBUG (utils.py): Nanum fonts not found by name or specific path. Trying local OS fallbacks.")
        if os.name == 'nt': # Windows
            if 'Malgun Gothic' in [f.name for f in fm.fontManager.ttflist]:
                plt.rc('font', family='Malgun Gothic'); plt.rcParams['font.family'] = 'Malgun Gothic'; font_found = True
        elif os.name == 'posix' and "darwin" in os.uname().sysname.lower(): # macOS
            if 'AppleGothic' in [f.name for f in fm.fontManager.ttflist]:
                plt.rc('font', family='AppleGothic'); plt.rcParams['font.family'] = 'AppleGothic'; font_found = True

    if not font_found:
        st.sidebar.warning(
             "한글 폰트(예: NanumGothic)를 시스템에서 찾거나 설정할 수 없습니다. "
             "그래프의 한글이 깨질 수 있습니다. "
             "배포 환경에서는 'packages.txt'에 'fonts-nanum*'이 포함되어 있는지, "
             "앱 재부팅 후에도 문제가 지속되는지 확인해주세요."
        )
        # 기본 폰트(DejaVu Sans 등)로 설정되어 한글이 깨지게 됨
        # plt.rc('font', family='sans-serif') # 이 줄은 오히려 DejaVu Sans를 강제할 수 있음
        # plt.rcParams['font.family'] = 'sans-serif'
        # print(f"DEBUG (utils.py): Korean font not found. Current font: {plt.rcParams['font.family']}")
    # else:
        # print(f"DEBUG (utils.py): Final Matplotlib font family: {plt.rcParams['font.family']}")
    
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지

# --- load_csv, load_geojson, load_csv_from_upload 함수는 이전과 동일 ---
@st.cache_data
def load_csv(
    file_path, 
    encoding_options=['utf-8-sig', 'utf-8', 'cp949', 'euc-kr'], 
    header_config=None, 
    skiprows_config=None, 
    nrows_config=None, 
    na_values_config=None,
    sep_config=','
):
    full_path = file_path
    
    if not os.path.exists(full_path):
        # print(f"ERROR (utils.py): 데이터 파일을 찾을 수 없습니다: {full_path}")
        st.error(f"데이터 파일을 찾을 수 없습니다: {full_path}") # 사용자에게 알림
        return None

    for encoding in encoding_options:
        try:
            read_options = {'encoding': encoding, 'sep': sep_config}
            if header_config is not None:
                read_options['header'] = header_config
            if skiprows_config is not None:
                read_options['skiprows'] = skiprows_config
            if nrows_config is not None:
                read_options['nrows'] = nrows_config
            if na_values_config is not None:
                read_options['na_values'] = na_values_config
            
            df = pd.read_csv(full_path, **read_options)
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            # print(f"WARNING (utils.py): '{os.path.basename(full_path)}' 파일 로드 중 오류 (인코딩: {encoding}): {e}")
            st.warning(f"'{os.path.basename(full_path)}' 파일 로드 중 오류 (인코딩: {encoding}): {e}")
            return None
            
    # print(f"ERROR (utils.py): '{os.path.basename(full_path)}' 파일 로드 실패. 지원되는 인코딩을 찾을 수 없습니다.")
    st.error(f"'{os.path.basename(full_path)}' 파일 로드 실패. 지원되는 인코딩을 찾을 수 없습니다.")
    return None

@st.cache_data
def load_geojson(path_or_url):
    try:
        if path_or_url.startswith('http'):
            response = requests.get(path_or_url)
            response.raise_for_status()
            return response.json()
        else:
            full_path = path_or_url
            if not os.path.exists(full_path):
                # print(f"ERROR (utils.py): GeoJSON 파일을 찾을 수 없습니다: {full_path}")
                st.error(f"GeoJSON 파일을 찾을 수 없습니다: {full_path}")
                return None
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except requests.exceptions.RequestException as e:
        # print(f"ERROR (utils.py): GeoJSON URL에서 데이터 로드 중 오류: {e}")
        st.error(f"GeoJSON URL에서 데이터 로드 중 오류: {e}")
        return None
    except FileNotFoundError:
        # print(f"ERROR (utils.py): GeoJSON 파일을 찾을 수 없습니다: {path_or_url}")
        st.error(f"GeoJSON 파일을 찾을 수 없습니다: {path_or_url}")
        return None
    except json.JSONDecodeError as e:
        # print(f"ERROR (utils.py): GeoJSON 파일 파싱 중 오류: {e}")
        st.error(f"GeoJSON 파일 파싱 중 오류: {e}")
        return None
    except Exception as e:
        # print(f"ERROR (utils.py): GeoJSON 로드 중 예기치 않은 오류 발생: {e}")
        st.error(f"GeoJSON 로드 중 예기치 않은 오류 발생: {e}")
        return None

@st.cache_data
def load_csv_from_upload(
    _uploaded_file_object,
    encoding_options=['utf-8-sig', 'utf-8', 'cp949', 'euc-kr'], 
    header_config=None, 
    skiprows_config=None, 
    na_values_config=None,
    sep_config=','
):
    if _uploaded_file_object is None:
        return None
        
    uploaded_file_object = _uploaded_file_object 
    
    for encoding in encoding_options:
        try:
            bytes_data = uploaded_file_object.getvalue()
            
            read_options = {'encoding': encoding, 'sep': sep_config}
            if header_config is not None: read_options['header'] = header_config
            if skiprows_config is not None: read_options['skiprows'] = skiprows_config
            if na_values_config is not None: read_options['na_values'] = na_values_config

            df = pd.read_csv(io.BytesIO(bytes_data), **read_options)
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            # print(f"WARNING (utils.py): 업로드된 파일 '{uploaded_file_object.name}' 로드 중 오류 (인코딩: {encoding}): {e}")
            st.warning(f"업로드된 파일 '{uploaded_file_object.name}' 로드 중 오류 (인코딩: {encoding}): {e}")
            return None
            
    # print(f"ERROR (utils.py): 업로드된 파일 '{uploaded_file_object.name}' 로드 실패. 모든 인코딩 시도 실패.")
    st.error(f"업로드된 파일 '{uploaded_file_object.name}' 로드 실패. 모든 인코딩 시도 실패.")
    return None
# --- END OF utils.py ---
