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
    """
    Matplotlib에서 한글 사용을 위한 설정을 수행합니다.
    - matplotlibrc 파일을 통해 주 폰트 설정이 이루어진다고 가정합니다.
    - 여기서는 unicode_minus 설정을 하고, 현재 설정된 폰트를 확인하여 경고를 표시합니다.
    """
    # print("DEBUG (utils.py): set_korean_font() function CALLED.") # 함수 호출 확인

    # 1. matplotlibrc 파일이 로드되었는지 확인하기 위해 현재 폰트 설정 값 확인
    # rcParams는 딕셔너리처럼 동작하며, font.family는 리스트를 반환할 수 있음
    current_font_family_list = plt.rcParams.get('font.family', ['Unknown'])
    current_font_family = current_font_family_list[0] if current_font_family_list else 'Unknown'
    
    # print(f"DEBUG (utils.py): Current plt.rcParams['font.family'] = {current_font_family_list}")

    # 2. 실제로 해당 폰트가 시스템에 있고, Matplotlib이 인식하는지 확인
    #    font_manager를 통해 지정된 폰트의 실제 경로를 찾아보려고 시도합니다.
    font_path_found = None
    try:
        if current_font_family != 'Unknown' and current_font_family != 'sans-serif':
            font_path_found = fm.findfont(current_font_family, fallback_to_default=False, rebuild_if_missing=False)
            # print(f"DEBUG (utils.py): fm.findfont('{current_font_family}') found path: {font_path_found}")
    except Exception as e:
        # print(f"DEBUG (utils.py): fm.findfont('{current_font_family}') FAILED: {e}")
        font_path_found = None # 못 찾으면 None

    # 3. 경고 메시지 조건 강화
    #    현재 설정된 폰트가 우리가 기대하는 한글 폰트가 아니거나,
    #    또는 이름은 맞지만 실제 폰트 파일을 찾지 못하는 경우 경고
    expected_korean_fonts = ['nanumgothic', 'nanumbarungothic', 'nanumsquare', 'noto sans cjk kr'] # 소문자로 비교
    
    is_korean_font_set_properly = False
    if font_path_found: # 경로를 찾았다는 것은 matplotlib이 인식한다는 의미
        font_name_from_path = fm.FontProperties(fname=font_path_found).get_name().lower()
        if any(expected_font in font_name_from_path for expected_font in expected_korean_fonts):
            is_korean_font_set_properly = True
            # print(f"DEBUG (utils.py): Korean font '{font_name_from_path}' seems to be set correctly via path '{font_path_found}'.")
        # else:
            # print(f"DEBUG (utils.py): Font '{font_name_from_path}' (from path) is not an expected Korean font.")
    elif current_font_family.lower() in expected_korean_fonts:
        # 이름은 맞지만 경로를 못 찾은 경우 (이 경우에도 깨질 수 있음)
        # print(f"DEBUG (utils.py): Font family set to '{current_font_family}', but fm.findfont did not find a path. This might still cause issues.")
        # 이 경우, fontManager.ttflist에 있는지 확인하는 것이 더 나을 수 있음
        pass


    if not is_korean_font_set_properly:
        # print(f"DEBUG (utils.py): Korean font NOT set properly. Current effective font: '{current_font_family}'. Path found by findfont: {font_path_found}")
        st.sidebar.warning(
             f"한글 폰트가 올바르게 설정되지 않았습니다 (현재: {current_font_family}). "
             "그래프의 한글이 깨질 수 있습니다. 'matplotlibrc'와 'packages.txt' 설정을 확인하고, "
             "앱을 재부팅(Reboot) 해보세요."
        )
        # 강제로 sans-serif로 설정할 필요는 없음. matplotlibrc의 설정을 믿어보거나,
        # matplotlib의 기본 폴백 메커니즘에 맡김.
        # plt.rcParams['font.family'] = 'sans-serif' # 이 줄은 문제를 해결하지 못함

    plt.rcParams['axes.unicode_minus'] = False
    # print(f"DEBUG (utils.py): Final plt.rcParams['font.family'] for rendering: {plt.rcParams.get('font.family')}")

# --- load_csv, load_geojson, load_csv_from_upload 함수는 이전 답변과 동일하게 유지 ---
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
        st.error(f"데이터 파일을 찾을 수 없습니다: {full_path}")
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
            st.warning(f"'{os.path.basename(full_path)}' 파일 로드 중 오류 발생 (인코딩: {encoding}): {e}")
            return None
            
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
                st.error(f"GeoJSON 파일을 찾을 수 없습니다: {full_path}")
                return None
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except requests.exceptions.RequestException as e:
        st.error(f"GeoJSON URL에서 데이터 로드 중 오류: {e}")
        return None
    except FileNotFoundError:
        st.error(f"GeoJSON 파일을 찾을 수 없습니다: {path_or_url}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"GeoJSON 파일 파싱 중 오류: {e}")
        return None
    except Exception as e:
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
            st.warning(f"업로드된 파일 '{uploaded_file_object.name}' 로드 중 오류 (인코딩: {encoding}): {e}")
            return None
            
    st.error(f"업로드된 파일 '{uploaded_file_object.name}' 로드 실패. 모든 인코딩 시도 실패.")
    return None
# --- END OF utils.py ---
