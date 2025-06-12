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
    리포지토리에 포함된 폰트 파일을 직접 사용합니다.
    """
    # print("DEBUG (utils.py): set_korean_font() CALLED - Attempting to use font file from repo.")
    font_found = False

    # --- 사용자가 리포지토리에 추가할 폰트 파일명 및 경로 설정 ---
    # 예시: 폰트 파일이 리포지토리 루트에 있다면:
    font_filename = "NanumGothic.ttf" # 또는 실제 사용하는 파일명 (NanumGothic.otf 등)
    font_path_in_repo = font_filename 

    # 예시: 폰트 파일이 'assets/fonts/' 폴더 안에 있다면:
    # font_path_in_repo = os.path.join("assets", "fonts", font_filename)
    # -----------------------------------------------------------

    if os.path.exists(font_path_in_repo):
        try:
            font_prop = fm.FontProperties(fname=font_path_in_repo)
            font_name_from_file = font_prop.get_name() # 폰트 파일에서 실제 폰트 이름을 가져옴

            plt.rc('font', family=font_name_from_file)
            plt.rcParams['font.family'] = font_name_from_file
            font_found = True
            # print(f"DEBUG (utils.py): Successfully SET font to '{font_name_from_file}' using repo file: {font_path_in_repo}")
        except Exception as e:
            # print(f"DEBUG (utils.py): FAILED to set font using repo file '{font_path_in_repo}': {e}")
            pass
    # else:
        # print(f"DEBUG (utils.py): Font file NOT FOUND in repo at: {font_path_in_repo} (relative to app root)")
        # print(f"DEBUG (utils.py): Current working directory: {os.getcwd()}")
        # files_in_cwd = os.listdir(os.getcwd()) if os.path.isdir(os.getcwd()) else []
        # print(f"DEBUG (utils.py): Files in CWD: {files_in_cwd}")


    if not font_found:
        # print("WARNING (utils.py): Could not set Korean font from repository file.")
        st.sidebar.warning(
             f"리포지토리에 포함된 한글 폰트 파일 ('{font_path_in_repo}')을 찾거나 설정할 수 없습니다. "
             "그래프의 한글이 깨질 수 있습니다. 파일 경로와 GitHub 리포지토리를 확인해주세요."
        )
        # 폴백으로 시스템 기본 폰트 사용 (한글 깨짐 가능성 높음)
        plt.rcParams['font.family'] = 'sans-serif'


    plt.rcParams['axes.unicode_minus'] = False
    # current_font_check = plt.rcParams.get('font.family', ['Unknown'])[0]
    # print(f"DEBUG (utils.py): Final Matplotlib font family for rendering: {current_font_check}")

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
