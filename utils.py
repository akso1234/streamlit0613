# --- START OF utils.py (경고 메시지 제거 버전) ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import json
import requests
import io

def set_korean_font():
    # print("DEBUG (utils.py): set_korean_font() CALLED - Attempting to use font file from repo.") # 디버깅 완료 후 주석 처리 권장
    font_found = False
    font_to_set = 'sans-serif' 

    font_filename_in_repo = "NanumGothic.ttf" 
    font_path_in_repo = font_filename_in_repo
    # font_path_in_repo = os.path.join("assets", "fonts", font_filename_in_repo) # 하위 폴더 사용 시

    # print(f"DEBUG (utils.py): Expected font file: '{font_path_in_repo}'") # 디버깅 완료 후 주석 처리 권장
    current_working_dir = os.getcwd()
    # print(f"DEBUG (utils.py): CWD: {current_working_dir}") # 디버깅 완료 후 주석 처리 권장
    absolute_font_path = os.path.join(current_working_dir, font_path_in_repo)
    # print(f"DEBUG (utils.py): Absolute font path check: '{absolute_font_path}'") # 디버깅 완료 후 주석 처리 권장

    if os.path.exists(absolute_font_path):
        # print(f"DEBUG (utils.py): Font file '{font_filename_in_repo}' FOUND at '{absolute_font_path}'.") # 디버깅 완료 후 주석 처리 권장
        try:
            # print(f"DEBUG (utils.py): Attempting fm.fontManager.addfont('{absolute_font_path}')") # 디버깅 완료 후 주석 처리 권장
            fm.fontManager.addfont(absolute_font_path)
            # print(f"DEBUG (utils.py): fm.fontManager.addfont() executed.") # 디버깅 완료 후 주석 처리 권장
            
            # print(f"DEBUG (utils.py): Attempting fm._rebuild()...") # 디버깅 완료 후 주석 처리 권장 (또는 아예 제거)
            fm._rebuild() 
            # print(f"DEBUG (utils.py): fm._rebuild() executed.") # 디버깅 완료 후 주석 처리 권장
            
            font_prop = fm.FontProperties(fname=absolute_font_path)
            font_name_from_file = font_prop.get_name()
            # print(f"DEBUG (utils.py): Font name extracted from file: '{font_name_from_file}'") # 디버깅 완료 후 주석 처리 권장

            plt.rcParams['font.family'] = font_name_from_file
            
            if plt.rcParams['font.family'] and plt.rcParams['font.family'][0] == font_name_from_file:
                font_found = True
                font_to_set = font_name_from_file
                # print(f"DEBUG (utils.py): Successfully SET plt.rcParams['font.family'] to '{font_name_from_file}'.") # 디버깅 완료 후 주석 처리 권장
            # else:
                # print(f"WARNING (utils.py): Tried to set font.family to '{font_name_from_file}', but rcParams shows: {plt.rcParams.get('font.family', ['Unknown'])}.") # 디버깅 완료 후 주석 처리 권장
        except Exception as e:
            # print(f"ERROR (utils.py): FAILED to process/set font from repository file '{absolute_font_path}'. Error: {e}") # 디버깅 완료 후 주석 처리 권장
            font_found = False
    # else:
        # print(f"ERROR (utils.py): Font file '{font_filename_in_repo}' NOT FOUND at '{absolute_font_path}'.") # 디버깅 완료 후 주석 처리 권장

    if not font_found:
        # st.sidebar.warning( # <--- 이 부분을 주석 처리 하거나 삭제합니다.
        #      f"리포지토리에 포함된 한글 폰트 파일 ('{font_filename_in_repo}')을 찾거나 설정할 수 없습니다. "
        #      "그래프의 한글이 깨질 수 있습니다. 파일 경로와 GitHub 리포지토리를 확인해주세요."
        # )
        print(f"WARNING (utils.py): Korean font setup from repo file FAILED. Font may be broken if no system font is found.")
        plt.rcParams['font.family'] = 'sans-serif' # 최후의 기본값
        font_to_set = 'sans-serif'
    # else: # 성공 시에는 특별한 메시지 없이 넘어감 (또는 print로만 남김)
        # print(f"INFO (utils.py): Matplotlib font family appears to be set to: {font_to_set}.")

    plt.rcParams['axes.unicode_minus'] = False
    # final_font_check = plt.rcParams.get('font.family', ['Unknown'])[0]
    # print(f"DEBUG (utils.py): Final Matplotlib font family for rendering: {final_font_check}") # 디버깅 완료 후 주석 처리 권장

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
