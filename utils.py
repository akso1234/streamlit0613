# --- START OF utils.py (디버깅 정보 사이드바 및 로그 동시 출력) ---
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
    리포지토리에 포함된 폰트 파일을 직접 사용하고, 디버깅 정보를 사이드바 및 로그에 표시합니다.
    """
    # Streamlit UI 요소 사용은 set_page_config 이후에 호출되어야 하므로,
    # 이 함수가 페이지 스크립트의 적절한 위치에서 호출된다고 가정합니다.
    
    log_messages = [] # 로그 및 사이드바 메시지 저장용 리스트

    log_messages.append("DEBUG: set_korean_font() CALLED - Attempting to use font file from repo.")
    font_found = False
    font_to_set = 'sans-serif' 

    # --- 사용자가 리포지토리에 추가할 폰트 파일명 및 경로 설정 ---
    # !!! 중요: 실제 사용하는 폰트 파일명으로 정확히 수정하세요 !!!
    font_filename_in_repo = "NanumGothic.ttf" 
    
    # 폰트 파일이 리포지토리 루트에 있다고 가정합니다.
    font_path_in_repo = font_filename_in_repo 

    # 만약 'assets/fonts/' 폴더 안에 있다면 아래 주석을 해제하고 위 라인을 주석 처리:
    # font_path_in_repo = os.path.join("assets", "fonts", font_filename_in_repo)
    # -----------------------------------------------------------

    log_messages.append(f"DEBUG: Expected font file relative path in repo: '{font_path_in_repo}'")
    current_working_dir = os.getcwd()
    log_messages.append(f"DEBUG: Current working directory (os.getcwd()): {current_working_dir}")
    
    absolute_font_path = os.path.join(current_working_dir, font_path_in_repo)
    log_messages.append(f"DEBUG: Checking for font file at absolute path: '{absolute_font_path}'")

    if os.path.exists(absolute_font_path):
        log_messages.append(f"DEBUG: Font file '{font_filename_in_repo}' FOUND at '{absolute_font_path}'.")
        try:
            # print(f"DEBUG: Attempting fm.fontManager.addfont('{absolute_font_path}')") # 로그 중복 방지
            # fm.fontManager.addfont(absolute_font_path) # 이 줄은 캐시 재빌드 전에 실행
            # print(f"DEBUG: fm.fontManager.addfont() executed.")
            
            # print(f"DEBUG: Attempting fm._rebuild()...") # 로그 중복 방지
            # fm._rebuild() # 캐시 재빌드는 앱 시작 속도에 영향 줄 수 있어 주의
            # print(f"DEBUG: fm._rebuild() executed.")
            
            font_prop = fm.FontProperties(fname=absolute_font_path)
            font_name_from_file = font_prop.get_name()
            log_messages.append(f"DEBUG: Font name extracted from file '{font_path_in_repo}' is: '{font_name_from_file}'")

            plt.rcParams['font.family'] = font_name_from_file
            
            if plt.rcParams['font.family'] and plt.rcParams['font.family'][0] == font_name_from_file:
                font_found = True
                font_to_set = font_name_from_file
                log_messages.append(f"DEBUG: Successfully SET plt.rcParams['font.family'] to '{font_name_from_file}'.")
            else:
                log_messages.append(f"WARNING: Tried to set font.family to '{font_name_from_file}', but rcParams shows: {plt.rcParams.get('font.family', ['Unknown'])}.")
        except Exception as e:
            log_messages.append(f"ERROR: FAILED to process or set font from repository file '{absolute_font_path}'. Error: {e}")
            font_found = False
    else:
        log_messages.append(f"ERROR: Font file '{font_filename_in_repo}' NOT FOUND at '{absolute_font_path}'.")
        log_messages.append(f"DEBUG: Listing files in CWD ('{current_working_dir}'):")
        try:
            for item in os.listdir(current_working_dir):
                log_messages.append(f"  - {item}")
        except Exception as e_ls:
            log_messages.append(f"  Could not list files in CWD: {e_ls}")
        
        if "assets" in font_path_in_repo and font_path_in_repo != font_filename_in_repo :
            assets_fonts_path_check_dir = os.path.dirname(absolute_font_path)
            log_messages.append(f"DEBUG: Also checking configured subfolder path: '{assets_fonts_path_check_dir}'")
            if os.path.exists(assets_fonts_path_check_dir) and os.path.isdir(assets_fonts_path_check_dir):
                log_messages.append(f"DEBUG: Listing files in '{assets_fonts_path_check_dir}':")
                try:
                    for item in os.listdir(assets_fonts_path_check_dir):
                        log_messages.append(f"  - {item}")
                except Exception as e_ls_af:
                    log_messages.append(f"  Could not list files in configured subfolder: {e_ls_af}")
            else:
                log_messages.append(f"  Path '{assets_fonts_path_check_dir}' does not exist or is not a directory.")

    plt.rcParams['axes.unicode_minus'] = False
    
    final_font_family_list_after_repo_attempt = plt.rcParams.get('font.family', ['Unknown'])
    final_font_family_after_repo_attempt = final_font_family_list_after_repo_attempt[0] if final_font_family_list_after_repo_attempt else 'Unknown'
    log_messages.append(f"DEBUG: Font family after repository font attempt: {final_font_family_after_repo_attempt}")

    # Streamlit UI에 디버깅 정보 표시 (사이드바)
    # 이 부분은 set_page_config() 호출 이후에 set_korean_font()가 호출될 때만 안전합니다.
    if st.runtime.exists(): # Streamlit 스크립트 실행 환경인지 확인
        st.sidebar.subheader("Font Setup Debug Log")
        for msg in log_messages:
            if "ERROR:" in msg:
                st.sidebar.error(msg)
            elif "WARNING:" in msg:
                st.sidebar.warning(msg)
            else:
                st.sidebar.caption(msg) # DEBUG 또는 INFO 메시지

        # 최종적으로 설정된 폰트와 기대하는 폰트 비교 후 메시지 표시
        expected_korean_font_names_lower_final = ['nanumgothic', 'nanumbarungothic', 'nanumsquare', 'nanummyeongjo']
        is_korean_font_actually_set = any(expected_name in final_font_family_after_repo_attempt.lower() for expected_name in expected_korean_font_names_lower_final)

        if not is_korean_font_actually_set:
            st.sidebar.error( # 이전의 warning을 error로 변경하여 더 눈에 띄게
                f"최종 Matplotlib 폰트가 '{final_font_family_after_repo_attempt}'(으)로, 한글 지원이 안 될 수 있습니다. "
                "위 로그를 확인하여 폰트 파일 경로 및 처리 과정을 점검하세요."
            )
        else:
            st.sidebar.success(f"최종 Matplotlib 폰트가 '{final_font_family_after_repo_attempt}'(으)로 설정되어 한글을 지원할 것으로 보입니다.")
        st.sidebar.markdown("---")

    # 터미널 로그에도 중요 정보 한 번 더 출력 (Cloud 로그 확인용)
    print(f"FINAL_FONT_CHECK (utils.py): plt.rcParams['font.family'] = {plt.rcParams.get('font.family')}")


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
