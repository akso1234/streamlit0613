# --- START OF utils.py (모든 디버깅 로그 활성화 버전) ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib # matplotlib_fname() 사용을 위해 추가
import os
import json
import requests
import io

def set_korean_font():
    """
    Matplotlib에서 한글 사용을 위한 설정을 수행합니다.
    리포지토리에 포함된 폰트 파일을 직접 사용합니다. (모든 디버깅 로그 활성화)
    """
    print("DEBUG (utils.py): set_korean_font() CALLED - Attempting to use font file from repo.")
    font_found = False

    # --- 사용자가 리포지토리에 추가할 폰트 파일명 및 경로 설정 ---
    # !!! 중요: 실제 사용하는 폰트 파일명으로 수정하세요 !!!
    font_filename = "NanumGothic.ttf" 
    
    # 폰트 파일이 리포지토리 루트에 있다고 가정
    font_path_in_repo = font_filename 

    # 만약 'assets/fonts/' 폴더 안에 있다면 아래 주석을 해제하고 위 라인을 주석 처리:
    # font_path_in_repo = os.path.join("assets", "fonts", font_filename)
    # -----------------------------------------------------------

    print(f"DEBUG (utils.py): Expected font file relative path in repo: '{font_path_in_repo}'")
    print(f"DEBUG (utils.py): Current working directory (os.getcwd()): {os.getcwd()}")
    
    absolute_font_path_check = os.path.join(os.getcwd(), font_path_in_repo)
    print(f"DEBUG (utils.py): Checking for font file at absolute path: '{absolute_font_path_check}'")

    if os.path.exists(font_path_in_repo):
        print(f"DEBUG (utils.py): Font file '{font_path_in_repo}' FOUND relative to app root.")
        try:
            font_prop = fm.FontProperties(fname=font_path_in_repo)
            font_name_from_file = font_prop.get_name()
            print(f"DEBUG (utils.py): Font name extracted from file '{font_path_in_repo}' is: '{font_name_from_file}'")

            # Matplotlib의 rcParams에 설정
            # plt.rc('font', family=font_name_from_file) # 이 줄 대신 아래 rcParams 직접 설정 사용
            plt.rcParams['font.family'] = font_name_from_file # 폰트 파일에서 가져온 이름으로 설정
            # sans-serif 목록의 가장 앞에 추가하여 우선순위를 높이는 것도 한 방법일 수 있습니다.
            # current_sans_serif = plt.rcParams['font.sans-serif']
            # if font_name_from_file not in current_sans_serif:
            #     plt.rcParams['font.sans-serif'] = [font_name_from_file] + current_sans_serif
            # print(f"DEBUG (utils.py): plt.rcParams['font.sans-serif'] after prepending: {plt.rcParams['font.sans-serif']}")
            
            font_found = True
            print(f"DEBUG (utils.py): Successfully SET plt.rcParams['font.family'] to '{font_name_from_file}' using repo file: {font_path_in_repo}")
        except Exception as e:
            print(f"DEBUG (utils.py): FAILED to set font using repo file '{font_path_in_repo}'. Error: {e}")
            pass
    else:
        print(f"DEBUG (utils.py): Font file '{font_path_in_repo}' NOT FOUND relative to app root.")
        print(f"DEBUG (utils.py): Listing files in current working directory ('{os.getcwd()}'):")
        try:
            for item in os.listdir(os.getcwd()):
                print(f"  - {item}")
        except Exception as e_ls:
            print(f"  Could not list files in CWD: {e_ls}")
        
        if "assets" in font_path_in_repo: # assets/fonts 경로를 사용했을 경우 추가 디버깅
            assets_fonts_path = os.path.join(os.getcwd(), "assets", "fonts")
            print(f"DEBUG (utils.py): Listing files in '{assets_fonts_path}' (if it exists):")
            if os.path.exists(assets_fonts_path) and os.path.isdir(assets_fonts_path):
                try:
                    for item in os.listdir(assets_fonts_path):
                        print(f"  - {item}")
                except Exception as e_ls_af:
                    print(f"  Could not list files in assets/fonts: {e_ls_af}")
            else:
                print(f"  Path '{assets_fonts_path}' does not exist or is not a directory.")


    if not font_found:
        print("WARNING (utils.py): Could not set Korean font from repository file. Will try system fallbacks if any.")
        st.sidebar.warning(
             f"리포지토리에 포함된 한글 폰트 파일 ('{font_path_in_repo}')을 찾거나 설정할 수 없습니다. "
             "그래프의 한글이 깨질 수 있습니다. 파일 경로와 GitHub 리포지토리를 확인해주세요."
        )
        # 시스템 기본 폰트로 폴백 시도
        preferred_system_fonts = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'sans-serif']
        for sys_font_name in preferred_system_fonts:
            try:
                # print(f"DEBUG (utils.py): Attempting fallback to system font: '{sys_font_name}'")
                plt.rcParams['font.family'] = sys_font_name
                # 설정이 실제로 적용되었는지 간단히 확인
                if plt.rcParams['font.family'][0] == sys_font_name: 
                    print(f"DEBUG (utils.py): Fallback to system font '{sys_font_name}' seems to be applied to rcParams.")
                    # font_found = True # 이 줄을 활성화하면, 시스템 폴백 성공 시 경고가 안 뜰 수 있음 (그러나 한글 지원 보장X)
                    break 
            except Exception as e_fallback:
                print(f"DEBUG (utils.py): Fallback to system font '{sys_font_name}' FAILED: {e_fallback}")
                continue
        if not font_found: # 리포지토리 폰트도, 시스템 폴백도 다 실패하면
             print(f"DEBUG (utils.py): All font setting attempts failed. Setting to 'sans-serif'.")
             plt.rcParams['font.family'] = 'sans-serif'


    plt.rcParams['axes.unicode_minus'] = False
    current_font_check_list = plt.rcParams.get('font.family', ['Unknown'])
    current_font_check = current_font_check_list[0] if current_font_check_list else 'Unknown'
    print(f"DEBUG (utils.py): Final Matplotlib font family for rendering: {current_font_check}")
    
    # 최종적으로 설정된 폰트가 실제 한글을 지원하는지 (이름 기반으로) 다시 한번 확인
    expected_korean_font_names_lower_check = ['nanumgothic', 'nanumbarungothic', 'nanumsquare', 'nanummyeongjo', 'noto sans cjk kr', 'malgun gothic', 'applegothic', 'apple sd gothic neo']
    if not any(expected_name in current_font_check.lower() for expected_name in expected_korean_font_names_lower_check):
        print(f"WARNING (utils.py): The final font '{current_font_check}' may NOT support Korean well. Expecting one of {expected_korean_font_names_lower_check}.")


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
