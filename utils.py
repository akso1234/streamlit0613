# --- START OF utils.py (디버깅 로그 활성화 버전) ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# import matplotlib # matplotlib_fname()는 이 방식에서 덜 중요
import os
import json
import requests
import io

def set_korean_font():
    """
    Matplotlib에서 한글 사용을 위한 설정을 수행합니다.
    리포지토리에 포함된 폰트 파일을 직접 사용합니다. (디버깅 로그 활성화)
    """
    print("DEBUG (utils.py): set_korean_font() CALLED - Attempting to use font file from repo.") # 로그 추가
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
    
    # Streamlit Cloud에서는 os.getcwd()가 앱 루트일 가능성이 높습니다.
    # 이를 기준으로 절대 경로를 만들어 확인해봅니다.
    # (더 확실한 방법은 앱 루트를 다른 방식으로 얻는 것이지만, 일단 CWD 기준으로 테스트)
    absolute_font_path_check = os.path.join(os.getcwd(), font_path_in_repo)
    print(f"DEBUG (utils.py): Checking for font file at absolute path: '{absolute_font_path_check}'")

    if os.path.exists(font_path_in_repo): # Streamlit Cloud는 앱 루트를 기준으로 이 상대 경로를 해석해야 함
        print(f"DEBUG (utils.py): Font file '{font_path_in_repo}' FOUND relative to app root.")
        try:
            font_prop = fm.FontProperties(fname=font_path_in_repo)
            font_name_from_file = font_prop.get_name()

            # Matplotlib의 rcParams에 설정
            plt.rc('font', family=font_name_from_file) # 폰트 파일에서 가져온 이름으로 설정
            plt.rcParams['font.family'] = font_name_from_file # 명시적 재설정
            font_found = True
            print(f"DEBUG (utils.py): Successfully SET font to '{font_name_from_file}' using repo file: {font_path_in_repo}")
        except Exception as e:
            print(f"DEBUG (utils.py): FAILED to set font using repo file '{font_path_in_repo}'. Error: {e}")
            # 폰트 파일을 찾았지만 FontProperties 생성 또는 rc 설정에서 오류 발생 시
            # 파일 권한, 폰트 파일 손상 등을 의심해볼 수 있으나 매우 드문 경우임
            pass
    else:
        print(f"DEBUG (utils.py): Font file '{font_path_in_repo}' NOT FOUND relative to app root.")
        print(f"DEBUG (utils.py): Listing files in current working directory ('{os.getcwd()}'):")
        try:
            for item in os.listdir(os.getcwd()):
                print(f"  - {item}")
        except Exception as e_ls:
            print(f"  Could not list files in CWD: {e_ls}")
        
        # 만약 assets/fonts 경로를 사용했다면 해당 경로도 확인
        if "assets" in font_path_in_repo:
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
        st.sidebar.warning( # 이 경고는 set_korean_font 함수가 호출된 후에 표시됨
             f"리포지토리에 포함된 한글 폰트 파일 ('{font_path_in_repo}')을 찾거나 설정할 수 없습니다. "
             "그래프의 한글이 깨질 수 있습니다. 파일 경로와 GitHub 리포지토리를 확인해주세요."
        )
        # 시스템 기본 폰트로 폴백 시도 (로컬 환경용, 또는 Cloud에 다른 폰트가 우연히 있다면)
        preferred_system_fonts = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'sans-serif'] # sans-serif는 최후
        for sys_font_name in preferred_system_fonts:
            try:
                plt.rcParams['font.family'] = sys_font_name
                # 설정이 실제로 적용되었는지 확인 (일부 시스템에서는 이름만으로 안될 수 있음)
                if plt.rcParams['font.family'][0] == sys_font_name: 
                    font_found_fallback = True # 폴백 성공 여부 (최종 font_found와는 별개로)
                    print(f"DEBUG (utils.py): Fallback to system font attempt: '{sys_font_name}' (Might still break Korean if font doesn't support it)")
                    # font_found = True # 이 줄을 활성화하면, 시스템 폴백 성공 시 경고가 안 뜰 수 있음
                    break
            except:
                continue
        if not font_found: # 리포지토리 폰트도, 시스템 폴백도 다 실패하면
             plt.rcParams['font.family'] = 'sans-serif' # 최후의 기본값


    plt.rcParams['axes.unicode_minus'] = False
    current_font_check_list = plt.rcParams.get('font.family', ['Unknown'])
    current_font_check = current_font_check_list[0] if current_font_check_list else 'Unknown'
    print(f"DEBUG (utils.py): Final Matplotlib font family for rendering: {current_font_check}")
    if 'nanum' not in current_font_check.lower() and 'malgun' not in current_font_check.lower() and 'apple' not in current_font_check.lower():
        print(f"WARNING (utils.py): The final font '{current_font_check}' may not support Korean well.")


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
