# --- START OF utils.py (모든 디버깅 로그 활성화 및 폰트 설정 강화) ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib # matplotlib_fname() 사용을 위해 추가 (현재는 주석 처리된 로직에서만 사용)
import os
import json
import requests
import io

def set_korean_font():
    """
    Matplotlib에서 한글 사용을 위한 설정을 수행합니다.
    리포지토리에 포함된 폰트 파일을 직접 사용하고, 디버깅 정보를 사이드바 또는 로그에 표시합니다.
    """
    # Streamlit UI 요소 사용은 set_page_config 이후에 호출되어야 하므로,
    # 이 함수가 페이지 스크립트의 적절한 위치에서 호출된다고 가정합니다.
    # 초기 디버깅 시에는 print를 사용하고, 안정화되면 st.sidebar 등으로 변경 가능합니다.
    
    print("DEBUG (utils.py): set_korean_font() CALLED - Attempting to use font file from repo.")
    font_found = False
    font_to_set = 'sans-serif' # 최종적으로 설정될 폰트 이름 (기본값)

    # --- 사용자가 리포지토리에 추가할 폰트 파일명 및 경로 설정 ---
    # !!! 중요: 실제 사용하는 폰트 파일명으로 정확히 수정하세요 !!!
    # 예시: "NanumGothic.ttf", "NanumGothic.otf", "NanumBarunGothic.ttf" 등
    font_filename_in_repo = "NanumGothic.ttf" 
    
    # 폰트 파일이 리포지토리 루트에 있다고 가정합니다.
    # Streamlit Cloud에서는 앱의 루트 디렉토리가 현재 작업 디렉토리(os.getcwd())와 일치하는 경향이 있습니다.
    font_path_in_repo = font_filename_in_repo 

    # 만약 'assets/fonts/' 와 같은 하위 폴더에 폰트 파일을 두었다면, 아래와 같이 수정하세요:
    # font_path_in_repo = os.path.join("assets", "fonts", font_filename_in_repo)
    # -----------------------------------------------------------

    print(f"DEBUG (utils.py): Expected font file relative path in repo: '{font_path_in_repo}'")
    current_working_dir = os.getcwd()
    print(f"DEBUG (utils.py): Current working directory (os.getcwd()): {current_working_dir}")
    
    # 현재 작업 디렉토리를 기준으로 폰트 파일의 절대 경로를 만듭니다.
    absolute_font_path = os.path.join(current_working_dir, font_path_in_repo)
    print(f"DEBUG (utils.py): Checking for font file at absolute path: '{absolute_font_path}'")

    if os.path.exists(absolute_font_path): # 절대 경로로 존재 여부 확인
        print(f"DEBUG (utils.py): Font file '{font_filename_in_repo}' FOUND at '{absolute_font_path}'.")
        try:
            # 1. Matplotlib 폰트 매니저에 폰트 파일 경로를 직접 추가합니다.
            #    이렇게 하면 Matplotlib이 이 폰트의 존재를 알게 됩니다.
            print(f"DEBUG (utils.py): Attempting fm.fontManager.addfont('{absolute_font_path}')")
            fm.fontManager.addfont(absolute_font_path)
            print(f"DEBUG (utils.py): fm.fontManager.addfont() executed.")

            # 2. (필수일 수 있음) 폰트 캐시를 재빌드합니다.
            #    addfont 후에도 인식이 안 될 경우, 캐시 재빌드가 필요할 수 있습니다.
            #    주의: 이 작업은 앱 시작 시간을 늘릴 수 있습니다.
            print(f"DEBUG (utils.py): Attempting fm._rebuild()...")
            fm._rebuild() 
            print(f"DEBUG (utils.py): fm._rebuild() executed.")
            
            # 3. FontProperties를 통해 폰트 파일에서 실제 폰트 이름을 가져옵니다.
            #    이 이름이 Matplotlib 내부에서 사용됩니다.
            font_prop = fm.FontProperties(fname=absolute_font_path)
            font_name_from_file = font_prop.get_name()
            print(f"DEBUG (utils.py): Font name extracted from file '{font_path_in_repo}' is: '{font_name_from_file}'")

            # 4. Matplotlib의 rcParams에 추출된 폰트 이름을 사용하여 설정합니다.
            plt.rcParams['font.family'] = font_name_from_file
            
            # 5. 설정이 제대로 적용되었는지 확인합니다.
            if plt.rcParams['font.family'] and plt.rcParams['font.family'][0] == font_name_from_file:
                font_found = True
                font_to_set = font_name_from_file
                print(f"DEBUG (utils.py): Successfully SET plt.rcParams['font.family'] to '{font_name_from_file}'.")
            else:
                # 설정은 시도했으나, rcParams에 반영되지 않은 경우
                print(f"WARNING (utils.py): Tried to set font.family to '{font_name_from_file}', but rcParams shows: {plt.rcParams.get('font.family', ['Unknown'])}. This may indicate an issue.")
                # 이 경우, fontManager.ttflist에 해당 이름이 있는지 추가 확인
                if font_name_from_file in [f.name for f in fm.fontManager.ttflist]:
                    print(f"DEBUG (utils.py): Font '{font_name_from_file}' IS in fontManager.ttflist. rcParams setting might be overridden or delayed.")
                else:
                    print(f"WARNING (utils.py): Font '{font_name_from_file}' IS NOT in fontManager.ttflist even after addfont/rebuild. This is a problem.")


        except Exception as e:
            print(f"ERROR (utils.py): FAILED to process or set font from repository file '{absolute_font_path}'. Error: {e}")
            font_found = False # 오류 발생 시 확실히 False로
    else:
        print(f"ERROR (utils.py): Font file '{font_filename_in_repo}' NOT FOUND at '{absolute_font_path}'.")
        print(f"DEBUG (utils.py): Listing files in current working directory ('{current_working_dir}'):")
        try:
            for item in os.listdir(current_working_dir):
                print(f"  - {item}")
        except Exception as e_ls:
            print(f"  Could not list files in CWD: {e_ls}")
        
        # 만약 assets/fonts 경로를 사용하도록 설정했다면 해당 경로도 확인
        if "assets" in font_path_in_repo and font_path_in_repo != font_filename_in_repo :
            assets_fonts_path_check = os.path.join(current_working_dir, "assets", "fonts") # font_path_in_repo에서 파일명 제외한 디렉토리
            print(f"DEBUG (utils.py): Also checking configured subfolder path: '{os.path.dirname(absolute_font_path)}'")
            if os.path.exists(os.path.dirname(absolute_font_path)) and os.path.isdir(os.path.dirname(absolute_font_path)):
                print(f"DEBUG (utils.py): Listing files in '{os.path.dirname(absolute_font_path)}':")
                try:
                    for item in os.listdir(os.path.dirname(absolute_font_path)):
                        print(f"  - {item}")
                except Exception as e_ls_af:
                    print(f"  Could not list files in configured subfolder: {e_ls_af}")
            else:
                print(f"  Configured subfolder path '{os.path.dirname(absolute_font_path)}' does not exist or is not a directory.")


    # 최종적으로 설정된 폰트 패밀리 확인
    final_font_family_list = plt.rcParams.get('font.family', ['Unknown'])
    final_font_family = final_font_family_list[0] if final_font_family_list else 'Unknown'

    if not font_found: # 리포지토리 폰트 설정 실패 시
        st.sidebar.warning(
             f"리포지토리 폰트('{font_filename_in_repo}') 설정에 실패했습니다. Matplotlib 기본 폰트('{final_font_family}')가 사용될 수 있으며, 이 경우 한글이 깨질 수 있습니다. "
             "GitHub 리포지토리 루트에 폰트 파일이 올바르게 있는지, 파일명이 정확한지 확인하고 앱을 재부팅(Reboot)해주세요. "
             "Streamlit Cloud 로그의 'DEBUG' 메시지를 확인하여 상세 원인을 파악하세요."
        )
    else: # 리포지토리 폰트 설정 성공 시
        # 추가적으로, 설정된 폰트가 실제로 한글 지원인지 (이름 기반으로) 확인
        expected_korean_font_names_check = ['nanumgothic', 'nanumbarungothic', 'nanumsquare', 'nanummyeongjo']
        if any(expected_name in final_font_family.lower() for expected_name in expected_korean_font_names_check):
            st.sidebar.success(f"한글 폰트('{final_font_family}')가 성공적으로 설정되었습니다.")
            print(f"INFO (utils.py): 한글 폰트 '{final_font_family}' 설정 성공.")
        else:
            st.sidebar.warning(f"폰트가 '{final_font_family}'(으)로 설정되었으나, 기대한 나눔계열 폰트가 아닐 수 있습니다. 한글 표시를 확인해주세요.")
            print(f"WARNING (utils.py): Font set to '{final_font_family}', but it might not be the intended Nanum font.")

    plt.rcParams['axes.unicode_minus'] = False
    print(f"--- set_korean_font in utils.py FINISHED. Final effective font.family: {plt.rcParams.get('font.family')} ---")


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
