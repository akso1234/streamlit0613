# --- START OF utils.py ---
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
    - matplotlibrc 파일을 통해 주 폰트 설정이 이루어진다고 가정합니다.
    - 여기서는 unicode_minus 설정을 하고, 현재 설정된 폰트를 확인하여 디버깅 정보를 출력합니다.
    """
    # 디버깅 로그: 함수 호출 시점 및 Matplotlib 초기 상태 확인
    # print("DEBUG (utils.py): set_korean_font() function CALLED.")
    # print(f"DEBUG (utils.py): Matplotlib rc file Matplotlib thinks it's using: {matplotlib.matplotlib_fname()}")
    # print(f"DEBUG (utils.py): Initial plt.rcParams['font.family'] from Matplotlib: {plt.rcParams.get('font.family')}")

    # matplotlibrc 파일이 프로젝트 루트에 있는지 확인 (Streamlit Cloud 기준)
    # Streamlit Cloud에서 앱 루트는 보통 /mount/src/<your_repo_name>/
    # os.getcwd()는 스크립트 실행 위치에 따라 다를 수 있으므로, 상대 경로 기준이 더 안정적일 수 있음
    # 그러나 Streamlit Cloud에서는 일반적으로 앱 루트에서 실행됨
    
    # rc_file_in_root_path = 'matplotlibrc' # 프로젝트 루트에 있다고 가정
    # if os.path.exists(rc_file_in_root_path):
    #     print(f"DEBUG (utils.py): 'matplotlibrc' FOUND in app root: {os.path.abspath(rc_file_in_root_path)}")
    #     try:
    #         with open(rc_file_in_root_path, 'r') as f:
    #             rc_content = f.read()
    #             print(f"DEBUG (utils.py): Content of 'matplotlibrc':\n{rc_content}")
    #             if "NanumGothic" not in rc_content and "NanumBarunGothic" not in rc_content: # 더 많은 폰트 확인 가능
    #                 print("WARNING (utils.py): 'matplotlibrc' file does not seem to specify a Nanum font for font.family.")
    #     except Exception as e:
    #         print(f"ERROR (utils.py): Could not read 'matplotlibrc' content: {e}")
    # else:
    #     print(f"WARNING (utils.py): 'matplotlibrc' NOT FOUND in app root ('{os.getcwd()}'). Ensure it's in the GitHub repo root.")

    # 주된 폰트 설정은 matplotlibrc에 의존. 여기서는 unicode_minus만 확실히 설정.
    plt.rcParams['axes.unicode_minus'] = False

    # 최종적으로 matplotlib이 어떤 폰트를 사용하는지 확인하고,
    # 만약 한글 지원 폰트가 아니라면 사용자에게 경고 (st.sidebar.warning)
    # 이 경고는 st.set_page_config() 이후에 호출되어야 하므로,
    # 이 함수를 호출하는 페이지 스크립트에서 set_korean_font()를 적절한 위치에 배치해야 함.
    
    # 현재 Matplotlib에 의해 실제로 사용될 폰트 패밀리 확인
    # rcParams는 여러 값을 가질 수 있으므로 첫 번째 값을 주로 확인
    final_font_family_list = plt.rcParams.get('font.family', ['Unknown'])
    final_font_family = final_font_family_list[0] if final_font_family_list else 'Unknown'
    
    # print(f"DEBUG (utils.py): Final effective plt.rcParams['font.family'] after all settings: {final_font_family_list}")

    # 기대하는 한글 폰트 이름 목록 (소문자로 비교)
    expected_korean_font_names_lower = ['nanumgothic', 'nanumbarungothic', 'nanumsquare', 'noto sans cjk kr', 'malgun gothic', 'apple SD gothic neo', 'apple sd gothicneo', 'apple sd 산돌고딕 neo']

    is_korean_font_likely_set = any(expected_name in final_font_family.lower() for expected_name in expected_korean_font_names_lower)

    if not is_korean_font_likely_set:
        # print(f"WARNING (utils.py): Final font family '{final_font_family}' does not seem to be a Korean font.")
        st.sidebar.warning(
             f"한글 폰트가 올바르게 설정되지 않은 것 같습니다 (현재 Matplotlib 기본 폰트: {final_font_family}). "
             "그래프의 한글이 깨질 수 있습니다. 'matplotlibrc' 파일과 'packages.txt' 설정을 확인하고, "
             "GitHub 리포지토리 루트에 파일들이 올바르게 위치하는지, 앱을 재부팅(Reboot)했는지 확인해주세요."
        )
    # else:
        # print(f"INFO (utils.py): Matplotlib font family appears to be set to a Korean-supporting font: {final_font_family}.")


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
    full_path = file_path # Assuming file_path is already like "data/filename.csv"
    
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
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
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
    except FileNotFoundError: # This might be redundant if os.path.exists is checked first
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
    _uploaded_file_object, # Underscore to indicate it's used for caching key
    encoding_options=['utf-8-sig', 'utf-8', 'cp949', 'euc-kr'], 
    header_config=None, 
    skiprows_config=None, 
    na_values_config=None,
    sep_config=','
):
    if _uploaded_file_object is None:
        return None
        
    # For actual processing, use the passed object directly
    uploaded_file_object = _uploaded_file_object 
    
    for encoding in encoding_options:
        try:
            # The uploaded file object has a getvalue() method to get bytes
            bytes_data = uploaded_file_object.getvalue()
            
            read_options = {'encoding': encoding, 'sep': sep_config}
            if header_config is not None: read_options['header'] = header_config
            if skiprows_config is not None: read_options['skiprows'] = skiprows_config
            # nrows is not typically used with uploaded files as we usually read the whole thing
            if na_values_config is not None: read_options['na_values'] = na_values_config

            # Pass BytesIO object to pandas
            df = pd.read_csv(io.BytesIO(bytes_data), **read_options)
            return df
        except UnicodeDecodeError:
            continue # Try next encoding
        except Exception as e:
            st.warning(f"업로드된 파일 '{uploaded_file_object.name}' 로드 중 오류 (인코딩: {encoding}): {e}")
            return None # Return None on other errors
            
    st.error(f"업로드된 파일 '{uploaded_file_object.name}' 로드 실패. 모든 인코딩 시도 실패.")
    return None
# --- END OF utils.py ---
