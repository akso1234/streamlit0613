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
    Streamlit Cloud 환경에 중점을 둡니다.
    """
    # print("DEBUG (utils.py): set_korean_font() CALLED.")
    font_found = False
    
    try:
        # print("DEBUG (utils.py): Attempting to rebuild font cache with fm._rebuild()...")
        fm._rebuild() # 폰트 캐시 강제 재빌드 시도
        # print("DEBUG (utils.py): Font cache rebuild attempt finished.")
    except Exception as e:
        # print(f"DEBUG (utils.py): Error during font cache rebuild: {e}")
        pass

    # 1. 가장 먼저, 알려진 한글 폰트 이름으로 직접 설정 시도
    #    fm._rebuild()가 제대로 동작했다면, 새로 설치된 폰트가 여기에 잡혀야 함
    preferred_font_names = ['NanumGothic', 'NanumBarunGothic', 'NanumSquare', 'Noto Sans CJK KR']
    
    try:
        available_system_fonts_after_rebuild = [f.name for f in fm.fontManager.ttflist]
        # print(f"DEBUG (utils.py): System fonts available AFTER REBUILD (sample): {available_system_fonts_after_rebuild[:20]}")

        for font_name in preferred_font_names:
            if font_name in available_system_fonts_after_rebuild:
                plt.rcParams['font.family'] = font_name
                # rcParams에 family를 설정하면 plt.rc는 따로 안해도 될 수 있음
                # plt.rc('font', family=font_name) 
                font_found = True
                # print(f"DEBUG (utils.py): Successfully SET font to '{font_name}' by NAME from fontManager.ttflist.")
                break
            # else:
                # print(f"DEBUG (utils.py): Font name '{font_name}' NOT in fontManager.ttflist after rebuild.")
    except Exception as e:
        # print(f"DEBUG (utils.py): Error checking/setting font by NAME after rebuild: {e}")
        pass

    # 2. 이름으로 설정 실패 시, 표준 설치 경로에서 직접 파일 찾아 설정 시도
    if not font_found:
        # print("DEBUG (utils.py): Failed to set font by name. Trying specific paths...")
        nanum_font_paths_linux = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
            '/usr/share/fonts/truetype/nanum/NanumSquareR.ttf', # Regular
            '/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf',
        ]
        for font_path in nanum_font_paths_linux:
            if os.path.exists(font_path):
                try:
                    # 폰트 매니저에 폰트 파일 경로를 알리고, 해당 폰트 속성을 가져와 설정
                    font_entry = fm.FontEntry(fname=font_path, name=os.path.splitext(os.path.basename(font_path))[0])
                    # fm.fontManager.ttflist.append(font_entry) # 직접 ttflist 조작은 마지막 수단
                    
                    # FontProperties를 통해 이름 가져오고 설정
                    font_prop_name = fm.FontProperties(fname=font_path).get_name()
                    plt.rcParams['font.family'] = font_prop_name
                    font_found = True
                    # print(f"DEBUG (utils.py): Successfully SET font to '{font_prop_name}' using PATH '{font_path}'.")
                    break
                except Exception as e:
                    # print(f"DEBUG (utils.py): Failed to set font from PATH '{font_path}': {e}")
                    pass
            # else:
                # print(f"DEBUG (utils.py): Font path NOT FOUND: {font_path}")


    # 로컬 환경 폴백 (이전과 유사)
    if not font_found:
        # print("DEBUG (utils.py): Nanum fonts not found by name or specific path even after rebuild. Trying local OS fallbacks.")
        if os.name == 'nt': # Windows
            if 'Malgun Gothic' in [f.name for f in fm.fontManager.ttflist]:
                plt.rcParams['font.family'] = 'Malgun Gothic'; font_found = True
        elif os.name == 'posix' and "darwin" in os.uname().sysname.lower(): # macOS
            if 'AppleGothic' in [f.name for f in fm.fontManager.ttflist]: # AppleGothic이 더 일반적
                plt.rcParams['font.family'] = 'AppleGothic'; font_found = True
            elif 'Apple SD Gothic Neo' in [f.name for f in fm.fontManager.ttflist]:
                 plt.rcParams['font.family'] = 'Apple SD Gothic Neo'; font_found = True


    # 최종 확인 및 경고
    final_font_family_list = plt.rcParams.get('font.family', ['Unknown'])
    final_font_family = final_font_family_list[0] if final_font_family_list else 'Unknown'

    expected_korean_font_names_lower = ['nanumgothic', 'nanumbarungothic', 'nanumsquare', 'nanummyeongjo', 'noto sans cjk kr', 'malgun gothic', 'applegothic', 'apple sd gothic neo']

    is_korean_font_likely_set = any(expected_name in final_font_family.lower() for expected_name in expected_korean_font_names_lower)

    if not is_korean_font_likely_set:
        # print(f"WARNING (utils.py): Final font family '{final_font_family}' after all attempts does not seem to be a Korean font.")
        st.sidebar.warning(
             f"한글 폰트가 올바르게 설정되지 않았습니다 (현재 Matplotlib 기본 폰트: {final_font_family}). "
             "그래프의 한글이 깨질 수 있습니다. 'packages.txt'에 'fonts-nanum*'이 있고 앱이 재부팅되었는지 확인하세요. "
             "Streamlit Cloud 로그에서 'DEBUG' 메시지를 통해 폰트 설정을 추적해보세요."
        )
    # else:
        # print(f"INFO (utils.py): Matplotlib font family appears to be successfully set to: {final_font_family}.")

    plt.rcParams['axes.unicode_minus'] = False

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
