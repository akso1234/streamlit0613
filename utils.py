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
    
    # --- Streamlit Cloud (Linux) 환경 우선 순위 ---
    # packages.txt로 설치된 나눔 폰트 경로 직접 시도
    # 실제 Streamlit Cloud 환경에서는 이 경로들이 표준적임
    nanum_font_paths_linux = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
        # 필요한 경우 다른 나눔 폰트 파일 경로 추가
    ]
    for font_path in nanum_font_paths_linux:
        if os.path.exists(font_path):
            try:
                # fm.fontManager.addfont(font_path) # 필요시 폰트 매니저에 직접 추가
                font_name = fm.FontProperties(fname=font_path).get_name()
                plt.rc('font', family=font_name)
                plt.rcParams['font.family'] = font_name # 명시적으로 한 번 더 설정
                font_found = True
                # print(f"DEBUG (utils.py): Linux - 경로에서 폰트 설정 성공: {font_name} ({font_path})") # 터미널 로그용
                break 
            except Exception as e:
                # print(f"DEBUG (utils.py): Linux - 폰트 파일 ({font_path}) 로드 실패: {e}") # 터미널 로그용
                pass
    
    # 위 경로에서 못찾았거나, 다른 환경일 경우 이름으로 시스템 폰트 시도
    if not font_found:
        try:
            # matplotlib 폰트 캐시 재빌드는 Streamlit Cloud에서 문제를 일으키거나 효과가 없을 수 있으므로,
            # 매우 신중하게 사용하거나 사용하지 않는 것을 권장합니다.
            # fm._rebuild() # 주석 처리 또는 제거 권장

            # 시스템에 등록된 폰트 이름으로 찾기
            available_fonts_names = [f.name for f in fm.fontManager.ttflist]
            # print(f"DEBUG (utils.py): 사용 가능한 시스템 폰트 (일부): {available_fonts_names[:10]}") # 터미널 로그용

            preferred_font_names = ['NanumGothic', 'Noto Sans CJK KR', 'Malgun Gothic', 'AppleGothic', 'sans-serif']
            for font_name_system in preferred_font_names:
                if font_name_system in available_fonts_names:
                    plt.rc('font', family=font_name_system)
                    plt.rcParams['font.family'] = font_name_system
                    font_found = True
                    # print(f"DEBUG (utils.py): 시스템 폰트 '{font_name_system}' 설정 성공.") # 터미널 로그용
                    break
        except Exception as e:
            # print(f"DEBUG (utils.py): 시스템 폰트 검색/설정 중 오류: {e}") # 터미널 로그용
            pass

    # 로컬 환경 (Windows, macOS)에 대한 추가적인 경로 설정 (선택 사항, 주로 로컬 개발용)
    # Streamlit Cloud에서는 위 로직이 우선적으로 작동해야 함
    if not font_found:
        if os.name == 'nt': # Windows
            font_path_windows_nanum = "c:/Windows/Fonts/NanumGothic.ttf"
            font_path_windows_malgun = "c:/Windows/Fonts/malgun.ttf" # 맑은 고딕
            if os.path.exists(font_path_windows_nanum):
                try:
                    font_name = fm.FontProperties(fname=font_path_windows_nanum).get_name()
                    plt.rc('font', family=font_name); plt.rcParams['font.family'] = font_name; font_found = True
                except: pass
            if not font_found and os.path.exists(font_path_windows_malgun):
                try:
                    font_name = fm.FontProperties(fname=font_path_windows_malgun).get_name()
                    plt.rc('font', family=font_name); plt.rcParams['font.family'] = font_name; font_found = True
                except: pass
            if not font_found and 'Malgun Gothic' in [f.name for f in fm.fontManager.ttflist]: # 이름으로 재시도
                plt.rc('font', family='Malgun Gothic'); plt.rcParams['font.family'] = 'Malgun Gothic'; font_found = True

        elif os.name == 'posix' and "darwin" in os.uname().sysname.lower(): # macOS
            font_path_macos_nanum = "/Library/Fonts/NanumGothic.ttf" # 시스템에 따라 다름
            font_path_macos_apple = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
            if os.path.exists(font_path_macos_nanum):
                try:
                    font_name = fm.FontProperties(fname=font_path_macos_nanum).get_name()
                    plt.rc('font', family=font_name); plt.rcParams['font.family'] = font_name; font_found = True
                except: pass
            if not font_found and os.path.exists(font_path_macos_apple):
                try:
                    font_name = fm.FontProperties(fname=font_path_macos_apple).get_name()
                    plt.rc('font', family=font_name); plt.rcParams['font.family'] = font_name; font_found = True
                except: pass
            if not font_found and 'AppleGothic' in [f.name for f in fm.fontManager.ttflist]: # 이름으로 재시도
                plt.rc('font', family='AppleGothic'); plt.rcParams['font.family'] = 'AppleGothic'; font_found = True

    if not font_found:
        # 이 경고는 st.set_page_config() 이후에 호출되어야 하므로,
        # 이 함수를 호출하는 쪽(페이지 스크립트)에서 처리하거나,
        # 여기서는 print로만 남기고 페이지 스크립트에서 st.warning을 사용할 수 있습니다.
        # print("WARNING (utils.py): 한글 폰트를 시스템에서 찾거나 설정할 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")
        st.sidebar.warning( # set_page_config 이후에 호출되도록 페이지에서 관리
             "한글 폰트(예: NanumGothic)를 시스템에서 찾거나 설정할 수 없습니다. "
             "그래프의 한글이 깨질 수 있습니다. "
             "로컬에서는 폰트 설치, 배포 환경에서는 'packages.txt' 설정을 확인해주세요."
        )
        plt.rc('font', family='sans-serif') # 최후의 보루
        plt.rcParams['font.family'] = 'sans-serif'
    # else:
        # print(f"DEBUG (utils.py): 최종 Matplotlib 폰트: {plt.rcParams['font.family']}") # 터미널 로그용
    
    plt.rcParams['axes.unicode_minus'] = False

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
        # st.error()는 set_page_config() 규칙에 걸릴 수 있으므로 print 또는 로깅 사용 고려
        print(f"ERROR (utils.py): 데이터 파일을 찾을 수 없습니다: {full_path}")
        # 또는 호출하는 쪽에서 st.error 처리
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
            print(f"WARNING (utils.py): '{os.path.basename(full_path)}' 파일 로드 중 오류 (인코딩: {encoding}): {e}")
            return None
            
    print(f"ERROR (utils.py): '{os.path.basename(full_path)}' 파일 로드 실패. 지원되는 인코딩을 찾을 수 없습니다.")
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
                print(f"ERROR (utils.py): GeoJSON 파일을 찾을 수 없습니다: {full_path}")
                return None
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except requests.exceptions.RequestException as e:
        print(f"ERROR (utils.py): GeoJSON URL에서 데이터 로드 중 오류: {e}")
        return None
    except FileNotFoundError:
        print(f"ERROR (utils.py): GeoJSON 파일을 찾을 수 없습니다: {path_or_url}")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR (utils.py): GeoJSON 파일 파싱 중 오류: {e}")
        return None
    except Exception as e:
        print(f"ERROR (utils.py): GeoJSON 로드 중 예기치 않은 오류 발생: {e}")
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
            print(f"WARNING (utils.py): 업로드된 파일 '{uploaded_file_object.name}' 로드 중 오류 (인코딩: {encoding}): {e}")
            return None
            
    print(f"ERROR (utils.py): 업로드된 파일 '{uploaded_file_object.name}' 로드 실패. 모든 인코딩 시도 실패.")
    return None
# --- END OF utils.py ---
