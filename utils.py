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
    try:
        # ** matplotlib 폰트 캐시를 강제로 재빌드 시도 **
        # 이 작업은 시간이 다소 소요될 수 있으며, 앱 시작 시 한 번만 수행됩니다.
        # Streamlit Cloud 환경에서 폰트가 새로 설치된 후 인식되도록 돕습니다.
        # fm.findSystemFonts(fontpaths=None, fontext='ttf') # 시스템 폰트 다시 스캔 (선택적)
        # print("DEBUG (utils.py): Attempting to rebuild font cache...") # 로그
        fm._rebuild() # matplotlib의 내부 함수로, 폰트 캐시를 재구성합니다.
        # print("DEBUG (utils.py): Font cache rebuild attempt finished.") # 로그
    except Exception as e:
        # print(f"DEBUG (utils.py): Error during font cache rebuild: {e}") # 로그
        pass # 재빌드 실패 시에도 계속 진행


    nanum_font_paths_linux = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
    ]
    for font_path in nanum_font_paths_linux:
        if os.path.exists(font_path):
            try:
                font_name = fm.FontProperties(fname=font_path).get_name()
                plt.rc('font', family=font_name)
                plt.rcParams['font.family'] = font_name
                font_found = True
                # print(f"DEBUG (utils.py): Linux - 경로에서 폰트 설정 성공: {font_name} ({font_path})")
                break 
            except Exception as e:
                # print(f"DEBUG (utils.py): Linux - 폰트 파일 ({font_path}) 로드 실패: {e}")
                pass
    
    if not font_found:
        try:
            available_fonts_names = [f.name for f in fm.fontManager.ttflist]
            # print(f"DEBUG (utils.py): 사용 가능한 시스템 폰트 (캐시 재빌드 후): {available_fonts_names[:20]}")

            preferred_font_names = ['NanumGothic', 'Noto Sans CJK KR', 'Malgun Gothic', 'AppleGothic', 'sans-serif']
            for font_name_system in preferred_font_names:
                if font_name_system in available_fonts_names:
                    plt.rc('font', family=font_name_system)
                    plt.rcParams['font.family'] = font_name_system
                    font_found = True
                    # print(f"DEBUG (utils.py): 시스템 폰트 '{font_name_system}' 설정 성공.")
                    break
        except Exception as e:
            # print(f"DEBUG (utils.py): 시스템 폰트 검색/설정 중 오류 (캐시 재빌드 후): {e}")
            pass

    # 로컬 환경 설정 (이전과 동일하게 유지 가능)
    if not font_found:
        if os.name == 'nt': # Windows
            font_path_windows_nanum = "c:/Windows/Fonts/NanumGothic.ttf"
            font_path_windows_malgun = "c:/Windows/Fonts/malgun.ttf"
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
            if not font_found and 'Malgun Gothic' in [f.name for f in fm.fontManager.ttflist]:
                plt.rc('font', family='Malgun Gothic'); plt.rcParams['font.family'] = 'Malgun Gothic'; font_found = True

        elif os.name == 'posix' and "darwin" in os.uname().sysname.lower(): # macOS
            font_path_macos_nanum = "/Library/Fonts/NanumGothic.ttf"
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
            if not font_found and 'AppleGothic' in [f.name for f in fm.fontManager.ttflist]:
                plt.rc('font', family='AppleGothic'); plt.rcParams['font.family'] = 'AppleGothic'; font_found = True


    if not font_found:
        st.sidebar.warning( # 이 경고는 set_korean_font 함수가 호출된 후에 표시됨
             "한글 폰트(예: NanumGothic)를 시스템에서 찾거나 설정할 수 없습니다. "
             "그래프의 한글이 깨질 수 있습니다. "
             "로컬에서는 폰트 설치, 배포 환경에서는 'packages.txt' 설정을 확인해주세요."
        )
        plt.rc('font', family='sans-serif')
        plt.rcParams['font.family'] = 'sans-serif'
    # else:
        # print(f"DEBUG (utils.py): 최종 Matplotlib 폰트: {plt.rcParams['font.family']}")
    
    plt.rcParams['axes.unicode_minus'] = False

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
        print(f"ERROR (utils.py): 데이터 파일을 찾을 수 없습니다: {full_path}")
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
