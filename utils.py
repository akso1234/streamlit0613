import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import json
import requests
import io # 파일 업로드 또는 BytesIO 사용 시 필요

# 폰트 설정 함수
def set_korean_font():
    """Matplotlib에서 한글을 사용하기 위한 폰트 설정을 수행합니다."""
    
    # Streamlit 배포 환경 (Linux) 또는 로컬 Linux 환경
    font_path_linux = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    
    # 로컬 Windows 환경
    font_path_windows = "c:/Windows/Fonts/NanumGothic.ttf" # 예시, 실제 경로 확인 필요
    # 또는 'Malgun Gothic' 사용
    # font_name_windows = 'Malgun Gothic'

    # 로컬 macOS 환경
    font_path_macos = "/Library/Fonts/NanumGothic.ttf" # 예시, 실제 경로 확인 필요
    # 또는 'AppleGothic' 사용
    # font_name_macos = 'AppleGothic'

    font_name_to_set = None
    font_found = False

    if os.path.exists(font_path_linux):
        try:
            font_name_to_set = fm.FontProperties(fname=font_path_linux).get_name()
            plt.rc('font', family=font_name_to_set)
            font_found = True
            # st.sidebar.info(f"Linux 환경: '{font_name_to_set}' 폰트 설정 완료.") # 디버깅용
        except Exception:
            pass # 다른 방법 시도
    
    if not font_found and os.name == 'nt': # Windows
        if os.path.exists(font_path_windows):
            try:
                font_name_to_set = fm.FontProperties(fname=font_path_windows).get_name()
                plt.rc('font', family=font_name_to_set)
                font_found = True
                # st.sidebar.info(f"Windows 환경: '{font_name_to_set}' 폰트 설정 완료.") # 디버깅용
            except Exception:
                pass
        if not font_found and 'Malgun Gothic' in [f.name for f in fm.fontManager.ttflist]:
            plt.rc('font', family='Malgun Gothic')
            font_found = True
            # st.sidebar.info("Windows 환경: 'Malgun Gothic' 폰트 설정 완료.") # 디버깅용

    if not font_found and os.name == 'posix' and "darwin" in os.uname().sysname.lower(): # macOS
        if os.path.exists(font_path_macos):
            try:
                font_name_to_set = fm.FontProperties(fname=font_path_macos).get_name()
                plt.rc('font', family=font_name_to_set)
                font_found = True
                # st.sidebar.info(f"macOS 환경: '{font_name_to_set}' 폰트 설정 완료.") # 디버깅용
            except Exception:
                pass
        if not font_found and 'AppleGothic' in [f.name for f in fm.fontManager.ttflist]:
            plt.rc('font', family='AppleGothic')
            font_found = True
            # st.sidebar.info("macOS 환경: 'AppleGothic' 폰트 설정 완료.") # 디버깅용

    if not font_found: # 어떤 한글 폰트도 찾지 못한 경우
        # 나눔고딕이 시스템에 설치되어 있는지 다시 한번 확인 (경로 없이 이름으로)
        available_system_fonts = [f.name for f in fm.fontManager.ttflist]
        if 'NanumGothic' in available_system_fonts:
            plt.rc('font', family='NanumGothic')
            font_found = True
            # st.sidebar.info("시스템 'NanumGothic' 폰트 설정 완료.") # 디버깅용
        else:
            st.sidebar.warning(
                "한글 폰트(예: NanumGothic)를 시스템에서 찾을 수 없습니다. "
                "그래프의 한글이 깨질 수 있습니다. "
                "폰트 설치 또는 경로 설정을 확인해주세요."
            )
            plt.rc('font', family='sans-serif') # 기본 sans-serif 사용
    
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

@st.cache_data # 데이터 로딩 결과를 캐싱하여 성능 향상
def load_csv(
    file_path, 
    encoding_options=['utf-8-sig', 'utf-8', 'cp949', 'euc-kr'], 
    header_config=None, 
    skiprows_config=None, 
    nrows_config=None, 
    na_values_config=None,
    sep_config=',' # 기본 구분자 쉼표
):
    """
    주어진 경로에서 CSV 파일을 로드합니다.
    여러 인코딩 옵션을 시도하며, 추가적인 pandas read_csv 옵션을 지원합니다.
    """
    full_path = file_path # 이미 'data/filename.csv' 형태의 상대 경로로 전달됨
    
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
            # st.sidebar.info(f"'{os.path.basename(full_path)}' 파일 로드 성공 (인코딩: {encoding}).") # 디버깅용
            return df
        except UnicodeDecodeError:
            continue # 다음 인코딩 시도
        except Exception as e:
            # 파일은 존재하지만 다른 오류로 로드 실패
            st.warning(f"'{os.path.basename(full_path)}' 파일 로드 중 오류 발생 (인코딩: {encoding}): {e}")
            return None # 오류 발생 시 None 반환
            
    # 모든 인코딩 시도 실패
    st.error(f"'{os.path.basename(full_path)}' 파일 로드 실패. 지원되는 인코딩을 찾을 수 없습니다.")
    return None

@st.cache_data
def load_geojson(path_or_url):
    """GeoJSON 파일을 로컬 경로 또는 URL에서 로드합니다."""
    try:
        if path_or_url.startswith('http'): # URL인 경우
            response = requests.get(path_or_url)
            response.raise_for_status() # HTTP 오류 발생 시 예외 처리
            # st.sidebar.info(f"GeoJSON 로드 성공 (URL: {path_or_url[:50]}...).") # 디버깅용
            return response.json()
        else: # 로컬 파일 경로인 경우
            full_path = path_or_url # 예: 'assets/seoul_municipalities_geo_simple.json'
            if not os.path.exists(full_path):
                st.error(f"GeoJSON 파일을 찾을 수 없습니다: {full_path}")
                return None
            with open(full_path, 'r', encoding='utf-8') as f:
                # st.sidebar.info(f"GeoJSON 로드 성공 (파일: {full_path}).") # 디버깅용
                return json.load(f)
    except requests.exceptions.RequestException as e:
        st.error(f"GeoJSON URL에서 데이터 로드 중 오류: {e}")
        return None
    except FileNotFoundError: # 중복이지만 명시적 처리
        st.error(f"GeoJSON 파일을 찾을 수 없습니다: {full_path}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"GeoJSON 파일 파싱 중 오류: {e}")
        return None
    except Exception as e:
        st.error(f"GeoJSON 로드 중 예기치 않은 오류 발생: {e}")
        return None

# Streamlit 파일 업로드 위젯을 통해 업로드된 CSV 파일을 로드하는 함수 (선택 사항)
# 현재 프로젝트에서는 고정된 경로의 파일을 사용하므로, 이 함수는 직접 사용되지 않을 수 있음.
@st.cache_data
def load_csv_from_upload(
    _uploaded_file_object, # Streamlit UploadedFile 객체 (캐시 키로 사용하기 위해 _ 추가)
    encoding_options=['utf-8-sig', 'utf-8', 'cp949', 'euc-kr'], 
    header_config=None, 
    skiprows_config=None, 
    na_values_config=None,
    sep_config=','
):
    """
    Streamlit의 file_uploader로 업로드된 CSV 파일을 DataFrame으로 로드합니다.
    _uploaded_file_object는 st.file_uploader()의 반환값입니다.
    """
    if _uploaded_file_object is None:
        return None
        
    # 실제 파일 데이터는 _uploaded_file_object에서 다시 가져옴
    # 캐싱을 위해 함수 인자로 UploadedFile 객체 자체를 받지만, 실제 사용 시에는 내부 데이터를 사용
    uploaded_file_object = _uploaded_file_object 
    
    for encoding in encoding_options:
        try:
            # 업로드된 파일은 BytesIO 객체로 변환하여 pandas에서 읽어야 함
            bytes_data = uploaded_file_object.getvalue()
            
            read_options = {'encoding': encoding, 'sep': sep_config}
            if header_config is not None: read_options['header'] = header_config
            if skiprows_config is not None: read_options['skiprows'] = skiprows_config
            if na_values_config is not None: read_options['na_values'] = na_values_config

            df = pd.read_csv(io.BytesIO(bytes_data), **read_options)
            # st.sidebar.info(f"업로드된 파일 '{uploaded_file_object.name}' 로드 성공 (인코딩: {encoding}).") # 디버깅용
            return df
        except UnicodeDecodeError:
            continue # 다음 인코딩 시도
        except Exception as e:
            st.warning(f"업로드된 파일 '{uploaded_file_object.name}' 로드 중 오류 (인코딩: {encoding}): {e}")
            return None
            
    st.error(f"업로드된 파일 '{uploaded_file_object.name}' 로드 실패. 모든 인코딩 시도 실패.")
    return None