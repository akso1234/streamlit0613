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
    """
    Matplotlib에서 한글 사용을 위한 설정을 수행합니다.
    리포지토리에 포함된 폰트 파일을 직접 사용합니다.
    """
    font_found = False
    font_to_set = 'sans-serif' 

    font_filename_in_repo = "NanumGothic.ttf" 
    font_path_in_repo = font_filename_in_repo
    # 만약 'assets/fonts/' 폴더 안에 있다면 아래 주석을 해제하고 위 라인을 주석 처리:
    # font_path_in_repo = os.path.join("assets", "fonts", font_filename_in_repo)

    current_working_dir = os.getcwd()
    absolute_font_path = os.path.join(current_working_dir, font_path_in_repo)

    if os.path.exists(absolute_font_path):
        try:
            fm.fontManager.addfont(absolute_font_path)
            # fm._rebuild() # 일반적으로 addfont 후에는 필요 없을 수 있으며, 시작 시간 지연 가능

            font_prop = fm.FontProperties(fname=absolute_font_path)
            font_name_from_file = font_prop.get_name()
            
            plt.rcParams['font.family'] = font_name_from_file
            
            if plt.rcParams['font.family'] and plt.rcParams['font.family'][0] == font_name_from_file:
                font_found = True
                font_to_set = font_name_from_file
        except Exception:
            font_found = False # 오류 발생 시 명시적으로 False
            pass # 오류 발생 시 조용히 넘어감 (이미 로컬 폴백 로직이 있음)

    if not font_found:
        # 로컬 환경 폴백 (Streamlit Cloud에서는 위 로직이 성공해야 함)
        preferred_system_fonts = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'sans-serif']
        for sys_font_name in preferred_system_fonts:
            try:
                plt.rcParams['font.family'] = sys_font_name
                if plt.rcParams['font.family'][0] == sys_font_name: 
                    font_found = True 
                    font_to_set = sys_font_name
                    break 
            except:
                continue
        if not font_found:
             plt.rcParams['font.family'] = 'sans-serif'
             font_to_set = 'sans-serif'
    
    final_font_family_list_check = plt.rcParams.get('font.family', ['Unknown'])
    final_font_family_check = final_font_family_list_check[0] if final_font_family_list_check else 'Unknown'
    
    expected_korean_font_names_lower_check = ['nanumgothic', 'nanumbarungothic', 'nanumsquare', 'nanummyeongjo', 'malgun gothic', 'applegothic', 'apple sd gothic neo']
    
    is_korean_font_actually_set = any(expected_name in final_font_family_check.lower() for expected_name in expected_korean_font_names_lower_check)

    if not is_korean_font_actually_set:
        if st.runtime.exists():
            st.sidebar.warning(
                 f"한글 폰트 자동 설정에 실패하여 기본 폰트('{final_font_family_check}')가 사용될 수 있습니다. "
                 "그래프의 한글이 깨질 경우, GitHub 리포지토리에 'NanumGothic.ttf'와 같은 한글 폰트 파일을 올바르게 추가했는지 확인해주세요."
            )
    plt.rcParams['axes.unicode_minus'] = False

@st.cache_data
def load_csv(
    file_path, 
    encoding=None, # encoding 인자 추가
    encoding_options=['utf-8-sig', 'utf-8', 'cp949', 'euc-kr'], 
    header_config=None, 
    skiprows_config=None, 
    nrows_config=None, 
    na_values_config=None,
    sep_config=','
):
    full_path = file_path
    
    if not os.path.exists(full_path):
        # st.error(f"데이터 파일을 찾을 수 없습니다: {full_path}") # 이전 코드에서는 st.error 사용
        print(f"데이터 파일을 찾을 수 없습니다: {full_path}") # Streamlit UI 요소 대신 print 사용 (st.cache_data 내부 디버깅용)
        return None

    encodings_to_try = []
    if encoding: # 명시적으로 encoding이 제공된 경우
        encodings_to_try.append(encoding)
    encodings_to_try.extend(enc for enc in encoding_options if enc != encoding) # 나머지 옵션 추가 (중복 방지)


    for enc in encodings_to_try:
        try:
            read_options = {'encoding': enc, 'sep': sep_config}
            if header_config is not None:
                read_options['header'] = header_config
            if skiprows_config is not None:
                read_options['skiprows'] = skiprows_config
            if nrows_config is not None:
                read_options['nrows'] = nrows_config
            if na_values_config is not None:
                read_options['na_values'] = na_values_config
            
            # print(f"Attempting to load {full_path} with encoding: {enc}, options: {read_options}") # 디버깅 로그
            df = pd.read_csv(full_path, **read_options)
            # print(f"Successfully loaded {full_path} with encoding: {enc}") # 디버깅 로그
            return df
        except UnicodeDecodeError:
            # print(f"UnicodeDecodeError with encoding: {enc} for file: {full_path}") # 디버깅 로그
            continue
        except Exception as e:
            # st.warning(f"'{os.path.basename(full_path)}' 파일 로드 중 오류 발생 (인코딩: {enc}): {e}")
            print(f"'{os.path.basename(full_path)}' 파일 로드 중 오류 발생 (인코딩: {enc}): {e}")
            return None # 다른 심각한 오류 발생 시 None 반환
            
    # st.error(f"'{os.path.basename(full_path)}' 파일 로드 실패. 지원되는 인코딩을 찾을 수 없습니다.")
    print(f"'{os.path.basename(full_path)}' 파일 로드 실패. 지원되는 인코딩을 찾을 수 없습니다.")
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
    encoding=None, # encoding 인자 추가
    encoding_options=['utf-8-sig', 'utf-8', 'cp949', 'euc-kr'], 
    header_config=None, 
    skiprows_config=None, 
    na_values_config=None,
    sep_config=','
):
    if _uploaded_file_object is None:
        return None
        
    uploaded_file_object = _uploaded_file_object 
    
    encodings_to_try = []
    if encoding:
        encodings_to_try.append(encoding)
    encodings_to_try.extend(enc for enc in encoding_options if enc != encoding)

    for enc in encodings_to_try:
        try:
            bytes_data = uploaded_file_object.getvalue()
            
            read_options = {'encoding': enc, 'sep': sep_config}
            if header_config is not None: read_options['header'] = header_config
            if skiprows_config is not None: read_options['skiprows'] = skiprows_config
            if na_values_config is not None: read_options['na_values'] = na_values_config

            df = pd.read_csv(io.BytesIO(bytes_data), **read_options)
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.warning(f"업로드된 파일 '{uploaded_file_object.name}' 로드 중 오류 (인코딩: {enc}): {e}")
            return None
            
    st.error(f"업로드된 파일 '{uploaded_file_object.name}' 로드 실패. 모든 인코딩 시도 실패.")
    return None
# --- END OF utils.py ---
