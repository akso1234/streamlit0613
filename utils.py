# --- START OF utils.py ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 여전히 디버깅이나 다른 용도로 필요할 수 있음
import os
import json
import requests
import io

def set_korean_font():
    """
    Matplotlib에서 한글 사용을 위한 설정을 수행합니다.
    주요 폰트 설정은 matplotlibrc 파일을 통해 이루어진다고 가정합니다.
    여기서는 unicode_minus 설정만 처리하거나,
    matplotlibrc가 제대로 로드되었는지 간단히 확인할 수 있습니다.
    """
    # print("DEBUG (utils.py): set_korean_font() called.")
    # print(f"DEBUG (utils.py): Current Matplotlib font family (before rcParams): {plt.rcParams['font.family']}")

    # matplotlibrc 파일이 정상적으로 로드되었다면, font.family가 이미 설정되어 있어야 합니다.
    # 이 함수에서는 주로 unicode_minus만 설정합니다.
    plt.rcParams['axes.unicode_minus'] = False

    # 현재 설정된 폰트가 한글 지원 폰트인지 간단히 확인 (선택적 디버깅)
    # current_font_family = plt.rcParams.get('font.family', [''])[0] # 리스트일 수 있음
    # if 'nanum' not in current_font_family.lower() and \
    #    'malgun' not in current_font_family.lower() and \
    #    'apple' not in current_font_family.lower() and \
    #    'noto sans cjk kr' not in current_font_family.lower():
    #     st.sidebar.warning(
    #         f"Matplotlib 기본 폰트가 '{current_font_family}'(으)로 설정되어 있습니다. "
    #         "한글이 깨질 수 있습니다. 'matplotlibrc' 파일과 'packages.txt' 설정을 확인해주세요."
    #     )
    # else:
    #     st.sidebar.info(f"Matplotlib 한글 폰트 '{current_font_family}'(으)로 설정된 것으로 보입니다 (matplotlibrc 확인 필요).")
        
# --- load_csv, load_geojson, load_csv_from_upload 함수는 이전과 동일 ---
# (이전 답변의 해당 함수들 그대로 사용)
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
