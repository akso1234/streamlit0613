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
    packages.txt로 설치된 나눔 폰트를 찾아 명시적으로 설정하려고 시도합니다.
    """
    # print("DEBUG (utils.py): set_korean_font() CALLED.")
    font_found = False
    
    # 1. 알려진 나눔 폰트 이름 목록 (Streamlit Cloud에서 설치될 가능성이 높은 이름)
    nanum_font_names = ['NanumGothic', 'NanumBarunGothic', 'NanumSquare', 'NanumMyeongjo']
    
    # 2. FontManager를 통해 설치된 폰트 목록에서 직접 찾아 설정 시도
    #    fm._rebuild() # 캐시 재빌드는 마지막 수단으로 고려하거나, 앱 재부팅으로 대체
    
    try:
        font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        # print(f"DEBUG (utils.py): Found {len(font_list)} system fonts.")
        
        # 나눔 폰트 경로 찾기 (대소문자 구분 없이)
        found_nanum_path = None
        for font_path_system in font_list:
            font_filename = os.path.basename(font_path_system).lower()
            if 'nanumgothic.ttf' in font_filename or \
               'nanumbarungothic.ttf' in font_filename or \
               'nanumsquare.ttf' in font_filename or \
               'nanummyeongjo.ttf' in font_filename:
                found_nanum_path = font_path_system
                # print(f"DEBUG (utils.py): Found a Nanum font file: {found_nanum_path}")
                break # 하나 찾으면 사용

        if found_nanum_path:
            # 찾은 폰트 파일 경로를 사용하여 FontProperties 생성 후 rcParams 설정
            font_prop = fm.FontProperties(fname=found_nanum_path)
            font_name_from_path = font_prop.get_name()
            
            plt.rc('font', family=font_name_from_path) # 이름으로 설정
            plt.rcParams['font.family'] = font_name_from_path # 명시적 재설정
            # plt.rcParams['font.sans-serif'] = [font_name_from_path] + plt.rcParams['font.sans-serif'] # sans-serif 목록에도 추가 시도
            font_found = True
            # print(f"DEBUG (utils.py): Successfully set font to '{font_name_from_path}' using path '{found_nanum_path}'.")
        # else:
            # print("DEBUG (utils.py): No Nanum font file found via findSystemFonts scan.")

    except Exception as e:
        # print(f"DEBUG (utils.py): Error during findSystemFonts or setting font by path: {e}")
        pass

    # 3. 위에서 경로 기반 설정에 실패했다면, 이름으로 다시 시도 (fontManager.ttflist 기반)
    if not font_found:
        try:
            available_fm_fonts = [f.name for f in fm.fontManager.ttflist]
            # print(f"DEBUG (utils.py): Fonts in fm.fontManager.ttflist (sample): {available_fm_fonts[:10]}")
            for font_name in nanum_font_names:
                if font_name in available_fm_fonts:
                    plt.rc('font', family=font_name)
                    plt.rcParams['font.family'] = font_name
                    font_found = True
                    # print(f"DEBUG (utils.py): Successfully set font to '{font_name}' by fontManager.ttflist name.")
                    break
                # else:
                    # print(f"DEBUG (utils.py): Font name '{font_name}' not in fontManager.ttflist.")
        except Exception as e:
            # print(f"DEBUG (utils.py): Error while checking fm.fontManager.ttflist: {e}")
            pass

    # 4. 최종 확인 및 경고 (st.set_page_config 이후 호출 가정)
    current_font_family_list = plt.rcParams.get('font.family', ['Unknown'])
    current_font_family = current_font_family_list[0] if current_font_family_list else 'Unknown'

    expected_korean_font_names_lower = ['nanumgothic', 'nanumbarungothic', 'nanumsquare', 'nanummyeongjo']
    is_korean_font_likely_set = any(expected_name in current_font_family.lower() for expected_name in expected_korean_font_names_lower)

    if not is_korean_font_likely_set:
        # print(f"WARNING (utils.py): Final font family '{current_font_family}' does not seem to be a Korean font.")
        st.sidebar.warning(
             f"한글 폰트가 올바르게 설정되지 않은 것 같습니다 (현재 Matplotlib 기본 폰트: {current_font_family}). "
             "그래프의 한글이 깨질 수 있습니다. 'packages.txt' 설정을 확인하고, "
             "앱을 재부팅(Reboot)했는지 확인해주세요. 로그에서 'DEBUG' 메시지를 확인해보세요."
        )
    # else:
        # print(f"INFO (utils.py): Matplotlib font family appears to be set to: {current_font_family}.")

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
