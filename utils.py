# --- START OF utils.py (디버깅 정보를 Streamlit 사이드바에 표시) ---
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
    리포지토리에 포함된 폰트 파일을 직접 사용하고, 디버깅 정보를 사이드바에 표시합니다.
    """
    st.sidebar.subheader("Font Debugging Info:") # 디버깅 섹션 제목
    st.sidebar.write(f"Function `set_korean_font()` CALLED.")
    font_found = False

    font_filename_in_repo = "NanumGothic.ttf" 
    font_path_in_repo = font_filename_in_repo
    # font_path_in_repo = os.path.join("assets", "fonts", font_filename_in_repo) # 하위 폴더라면 이 줄 사용

    st.sidebar.write(f"1. Expected font file: '{font_path_in_repo}' (relative to app root)")
    
    current_working_dir = os.getcwd()
    st.sidebar.write(f"2. Current Working Dir: `{current_working_dir}`")
    
    absolute_font_path_check = os.path.join(current_working_dir, font_path_in_repo)
    st.sidebar.write(f"3. Absolute path being checked: `{absolute_font_path_check}`")

    if os.path.exists(font_path_in_repo):
        st.sidebar.success(f"4. Font file '{font_path_in_repo}' FOUND in app structure.")
        try:
            font_prop = fm.FontProperties(fname=font_path_in_repo)
            font_name_from_file = font_prop.get_name()
            st.sidebar.write(f"5. Extracted font name from file: '{font_name_from_file}'")

            plt.rcParams['font.family'] = font_name_from_file
            font_found = True
            st.sidebar.success(f"6. Successfully SET `plt.rcParams['font.family']` to: '{font_name_from_file}'")
        except Exception as e:
            st.sidebar.error(f"5a. FAILED to process/set font from repo file '{font_path_in_repo}'. Error: {e}")
            font_found = False # 명시적 실패 처리
    else:
        st.sidebar.error(f"4. Font file '{font_path_in_repo}' NOT FOUND in app structure.")
        st.sidebar.write(f"   Files in CWD ('{current_working_dir}'):")
        try:
            # 현재 작업 디렉토리의 파일 목록을 조금 더 보기 쉽게 표시
            cwd_files = os.listdir(current_working_dir)
            if cwd_files:
                for item in cwd_files[:10]: # 너무 많으면 일부만
                    st.sidebar.caption(f"     - {item}")
                if len(cwd_files) > 10:
                    st.sidebar.caption(f"     ... and {len(cwd_files)-10} more items.")
            else:
                st.sidebar.caption("     (No files found in CWD or CWD is not a directory)")
        except Exception as e_ls:
            st.sidebar.caption(f"     Could not list files in CWD: {e_ls}")
        
        # assets/fonts 경로도 확인 (만약 해당 경로를 사용하도록 설정했다면)
        if "assets" in font_path_in_repo:
            assets_fonts_path_check = os.path.join(current_working_dir, "assets", "fonts")
            st.sidebar.write(f"   Checking alternative path: '{assets_fonts_path_check}'")
            if os.path.exists(assets_fonts_path_check) and os.path.isdir(assets_fonts_path_check):
                st.sidebar.write(f"   Files in '{assets_fonts_path_check}':")
                try:
                    for item in os.listdir(assets_fonts_path_check):
                        st.sidebar.caption(f"     - {item}")
                except Exception as e_ls_af:
                    st.sidebar.caption(f"     Could not list files in assets/fonts: {e_ls_af}")
            else:
                st.sidebar.write(f"   Path '{assets_fonts_path_check}' does not exist or is not a directory.")


    # 시스템 폰트 목록에서 이름으로 찾는 시도 (폴백 또는 추가 확인용)
    if not font_found:
        st.sidebar.write("7. Font from repo not set. Attempting system font names...")
        nanum_font_names_system = ['NanumGothic', 'NanumBarunGothic', 'NanumSquare']
        try:
            # fm._rebuild() # 캐시 재빌드는 매우 신중하게, 필요시 주석 해제
            
            font_manager_font_list_names = [f.name for f in fm.fontManager.ttflist]
            st.sidebar.write(f"   Available system font names (sample):")
            displayed_count = 0
            for f_name_fm in font_manager_font_list_names:
                if 'nanum' in f_name_fm.lower() or displayed_count < 5 : # 나눔 계열 또는 처음 5개 표시
                    st.sidebar.caption(f"     - {f_name_fm}")
                    displayed_count +=1
            if len(font_manager_font_list_names) > displayed_count:
                 st.sidebar.caption(f"     ... and more.")


            for name_sys in nanum_font_names_system:
                if name_sys in font_manager_font_list_names:
                    plt.rcParams['font.family'] = name_sys
                    font_found = True # 시스템 폰트라도 찾았으면 True로
                    st.sidebar.success(f"8. Successfully SET `plt.rcParams['font.family']` to system font: '{name_sys}'")
                    break
                else:
                    st.sidebar.caption(f"   System font name '{name_sys}' NOT in fontManager.ttflist.")
        except Exception as e_sys:
            st.sidebar.error(f"8a. Error during system font name check: {e_sys}")

    plt.rcParams['axes.unicode_minus'] = False
    
    final_font_family_list_check = plt.rcParams.get('font.family', ['Unknown'])
    final_font_family_check = final_font_family_list_check[0] if final_font_family_list_check else 'Unknown'
    st.sidebar.write(f"9. Final `plt.rcParams['font.family']` for rendering: **{final_font_family_check}**")
    
    expected_korean_fonts_check = ['nanumgothic', 'nanumbarungothic', 'nanumsquare', 'noto sans cjk kr', 'malgun gothic', 'applegothic', 'apple sd gothic neo']
    if not any(expected_name in final_font_family_check.lower() for expected_name in expected_korean_fonts_check):
        st.sidebar.error(
             f" 최종적으로 설정된 폰트 '{final_font_family_check}'가 한글을 제대로 지원하지 않을 수 있습니다. "
             "그래프의 한글이 깨질 가능성이 높습니다."
        )
    else:
        st.sidebar.success(
            f"최종 폰트 '{final_font_family_check}'가 한글을 지원할 것으로 보입니다."
        )
    st.sidebar.markdown("---") # 디버깅 섹션 끝

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
