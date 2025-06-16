@st.cache_data
def preprocess_lonely_elderly_data_revised_cached(file_path):
    df_raw = load_csv(file_path, header_config=[0,1,2,3], encoding='utf-8-sig')
    
    if df_raw is None or df_raw.empty:
        st.warning(f"독거노인 파일 '{os.path.basename(file_path)}'을 로드하지 못했거나 비어있습니다.")
        return pd.DataFrame(columns=['시군구', '연도', '독거노인수', '성별'])

    try:
        sido_col_name_tuple = df_raw.columns[0]
        sigungu_col_name_tuple = df_raw.columns[1]
    except IndexError:
        st.error("독거노인 파일의 컬럼 구조가 예상과 다릅니다 (최소 2개 이상의 컬럼이 필요).")
        return pd.DataFrame(columns=['시군구', '연도', '독거노인수', '성별'])

    df_seoul = df_raw[df_raw[sido_col_name_tuple] == '서울특별시'].copy()
    df_gu_data = df_seoul[df_seoul[sigungu_col_name_tuple].str.endswith('구', na=False)].copy()

    if df_gu_data.empty:
        st.warning("독거노인 데이터에서 구별 데이터를 찾을 수 없습니다.")
        return pd.DataFrame(columns=['시군구', '연도', '독거노인수', '성별'])
    
    # ---!!! 중요 디버깅 코드 시작 !!!---
    print("DEBUG (Dokgo): df_gu_data.columns (first 30 columns):")
    for i, col_tuple in enumerate(df_gu_data.columns):
        if i < 30 : # 너무 많으면 잘라서 출력
            print(f"  Col {i}: {col_tuple} (Type: {type(col_tuple)})")
        else:
            break
    print("DEBUG (Dokgo): End of df_gu_data.columns print")
    # ---!!! 중요 디버깅 코드 끝 !!!---

    result_data = []
    available_years_in_file = sorted(list(set(str(c[0]).strip() for c in df_gu_data.columns if isinstance(c, tuple) and len(c) > 0 and str(c[0]).strip().isdigit())))
    
    print(f"DEBUG (Dokgo): Available years extracted from df_gu_data columns: {available_years_in_file}")

    for year_str in available_years_in_file:
        if not (2021 <= int(year_str) <= 2023): 
            continue
        
        level1_expected = '합계'
        level2_expected = '소계'
        level3_expected = '계'

        target_col_found = None
        # ---!!! 중요 디버깅 코드 시작 !!!---
        print(f"DEBUG (Dokgo): Searching for target column for year: {year_str}")
        print(f"  Expected tuple structure: ({year_str}, '{level1_expected}', '{level2_expected}', '{level3_expected}')")
        # ---!!! 중요 디버깅 코드 끝 !!!---

        for col_tuple_from_df in df_gu_data.columns:
            if isinstance(col_tuple_from_df, tuple) and len(col_tuple_from_df) == 4:
                l0, l1, l2, l3 = str(col_tuple_from_df[0]).strip(), str(col_tuple_from_df[1]).strip(), str(col_tuple_from_df[2]).strip(), str(col_tuple_from_df[3]).strip()
                if l0 == year_str and l1 == level1_expected and l2 == level2_expected and l3 == level3_expected:
                    target_col_found = col_tuple_from_df
                    # ---!!! 중요 디버깅 코드 시작 !!!---
                    print(f"  SUCCESS: Found target column for {year_str}: {target_col_found}")
                    # ---!!! 중요 디버깅 코드 끝 !!!---
                    break
        
        if target_col_found is None:
            # ---!!! 중요 디버깅 코드 시작 !!!---
            print(f"  WARNING (Dokgo): {year_str}년 ('{level1_expected}', '{level2_expected}', '{level3_expected}') 컬럼을 찾지 못했습니다. Skipping this year.")
            # ---!!! 중요 디버깅 코드 끝 !!!---
            continue
            
        try:
            cols_to_select = [sigungu_col_name_tuple, target_col_found]
            if not all(col in df_gu_data.columns for col in cols_to_select):
                # st.warning(f"{year_str}년 데이터 추출 시 필요한 컬럼이 df_gu_data에 없습니다. 선택된 컬럼: {cols_to_select}")
                continue

            year_data = df_gu_data[cols_to_select].copy()
            
            year_data.rename(columns={
                sigungu_col_name_tuple: '시군구',
                target_col_found: '독거노인수'
            }, inplace=True)
            
            if '독거노인수' not in year_data.columns:
                # st.error(f"{year_str}년 데이터 rename 후 '독거노인수' 컬럼이 없습니다. 원본 컬럼: {target_col_found}")
                continue

            year_data['연도'] = int(year_str)
            result_data.append(year_data)
        except KeyError as ke:
            # st.warning(f"{year_str}년 독거노인 데이터 처리 중 KeyError: {ke}. 컬럼명: {cols_to_select}")
            continue
        except Exception as e:
            # st.warning(f"{year_str}년 독거노인 데이터 처리 중 예외 발생: {e}")
            continue
            
    if not result_data: # 이 부분에서 st.warning이 발생한 것임
        st.warning("처리할 수 있는 연도의 독거노인 데이터가 없습니다. (result_data is empty for Dokgo)")
        return pd.DataFrame(columns=['시군구', '연도', '독거노인수', '성별'])

    df_final_dokgo = pd.concat(result_data, ignore_index=True)
    
    if '독거노인수' not in df_final_dokgo.columns:
        st.error("독거노인 데이터 결합 후 '독거노인수' 컬럼이 생성되지 않았습니다. (Dokgo) 최종 DataFrame 컬럼: " + str(df_final_dokgo.columns.tolist()))
        return pd.DataFrame(columns=['시군구', '연도', '독거노인수', '성별'])

    df_final_dokgo['독거노인수'] = pd.to_numeric(df_final_dokgo['독거노인수'], errors='coerce').fillna(0).astype(int)
    df_final_dokgo['성별'] = '전체'
    
    return df_final_dokgo
