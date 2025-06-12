    if not font_found:
        # st.sidebar.warning( # <--- 이 라인 또는 유사한 st.sidebar.warning 라인을 주석 처리하거나 삭제합니다.
        #      f"리포지토리에 포함된 한글 폰트 파일 ('{font_filename_in_repo}')을 찾거나 설정할 수 없습니다. "
        #      "그래프의 한글이 깨질 수 있습니다. 파일 경로와 GitHub 리포지토리를 확인해주세요."
        # )
        print(f"WARNING (utils.py): Korean font setup from repo file FAILED. Font may be broken if no system font is found.") # 로그는 남겨둘 수 있습니다.
        plt.rcParams['font.family'] = 'sans-serif' 
        font_to_set = 'sans-serif'
    # else: # 성공 시 메시지도 필요 없다면 주석 처리 또는 삭제
        # st.sidebar.success(f"한글 폰트가 성공적으로 설정되었습니다: {font_to_set}")
        # print(f"INFO (utils.py): Matplotlib font family appears to be set to: {font_to_set}.")
