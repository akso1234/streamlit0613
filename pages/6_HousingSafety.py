# --- Streamlit 페이지 레이아웃 ---
def run_housing_safety_page():
    st.title("주거 안전사고와 노후 주택 현황")
    set_korean_font()

    df_rescue_raw = load_csv("data/서울특별시_구조활동현황.csv")
    df_elderly_raw_h = load_csv("data/고령자현황_20250531210628.csv", header_config=[0,1,2,3])
    df_housing_raw_h = load_csv("data/노후기간별+주택현황_20250601054647.csv", header_config=[0,1,2])

    if df_rescue_raw is None or df_elderly_raw_h is None or df_housing_raw_h is None:
        st.error("필수 데이터 파일을 로드하지 못했습니다. 'data' 폴더 내용을 확인해주세요."); return

    df_rescue_processed = preprocess_rescue_data_cached(df_rescue_raw)
    df_elderly_processed_h = preprocess_elderly_data_for_housing_cached(df_elderly_raw_h)
    df_housing_processed_h = preprocess_housing_data_cached(df_housing_raw_h)

    if df_rescue_processed.empty or df_elderly_processed_h.empty or df_housing_processed_h.empty:
        st.error("필수 데이터 처리 중 오류 발생. 위의 메시지를 확인해주세요."); return

    st.write("### 주거 안전사고 현황 (2023년 데이터 기준)")

    # 탭 순서 변경: "시간대별 사고"를 네 번째로 이동
    tab_titles_ordered = ["구별 사고 건수", "구별 사고 원인", "서울시 전체 사고 원인", "시간대별 사고", "상관관계 분석", "사고와의 상관관계 분석"]
    tabs = st.tabs(tab_titles_ordered)

    # 1. 구별 사고 건수
    with tabs[0]:
        st.subheader("구별 총 사고 발생 건수")
        plot_gu_incident_counts(df_rescue_processed)

    # 2. 구별 사고 원인
    with tabs[1]:
        st.subheader("구별 사고원인별 발생 건수")
        top_n_causes_stacked = st.slider("표시할 상위 사고원인 개수:", 3, 15, 7, key="stacked_bar_top_n_slider_housing_main")
        plot_stacked_bar_incident_causes_by_gu(df_rescue_processed, top_n_causes=top_n_causes_stacked)

    # 3. 서울시 전체 사고 원인
    with tabs[2]:
        st.subheader("서울시 전체 주요 사고원인 비율")
        top_n_causes_pie = st.slider("표시할 상위 사고원인 개수:", 3, 10, 7, key="pie_chart_top_n_slider_housing_main")
        plot_pie_major_incident_causes(df_rescue_processed, top_n=top_n_causes_pie)

    # 4. 시간대별 사고 (순서 변경)
    with tabs[3]:
        st.subheader("시간대별 사고 발생 추이")
        if '신고시각' in df_rescue_processed.columns:
            df_rescue_time_analysis = df_rescue_processed.copy()
            df_rescue_time_analysis['신고시간_dt'] = pd.to_datetime(df_rescue_time_analysis['신고시각'], format='%H:%M', errors='coerce')
            if df_rescue_time_analysis['신고시간_dt'].isnull().sum() > len(df_rescue_time_analysis) * 0.8:
                df_rescue_time_analysis['신고시간_dt'] = pd.to_datetime(df_rescue_time_analysis['신고시각'], format='%H:%M:%S', errors='coerce')
            
            if '신고시간_dt' in df_rescue_time_analysis.columns and not df_rescue_time_analysis['신고시간_dt'].isnull().all():
                df_rescue_time_analysis['신고시간(시)'] = df_rescue_time_analysis['신고시간_dt'].dt.hour
                hourly_incidents_counts = df_rescue_time_analysis.dropna(subset=['신고시간(시)'])['신고시간(시)'].astype(int).value_counts().sort_index()
                if not hourly_incidents_counts.empty:
                    fig_time, ax_time = plt.subplots(figsize=(12, 6))
                    sns.lineplot(x=hourly_incidents_counts.index, y=hourly_incidents_counts.values, marker='o', color='indigo', ax=ax_time)
                    ax_time.set_title('시간대별 사고 발생 추이', fontsize=15); ax_time.set_xlabel('신고 시간 (0시 ~ 23시)', fontsize=12); ax_time.set_ylabel('사고 건수', fontsize=12)
                    ax_time.set_xticks(ticks=range(0, 24)); ax_time.grid(True, linestyle='--', alpha=0.7)
                    ax_time.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}')); plt.tight_layout(); st.pyplot(fig_time)
                else: st.info("시간대별 사고 발생 추이 분석을 위한 데이터가 부족합니다.")
            else: st.info("구조활동 데이터의 '신고시각'을 유효한 시간 형식으로 변환하는데 실패했습니다.")
        else: st.info("구조활동 데이터에 '신고시각' 컬럼이 없어 시간대별 분석을 수행할 수 없습니다.")

    # 5. 상관관계 분석 (순서 변경)
    with tabs[4]:
        st.subheader("노인 인구 비율 및 노후 주택 비율의 상관관계 분석")
        if '사고원인' in df_rescue_processed.columns:
            unique_causes_list_h_corr = sorted(df_rescue_processed['사고원인'].unique())
            default_idx_corr = unique_causes_list_h_corr.index('화재') if '화재' in unique_causes_list_h_corr else 0
            selected_cause_corr = st.selectbox("상관관계 분석 사고 원인:", unique_causes_list_h_corr, index=default_idx_corr, key="corr_cause_select_housing_main")
            if selected_cause_corr:
                cause_incidents_df = df_rescue_processed[df_rescue_processed['사고원인'] == selected_cause_corr]['발생장소_구'].value_counts().reset_index()
                cause_incidents_df.columns = ['발생장소_구', f'{selected_cause_corr}건수']

                merged_df_corr_elderly = pd.merge(cause_incidents_df, df_elderly_processed_h, on='발생장소_구', how='inner')
                plot_correlation_scatter_housing(merged_df_corr_elderly, '노인인구수', f'{selected_cause_corr}건수', f"노인 인구수와 {selected_cause_corr} 발생 건수")
                st.divider()
                merged_df_corr_housing = pd.merge(cause_incidents_df, df_housing_processed_h, on='발생장소_구', how='inner')
                plot_correlation_scatter_housing(merged_df_corr_housing, '노후주택수', f'{selected_cause_corr}건수', f"노후 주택수와 {selected_cause_corr} 발생 건수")
        else: st.info("구조활동 데이터에 '사고원인' 컬럼이 없어 상관관계 분석을 수행할 수 없습니다.")

    # 6. 사고와의 상관관계 분석 (순서 변경)
    with tabs[5]:
        st.subheader("노후 주택과 고령 인구의 특정 사고 상관관계 분석")
        unique_causes_list_h_bubble = sorted(df_rescue_processed['사고원인'].unique()) if '사고원인' in df_rescue_processed.columns else ['화재']
        default_idx_bubble = unique_causes_list_h_bubble.index('화재') if '화재' in unique_causes_list_h_bubble else 0
        cause_for_bubble = st.selectbox("버블 크기 기준 사고 원인:", unique_causes_list_h_bubble, index=default_idx_bubble, key="bubble_cause_select_housing_main")
        if cause_for_bubble:
            df_merged_bubble_step1 = pd.merge(df_housing_processed_h, df_elderly_processed_h, on='발생장소_구', how='inner')
            safety_accidents_for_bubble = df_rescue_processed[df_rescue_processed['사고원인'] == cause_for_bubble]['발생장소_구'].value_counts().reset_index()
            safety_accidents_for_bubble.columns = ['발생장소_구', f'{cause_for_bubble}건수']
            df_final_merged_for_bubble = pd.merge(df_merged_bubble_step1, safety_accidents_for_bubble, on='발생장소_구', how='left')
            df_final_merged_for_bubble[f'{cause_for_bubble}건수'] = df_final_merged_for_bubble[f'{cause_for_bubble}건수'].fillna(0).astype(int)
            if not df_final_merged_for_bubble.empty: plot_bubble_chart_housing(df_final_merged_for_bubble, cause_for_bubble)
            else: st.info(f"'{cause_for_bubble}' 사고 기준 종합 분석용 데이터 병합 결과가 비어있습니다.")

if __name__ == "__main__":
    run_housing_safety_page()
