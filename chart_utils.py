from matplotlib import font_manager # 이 import는 이제 직접적인 폰트 경로 설정에는 사용되지 않음
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt # Altair 예시 함수를 위해 유지
import pandas as pd
import numpy as np
# import pandas as pd # 중복 임포트 제거

# ——————————————————————————————————————————————————
# 한글 폰트 설정 (Matplotlib용)
# 이제 각 페이지 파일에서 utils.set_korean_font()를 호출하여 전역 설정을 하므로,
# chart_utils.py에서는 이 부분을 제거하거나, 최소한으로 남깁니다.
# plt.rcParams['axes.unicode_minus'] = False 는 set_korean_font()에서도 처리하므로 중복될 수 있으나,
# 여기서도 명시적으로 두는 것이 해롭지는 않습니다.
# ——————————————————————————————————————————————————
# font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc" # macOS 특정 경로 제거
# font_name = font_manager.FontProperties(fname=font_path).get_name() # 제거
# font_manager.fontManager.addfont(font_path) # 제거
# plt.rcParams['font.family'] = font_name # 제거
plt.rcParams['axes.unicode_minus'] = False # 이 설정은 유지
# ——————————————————————————————————————————————————


def draw_example_bar_chart(df: pd.DataFrame):
    """
    예시: Matplotlib으로 수평 막대그래프 그리기
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(df["영역"], df["값"])
    ax.set_xlabel("값(단위)")
    ax.set_ylabel("영역")
    ax.set_title("예시 수평 막대그래프")
    st.pyplot(fig)


def draw_example_altair_scatter(df: pd.DataFrame):
    """
    예시: Altair로 산점도 그리기
    """
    chart = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x="x:Q",
            y="y:Q",
            color="그룹:N",
            tooltip=["x", "y", "그룹"],
        )
        .properties(width=600, height=400, title="예시 Altair 산점도")
    )
    st.altair_chart(chart, use_container_width=True)


def draw_hospital_count_bar_charts(df_hosp: pd.DataFrame):
    """
    df_hosp: load_raw_data()에서 반환된 DataFrame.
             컬럼에는 'gu', '소계', '종합병원', '병원', '의원', '요양병원'이 있어야 함.
    """
    if df_hosp is None or df_hosp.empty:
        st.info("의료기관 수 데이터가 없어 막대 그래프를 그릴 수 없습니다.")
        return

    df_plot = df_hosp.copy()
    if "gu" in df_plot.columns:
        df_plot = df_plot[df_plot["gu"] != "소계"].reset_index(drop=True)
    else: # 'gu' 컬럼이 없으면 진행 불가
        st.error("draw_hospital_count_bar_charts: df_hosp에 'gu' 컬럼이 없습니다.")
        return


    types = ["소계", "종합병원", "병원", "의원", "요양병원"]
    # 필요한 모든 타입 컬럼이 있는지 확인
    missing_types = [t for t in types if t not in df_plot.columns]
    if missing_types:
        st.warning(f"draw_hospital_count_bar_charts: df_plot에 다음 컬럼이 없습니다: {missing_types}. 해당 그래프는 생략될 수 있습니다.")


    for inst in types:
        if inst not in df_plot.columns: # 컬럼이 없으면 해당 그래프 생략
            continue
        df_plot[inst] = pd.to_numeric(df_plot[inst], errors="coerce").fillna(0)

        fig, ax = plt.subplots(figsize=(12, 5)) # 높이 살짝 증가
        bars = ax.bar(df_plot["gu"], df_plot[inst], color='skyblue') # 색상 지정
        ax.set_title(f"구별 {inst} 수", fontsize=15) # 폰트 크기 조정
        ax.set_xlabel("자치구", fontsize=12)
        ax.set_ylabel("기관 수", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10) # x축 레이블 회전 및 정렬, 폰트 크기
        plt.yticks(fontsize=10) # y축 레이블 폰트 크기
        ax.grid(axis='y', linestyle='--', alpha=0.7) # y축 그리드 추가

        # 막대 위에 값 표시 (선택 사항)
        for bar in bars:
            yval = bar.get_height()
            if yval > 0 : # 0보다 큰 값만 표시
                plt.text(bar.get_x() + bar.get_width()/2.0, yval + (df_plot[inst].max()*0.01 if df_plot[inst].max() >0 else 0.1), 
                         f'{int(yval)}', ha='center', va='bottom', fontsize=9)
        
        max_y_val = df_plot[inst].max()
        if max_y_val > 0:
            ax.set_ylim(0, max_y_val * 1.1) # y축 범위 살짝 늘리기

        plt.tight_layout()
        st.pyplot(fig)

def draw_aggregate_hospital_bed_charts(df_hosp: pd.DataFrame, df_beds: pd.DataFrame):
    if df_hosp is None or df_hosp.empty or df_beds is None or df_beds.empty:
        st.info("병원 수 또는 병상 수 데이터가 없어 집계 그래프를 그릴 수 없습니다.")
        return

    df_h = df_hosp.copy()
    df_b = df_beds.copy()
    
    if "gu" in df_h.columns:
        df_h = df_h[df_h["gu"] != "소계"].reset_index(drop=True)
    if "gu" in df_b.columns:
        df_b = df_b[df_b["gu"] != "소계"].reset_index(drop=True)

    types = ["종합병원", "병원", "의원", "요양병원"]
    missing_cols_h = [t for t in types if t not in df_h.columns]
    missing_cols_b = [t for t in types if t not in df_b.columns]
    if missing_cols_h or missing_cols_b:
        st.warning(f"draw_aggregate_hospital_bed_charts: 데이터에 필요한 유형 컬럼이 부족합니다 (병원수: {missing_cols_h}, 병상수: {missing_cols_b}).")
        # 일부 컬럼만으로 계산 시도 또는 중단
        # 여기서는 일단 진행하되, .get(t, 0) 등으로 안전하게 접근
    
    total_hosp = {}
    total_beds = {}
    for t in types:
        total_hosp[t] = pd.to_numeric(df_h.get(t, 0), errors="coerce").sum() # .get으로 안전하게
        total_beds[t] = pd.to_numeric(df_b.get(t, 0), errors="coerce").sum()

    avg_beds = {}
    for t in types:
        if total_hosp.get(t, 0) > 0: # .get으로 안전하게
            avg_beds[t] = total_beds.get(t, 0) / total_hosp.get(t, 0)
        else:
            avg_beds[t] = 0

    # 차트 디자인 일관성 유지
    bar_colors = {'병원 수': 'mediumseagreen', '병상 수': 'cornflowerblue', '평균 병상 수': 'lightcoral'}
    
    fig2, ax2 = plt.subplots(figsize=(8, 4.5)) # 높이 살짝 조정
    ax2.bar(total_hosp.keys(), total_hosp.values(), color=bar_colors['병원 수'])
    ax2.set_title('의료기관 유형별 전체 병원 수', fontsize=15)
    ax2.set_ylabel('병원 수', fontsize=12)
    ax2.set_xlabel('기관 유형', fontsize=12)
    plt.xticks(fontsize=10); plt.yticks(fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig2)

    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(total_beds.keys(), total_beds.values(), color=bar_colors['병상 수'])
    ax1.set_title('의료기관 유형별 전체 병상 수', fontsize=15)
    ax1.set_ylabel('병상 수', fontsize=12)
    ax1.set_xlabel('기관 유형', fontsize=12)
    plt.xticks(fontsize=10); plt.yticks(fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}')) # y축 콤마 포맷
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig1)

    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    ax3.bar(avg_beds.keys(), avg_beds.values(), color=bar_colors['평균 병상 수'])
    ax3.set_title('의료기관 유형별 병원당 평균 병상 수', fontsize=15)
    ax3.set_ylabel('평균 병상 수', fontsize=12)
    ax3.set_xlabel('기관 유형', fontsize=12)
    plt.xticks(fontsize=10); plt.yticks(fontsize=10)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.1f}')) # y축 소수점 한자리 콤마
    ax3.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig3)

def draw_avg_beds_heatmap(df_hosp: pd.DataFrame, df_beds: pd.DataFrame):
    if df_hosp is None or df_hosp.empty or df_beds is None or df_beds.empty:
        st.info("평균 병상 수 히트맵을 그릴 데이터가 없습니다.")
        return None # None 반환 명시

    types_heatmap = ["종합병원", "병원", "요양병원"] # 의원 제외 (원본과 동일하게)

    # 'gu' 컬럼 존재 확인 및 '소계' 행 제거
    if "gu" not in df_hosp.columns or "gu" not in df_beds.columns:
        st.error("draw_avg_beds_heatmap: df_hosp 또는 df_beds에 'gu' 컬럼이 없습니다.")
        return None
        
    df_h = df_hosp[df_hosp["gu"] != "소계"].copy()
    df_b = df_beds[df_beds["gu"] != "소계"].copy()

    # 필요한 유형 컬럼이 있는지 확인
    missing_h = [t for t in types_heatmap if t not in df_h.columns]
    missing_b = [t for t in types_heatmap if t not in df_b.columns]
    if missing_h or missing_b:
        st.warning(f"draw_avg_beds_heatmap: 히트맵 생성에 필요한 유형 컬럼이 부족합니다. (병원수: {missing_h}, 병상수: {missing_b})")
        # 필요한 컬럼만으로 진행하도록 types_heatmap 조정 또는 오류 처리
        types_heatmap = [t for t in types_heatmap if t in df_h.columns and t in df_b.columns]
        if not types_heatmap:
            st.error("draw_avg_beds_heatmap: 히트맵을 그릴 유효한 유형 컬럼이 없습니다.")
            return None

    df_h_indexed = df_h.set_index("gu")[types_heatmap].apply(pd.to_numeric, errors="coerce")
    df_b_indexed = df_b.set_index("gu")[types_heatmap].apply(pd.to_numeric, errors="coerce")
    
    # 공통 'gu'에 대해서만 계산
    common_gus = df_h_indexed.index.intersection(df_b_indexed.index)
    if common_gus.empty:
        st.warning("draw_avg_beds_heatmap: 공통 자치구가 없어 히트맵을 생성할 수 없습니다.")
        return None

    df_h_common = df_h_indexed.loc[common_gus]
    df_b_common = df_b_indexed.loc[common_gus]

    df_avg = df_b_common.divide(df_h_common.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan) # 0으로 나누기 방지
    df_avg = df_avg.fillna(0) # NaN은 0으로 표시 (선택 사항)

    if df_avg.empty:
        st.info("평균 병상 수 계산 결과가 비어 히트맵을 그릴 수 없습니다.")
        return None

    fig, ax = plt.subplots(figsize=(10, 12 if len(df_avg.index) > 15 else 8), dpi=100) # 크기 조정
    cmap = plt.cm.get_cmap("viridis_r", 10) # 컬러맵 변경 및 단계 조절 (역순 viridis)

    im = ax.imshow(
        df_avg.values,
        cmap=cmap,
        aspect="auto",
        vmin=0, # 최소값을 0으로 고정 (음수 방지 및 색상 스케일 일관성)
        vmax=np.nanmax(df_avg.values) if not df_avg.isnull().all().all() else 1 # 전체가 NaN이 아닐때만 vmax 설정
    )

    ax.set_xticks(np.arange(len(df_avg.columns))) # types_heatmap 대신 df_avg.columns 사용
    ax.set_xticklabels(df_avg.columns, fontsize=12, fontweight='bold') # 폰트 크기 조정
    ax.set_yticks(np.arange(len(df_avg.index)))
    ax.set_yticklabels(df_avg.index, fontsize=10) # 폰트 크기 조정
    ax.set_xlabel("기관 유형", fontsize=14, fontweight='bold')
    ax.set_ylabel("자치구", fontsize=14, fontweight='bold')
    ax.set_title("구별 기관 유형별 평균 병상 수", fontsize=16, fontweight='bold')

    data_values = df_avg.values
    min_val_for_text = np.nanmin(data_values[~np.isinf(data_values)]) if np.sum(~np.isnan(data_values) & ~np.isinf(data_values)) > 0 else 0
    max_val_for_text = np.nanmax(data_values[~np.isinf(data_values)]) if np.sum(~np.isnan(data_values) & ~np.isinf(data_values)) > 0 else 1
    
    for (i, j), val in np.ndenumerate(data_values):
        text = "-" if np.isnan(val) or np.isinf(val) else f"{val:.1f}"
        # 텍스트 색상 결정 로직 개선
        if pd.notna(val) and not np.isinf(val) and max_val_for_text > min_val_for_text:
            normalized_val = (val - min_val_for_text) / (max_val_for_text - min_val_for_text)
            text_color = "white" if normalized_val > 0.6 else ("black" if normalized_val < 0.3 else "grey") # 중간값에 가까우면 회색
        else:
            text_color = "black" # 기본값

        ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=9, fontweight='normal') # 폰트 크기/굵기 조정

    ax.set_xticks(np.arange(len(df_avg.columns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(df_avg.index) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.8) # 그리드 색상/두께 조정
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("평균 병상 수", fontsize=12, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    return df_avg


def draw_sheet0_charts(
    df_metrics,
    figsize1: tuple = (14, 6), # 크기 조정
    figsize2: tuple = (14, 6), # 크기 조정
    figsize3: tuple = (14, 7), # 크기 조정
    dpi: int = 100
) -> None:
    if df_metrics is None or df_metrics.empty:
        st.info("노인주거복지시설 데이터가 없어 차트를 그릴 수 없습니다.")
        return
        
    regions = df_metrics.index.tolist()
    x = np.arange(len(regions))
    n_regions = len(regions)
    
    # 막대 너비 동적 조절 (너무 빽빽하지 않게)
    base_width = 0.8 # 전체 x축 공간에서 막대가 차지할 비율
    num_groups_fig1 = 3
    bar_width_fig1 = base_width / num_groups_fig1 * 0.8 # 그룹 간 간격 고려

    # (1) 정원·현원·추가 수용
    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    rects1 = ax1.bar(x - bar_width_fig1, df_metrics['capacity'],   width=bar_width_fig1, label='정원', color='cornflowerblue')
    rects2 = ax1.bar(x,                  df_metrics['occupancy'], width=bar_width_fig1, label='현원', color='salmon')
    rects3 = ax1.bar(x + bar_width_fig1, df_metrics['additional'], width=bar_width_fig1, label='추가 수용', color='lightgreen')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax1.set_ylabel('인원 수 (명)', fontsize=12)
    ax1.set_title('구별 노인주거복지시설: 정원·현원·추가 수용 가능 인원', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val):,}'))


    # (2) 시설수·종사자수
    num_groups_fig2 = 2
    bar_width_fig2 = base_width / num_groups_fig2 * 0.7
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    rects4 = ax2.bar(x - bar_width_fig2/2, df_metrics['facility'], width=bar_width_fig2, label='시설수 (개소)', color='skyblue')
    # 종사자수는 다른 y축 사용 (스케일 차이 클 경우)
    ax2b = ax2.twinx()
    rects5 = ax2b.bar(x + bar_width_fig2/2, df_metrics['staff'],    width=bar_width_fig2, label='종사자수 (명)', color='lightcoral')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax2.set_title('구별 노인주거복지시설: 시설수 및 종사자수', fontsize=15, fontweight='bold')
    ax2.set_ylabel('시설 수 (개소)', fontsize=12, color='skyblue')
    ax2b.set_ylabel('종사자 수 (명)', fontsize=12, color='lightcoral')
    ax2.tick_params(axis='y', labelcolor='skyblue')
    ax2b.tick_params(axis='y', labelcolor='lightcoral')
    # 범례 통합
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.3)


    # (3) 정원/종사자 vs 현원/종사자 비율
    fig3, ax3 = plt.subplots(figsize=figsize3, dpi=dpi)
    rects6 = ax3.bar(x - bar_width_fig2/2, df_metrics['cap_per_staff'], width=bar_width_fig2, label='종사자 1인당 정원', color='mediumseagreen')
    rects7 = ax3.bar(x + bar_width_fig2/2, df_metrics['occ_per_staff'], width=bar_width_fig2, label='종사자 1인당 현원', color='mediumpurple')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax3.set_ylabel('담당 인원 수 (명/종사자 1인)', fontsize=12)
    ax3.set_title('구별 노인주거복지시설: 종사자 1인당 담당 인원 비교', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(axis='y', linestyle=':', alpha=0.7)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.1f}'))

    for fig_item in [fig1, fig2, fig3]: # 모든 차트에 tight_layout 적용
        fig_item.tight_layout() 
        st.pyplot(fig_item)


def draw_sheet1_charts(
    df_metrics,
    figsize1: tuple = (14, 6),
    figsize2: tuple = (14, 6),
    figsize3: tuple = (14, 7),
    dpi: int = 100
) -> None:
    if df_metrics is None or df_metrics.empty:
        st.info("노인의료복지시설 데이터가 없어 차트를 그릴 수 없습니다.")
        return

    regions = df_metrics.index.tolist()
    x = np.arange(len(regions))
    n_regions = len(regions)
    base_width = 0.8
    num_groups_fig1 = 3
    bar_width_fig1 = base_width / num_groups_fig1 * 0.8

    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    ax1.bar(x - bar_width_fig1, df_metrics['capacity'],   width=bar_width_fig1, label='정원', color='cornflowerblue')
    ax1.bar(x,                  df_metrics['occupancy'], width=bar_width_fig1, label='현원', color='salmon')
    ax1.bar(x + bar_width_fig1, df_metrics['additional'], width=bar_width_fig1, label='추가 수용', color='lightgreen')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax1.set_ylabel('인원 수 (명)', fontsize=12)
    ax1.set_title('구별 노인의료복지시설: 정원·현원·추가 수용 가능 인원', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val):,}'))

    num_groups_fig2 = 2
    bar_width_fig2 = base_width / num_groups_fig2 * 0.7
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    ax2.bar(x - bar_width_fig2/2, df_metrics['facility'], width=bar_width_fig2, label='시설수 (개소)', color='skyblue')
    ax2b = ax2.twinx()
    ax2b.bar(x + bar_width_fig2/2, df_metrics['staff'],    width=bar_width_fig2, label='종사자수 (명)', color='lightcoral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax2.set_title('구별 노인의료복지시설: 시설수 및 종사자수', fontsize=15, fontweight='bold')
    ax2.set_ylabel('시설 수 (개소)', fontsize=12, color='skyblue')
    ax2b.set_ylabel('종사자 수 (명)', fontsize=12, color='lightcoral')
    ax2.tick_params(axis='y', labelcolor='skyblue')
    ax2b.tick_params(axis='y', labelcolor='lightcoral')
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.3)

    fig3, ax3 = plt.subplots(figsize=figsize3, dpi=dpi)
    ax3.bar(x - bar_width_fig2/2, df_metrics['cap_per_staff'], width=bar_width_fig2, label='종사자 1인당 정원', color='mediumseagreen')
    ax3.bar(x + bar_width_fig2/2, df_metrics['occ_per_staff'], width=bar_width_fig2, label='종사자 1인당 현원', color='mediumpurple')
    ax3.set_xticks(x)
    ax3.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax3.set_ylabel('담당 인원 수 (명/종사자 1인)', fontsize=12)
    ax3.set_title('구별 노인의료복지시설: 종사자 1인당 담당 인원 비교', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(axis='y', linestyle=':', alpha=0.7)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.1f}'))

    for fig_item in [fig1, fig2, fig3]:
        fig_item.tight_layout()
        st.pyplot(fig_item)


def draw_nursing_csv_charts(
    df_welf: pd.DataFrame,
    df_centers: pd.DataFrame,
    figsize1: tuple = (14, 6), # 크기 조정
    figsize2: tuple = (14, 5),
    dpi: int = 100
) -> None:
    if (df_welf is None or df_welf.empty) and \
       (df_centers is None or df_centers.empty): # 둘 다 없으면 경고
        st.info("노인여가복지시설(CSV) 데이터가 없어 차트를 그릴 수 없습니다.")
        return

    if df_welf is not None and not df_welf.empty:
        regions_welf = df_welf.index.tolist()
        x_welf = np.arange(len(regions_welf))
        n_regions_welf = len(regions_welf)
        base_width_welf = 0.8
        num_groups_welf = 2
        bar_width_welf = base_width_welf / num_groups_welf * 0.7

        fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
        ax1.bar(x_welf - bar_width_welf/2, df_welf['facility'], width=bar_width_welf, label='시설수 (개소)', color='deepskyblue')
        ax1b = ax1.twinx() # 종사자수는 다른 y축 사용
        ax1b.bar(x_welf + bar_width_welf/2, df_welf['staff'],    width=bar_width_welf, label='종사자수 (명)', color='tomato')
        
        ax1.set_xticks(x_welf)
        ax1.set_xticklabels(regions_welf, rotation=45, ha='right', fontsize=10 if n_regions_welf <= 15 else 8)
        ax1.set_title('구별 노인복지관: 시설수 및 종사자수', fontsize=15, fontweight='bold')
        ax1.set_ylabel('시설 수 (개소)', fontsize=12, color='deepskyblue')
        ax1b.set_ylabel('종사자 수 (명)', fontsize=12, color='tomato')
        ax1.tick_params(axis='y', labelcolor='deepskyblue')
        ax1b.tick_params(axis='y', labelcolor='tomato')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1b.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)
        ax1.grid(axis='y', linestyle=':', alpha=0.3)
        fig1.tight_layout()
        st.pyplot(fig1)
    else:
        st.info("노인복지관(CSV) 데이터가 없어 '시설수 vs 종사자수' 차트를 그릴 수 없습니다.")

    if df_centers is not None and not df_centers.empty:
        regions_centers = df_centers.index.tolist()
        x_centers = np.arange(len(regions_centers))
        n_regions_centers = len(regions_centers)
        bar_width_centers = 0.6 # 단일 막대이므로 조금 넓게

        fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
        ax2.bar(x_centers, df_centers['facility'], width=bar_width_centers, color='mediumseagreen', label='시설수 (개소)')
        ax2.set_xticks(x_centers)
        ax2.set_xticklabels(regions_centers, rotation=45, ha='right', fontsize=10 if n_regions_centers <= 15 else 8)
        ax2.set_ylabel('시설 수 (개소)', fontsize=12)
        ax2.set_title('구별 경로당 및 노인교실 총 시설수', fontsize=15, fontweight='bold')
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(axis='y', linestyle=':', alpha=0.7)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val):,}'))
        fig2.tight_layout()
        st.pyplot(fig2)
    else:
        st.info("경로당+노인교실(CSV) 데이터가 없어 '총 시설수' 차트를 그릴 수 없습니다.")


def draw_sheet3_charts(
    df_metrics,
    figsize1: tuple = (14, 6),
    figsize2: tuple = (14, 6),
    figsize3: tuple = (14, 7),
    dpi: int = 100
) -> None:
    if df_metrics is None or df_metrics.empty:
        st.info("재가노인복지시설 데이터가 없어 차트를 그릴 수 없습니다.")
        return

    regions = df_metrics.index.tolist()
    x = np.arange(len(regions))
    n_regions = len(regions)
    base_width = 0.8
    
    num_groups_fig1 = 2 # 정원, 현원
    bar_width_fig1 = base_width / num_groups_fig1 * 0.8

    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    ax1.bar(x - bar_width_fig1/2, df_metrics['capacity'],   width=bar_width_fig1, label='정원', color='cornflowerblue')
    ax1.bar(x + bar_width_fig1/2, df_metrics['occupancy'], width=bar_width_fig1, label='현원', color='salmon')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax1.set_ylabel('인원 수 (명)', fontsize=12)
    ax1.set_title('구별 재가노인복지시설: 정원·현원 인원수', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val):,}'))

    num_groups_fig2 = 2
    bar_width_fig2 = base_width / num_groups_fig2 * 0.7
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    ax2.bar(x - bar_width_fig2/2, df_metrics['facility'], width=bar_width_fig2, label='시설수 (개소)', color='skyblue')
    ax2b = ax2.twinx()
    ax2b.bar(x + bar_width_fig2/2, df_metrics['staff'],    width=bar_width_fig2, label='종사자수 (명)', color='lightcoral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax2.set_title('구별 재가노인복지시설: 시설수 및 종사자수', fontsize=15, fontweight='bold')
    ax2.set_ylabel('시설 수 (개소)', fontsize=12, color='skyblue')
    ax2b.set_ylabel('종사자 수 (명)', fontsize=12, color='lightcoral')
    ax2.tick_params(axis='y', labelcolor='skyblue')
    ax2b.tick_params(axis='y', labelcolor='lightcoral')
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.3)

    fig3, ax3 = plt.subplots(figsize=figsize3, dpi=dpi)
    ax3.bar(x - bar_width_fig2/2, df_metrics['cap_per_staff'], width=bar_width_fig2, label='종사자 1인당 정원', color='mediumseagreen')
    ax3.bar(x + bar_width_fig2/2, df_metrics['occ_per_staff'], width=bar_width_fig2, label='종사자 1인당 현원', color='mediumpurple')
    ax3.set_xticks(x)
    ax3.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax3.set_ylabel('담당 인원 수 (명/종사자 1인)', fontsize=12)
    ax3.set_title('구별 재가노인복지시설: 종사자 1인당 담당 인원 비교', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(axis='y', linestyle=':', alpha=0.7)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.1f}'))

    for fig_item in [fig1, fig2, fig3]:
        fig_item.tight_layout()
        st.pyplot(fig_item)


def draw_sheet4_charts(
    df_metrics,
    figsize1: tuple = (14, 6), # figsize 통일
    dpi: int = 100
) -> None:
    if df_metrics is None or df_metrics.empty:
        st.info("노인일자리지원기관 데이터가 없어 차트를 그릴 수 없습니다.")
        return
        
    regions = df_metrics.index.tolist()
    x = np.arange(len(regions))
    n_regions = len(regions)
    base_width = 0.8
    num_groups = 2
    bar_width = base_width / num_groups * 0.7

    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    ax1.bar(x - bar_width/2, df_metrics['facility'], width=bar_width, label='시설수 (개소)', color='skyblue')
    ax1b = ax1.twinx() # 종사자수는 다른 y축 사용
    ax1b.bar(x + bar_width/2, df_metrics['staff'],    width=bar_width, label='종사자수 (명)', color='lightcoral')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax1.set_title('구별 노인일자리지원기관: 시설수 및 종사자수', fontsize=15, fontweight='bold')
    ax1.set_ylabel('시설 수 (개소)', fontsize=12, color='skyblue')
    ax1b.set_ylabel('종사자 수 (명)', fontsize=12, color='lightcoral')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1b.tick_params(axis='y', labelcolor='lightcoral')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1b.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)
    ax1.grid(axis='y', linestyle=':', alpha=0.3) # ax1에 대한 그리드
    
    fig1.tight_layout()
    st.pyplot(fig1)

def draw_sheet5_charts(
    df_metrics,
    figsize1: tuple = (14, 6),
    figsize2: tuple = (14, 6),
    figsize3: tuple = (14, 7),
    dpi: int = 100
) -> None:
    if df_metrics is None or df_metrics.empty:
        st.info("치매전담형 장기요양기관 데이터가 없어 차트를 그릴 수 없습니다.")
        return

    regions = df_metrics.index.tolist()
    x = np.arange(len(regions))
    n_regions = len(regions)
    base_width = 0.8
    num_groups_fig1 = 3
    bar_width_fig1 = base_width / num_groups_fig1 * 0.8

    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    ax1.bar(x - bar_width_fig1, df_metrics['capacity'],   width=bar_width_fig1, label='정원', color='cornflowerblue')
    ax1.bar(x,                  df_metrics['occupancy'], width=bar_width_fig1, label='현원', color='salmon')
    ax1.bar(x + bar_width_fig1, df_metrics['additional'], width=bar_width_fig1, label='추가 수용', color='lightgreen')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax1.set_ylabel('인원 수 (명)', fontsize=12)
    ax1.set_title('구별 치매전담형 장기요양기관: 정원·현원·추가 수용 가능 인원', 
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val):,}'))

    num_groups_fig2 = 2
    bar_width_fig2 = base_width / num_groups_fig2 * 0.7
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    ax2.bar(x - bar_width_fig2/2, df_metrics['facility'], width=bar_width_fig2, label='시설수 (개소)', color='skyblue')
    ax2b = ax2.twinx()
    ax2b.bar(x + bar_width_fig2/2, df_metrics['staff'],    width=bar_width_fig2, label='종사자수 (명)', color='lightcoral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax2.set_title('구별 치매전담형 장기요양기관: 시설수 및 종사자수', 
                  fontsize=15, fontweight='bold')
    ax2.set_ylabel('시설 수 (개소)', fontsize=12, color='skyblue')
    ax2b.set_ylabel('종사자 수 (명)', fontsize=12, color='lightcoral')
    ax2.tick_params(axis='y', labelcolor='skyblue')
    ax2b.tick_params(axis='y', labelcolor='lightcoral')
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.3)

    fig3, ax3 = plt.subplots(figsize=figsize3, dpi=dpi)
    ax3.bar(x - bar_width_fig2/2, df_metrics['cap_per_staff'], width=bar_width_fig2, label='종사자 1인당 정원', color='mediumseagreen')
    ax3.bar(x + bar_width_fig2/2, df_metrics['occ_per_staff'], width=bar_width_fig2, label='종사자 1인당 현원', color='mediumpurple')
    ax3.set_xticks(x)
    ax3.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax3.set_ylabel('담당 인원 수 (명/종사자 1인)', fontsize=12)
    ax3.set_title('구별 치매전담형 장기요양기관: 종사자 1인당 담당 인원 비교', 
                  fontsize=15, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(axis='y', linestyle=':', alpha=0.7)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.1f}'))

    for fig_item in [fig1, fig2, fig3]:
        fig_item.tight_layout()
        st.pyplot(fig_item)
