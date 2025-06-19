# --- START OF MODIFIED FILE chart_utils.py ---
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt # Altair 예시 함수를 위해 유지
import pandas as pd
import numpy as np

# ——————————————————————————————————————————————————
# 한글 폰트 설정 (Matplotlib용)
# 각 페이지 파일에서 utils.set_korean_font()를 호출하여 전역 설정을 하므로,
# chart_utils.py에서는 plt.rcParams['axes.unicode_minus'] = False 만 유지합니다.
# ——————————————————————————————————————————————————
plt.rcParams['axes.unicode_minus'] = False 
# ——————————————————————————————————————————————————


def draw_example_bar_chart(df: pd.DataFrame):
    """
    예시: Matplotlib으로 수평 막대그래프 그리기
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(df["영역"], df["값"], label="값")
    ax.set_xlabel("값(단위)")
    ax.set_ylabel("영역")
    ax.set_title("예시 수평 막대그래프")
    ax.legend(fontsize=10, loc='upper right')
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

    df_plot_original = df_hosp.copy()
    if "gu" in df_plot_original.columns:
        df_plot_original = df_plot_original[df_plot_original["gu"] != "소계"].reset_index(drop=True)
    else: 
        st.error("draw_hospital_count_bar_charts: df_hosp에 'gu' 컬럼이 없습니다.")
        return

    types = ["소계", "종합병원", "병원", "의원", "요양병원"]
    missing_types = [t for t in types if t not in df_plot_original.columns]
    if missing_types:
        st.warning(f"draw_hospital_count_bar_charts: df_plot에 다음 컬럼이 없습니다: {missing_types}. 해당 그래프는 생략될 수 있습니다.")

    for inst in types:
        if inst not in df_plot_original.columns: 
            continue
        
        df_plot = df_plot_original.copy() # 매번 원본에서 복사하여 정렬
        df_plot[inst] = pd.to_numeric(df_plot[inst], errors="coerce").fillna(0)

        # 2. Sort bars in descending order
        df_plot = df_plot.sort_values(by=inst, ascending=False).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(12, 5)) 
        # 3. Bring bars in front of grid lines (zorder)
        bars = ax.bar(df_plot["gu"], df_plot[inst], color='skyblue', label=inst, zorder=3) 
        ax.set_title(f"서울시 자치구별 {inst} 수", fontsize=15) 
        ax.set_xlabel("자치구", fontsize=12)
        ax.set_ylabel("기관 수", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10) 
        plt.yticks(fontsize=10)
        # 3. Bring bars in front of grid lines (zorder for grid)
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0) 

        # 1. Remove numbers on top of bars
        # for bar in bars:
        #     yval = bar.get_height()
        #     if yval > 0 : 
        #         max_val_for_offset = df_plot[inst].max() if not df_plot[inst].empty else yval # offset 계산을 위한 max 값
        #         plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max_val_for_offset*0.01 if max_val_for_offset >0 else 0.1), 
        #                  f'{int(yval)}', ha='center', va='bottom', fontsize=9)
        
        if not df_plot[inst].empty and df_plot[inst].max() > 0 :
             ax.set_ylim(0, df_plot[inst].max() * 1.15)
        
        ax.legend(fontsize=10, loc='upper right')
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
    if not all(t in df_h.columns for t in types) or not all(t in df_b.columns for t in types):
        st.warning("집계 병상/병원 차트: 필요한 유형 컬럼이 데이터에 없습니다.")
    
    total_hosp = {}
    total_beds = {}
    for t in types:
        total_hosp[t] = pd.to_numeric(df_h.get(t, 0), errors="coerce").sum()
        total_beds[t] = pd.to_numeric(df_b.get(t, 0), errors="coerce").sum()

    avg_beds = {}
    for t in types:
        hosp_count = total_hosp.get(t, 0)
        if hosp_count > 0:
            avg_beds[t] = total_beds.get(t, 0) / hosp_count
        else:
            avg_beds[t] = 0.0

    bar_colors = {'병원 수': 'mediumseagreen', '병상 수': 'cornflowerblue', '평균 병상 수': 'lightcoral'}
    
    fig_hosp, ax_hosp = plt.subplots(figsize=(8, 4.5)) 
    ax_hosp.bar(total_hosp.keys(), total_hosp.values(), color=bar_colors['병원 수'], label='병원 수')
    ax_hosp.set_title('의료기관 유형별 전체 병원 수', fontsize=15) 
    ax_hosp.set_ylabel('병원 수', fontsize=12)
    ax_hosp.set_xlabel('기관 유형', fontsize=12)
    # 4. Rotate x-axis labels
    plt.xticks(rotation=45, ha="right", fontsize=10); plt.yticks(fontsize=10)
    ax_hosp.grid(axis='y', linestyle=':', alpha=0.6)
    ax_hosp.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    st.pyplot(fig_hosp)

    fig_beds, ax_beds = plt.subplots(figsize=(8, 4.5))
    ax_beds.bar(total_beds.keys(), total_beds.values(), color=bar_colors['병상 수'], label='병상 수')
    ax_beds.set_title('의료기관 유형별 전체 병상 수', fontsize=15)
    ax_beds.set_ylabel('병상 수', fontsize=12)
    ax_beds.set_xlabel('기관 유형', fontsize=12)
    # 4. Rotate x-axis labels
    plt.xticks(rotation=45, ha="right", fontsize=10); plt.yticks(fontsize=10)
    ax_beds.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}')) 
    ax_beds.grid(axis='y', linestyle=':', alpha=0.6)
    ax_beds.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    st.pyplot(fig_beds)

    fig_avg_beds, ax_avg_beds = plt.subplots(figsize=(8, 4.5))
    ax_avg_beds.bar(avg_beds.keys(), avg_beds.values(), color=bar_colors['평균 병상 수'], label='평균 병상 수')
    ax_avg_beds.set_title('의료기관 유형별 병원당 평균 병상 수', fontsize=15)
    ax_avg_beds.set_ylabel('평균 병상 수', fontsize=12)
    ax_avg_beds.set_xlabel('기관 유형', fontsize=12)
    # 4. Rotate x-axis labels
    plt.xticks(rotation=45, ha="right", fontsize=10); plt.yticks(fontsize=10)
    ax_avg_beds.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.1f}')) 
    ax_avg_beds.grid(axis='y', linestyle=':', alpha=0.6)
    ax_avg_beds.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    st.pyplot(fig_avg_beds)

def draw_avg_beds_heatmap(df_hosp: pd.DataFrame, df_beds: pd.DataFrame):
    if df_hosp is None or df_hosp.empty or df_beds is None or df_beds.empty:
        st.info("평균 병상 수 히트맵을 그릴 데이터가 없습니다.")
        return None 

    types = ["종합병원", "병원", "요양병원"] 

    if "gu" not in df_hosp.columns or "gu" not in df_beds.columns:
        st.error("draw_avg_beds_heatmap: df_hosp 또는 df_beds에 'gu' 컬럼이 없습니다.")
        return None
        
    df_h = df_hosp[df_hosp["gu"] != "소계"].copy() 
    df_b = df_beds[df_beds["gu"] != "소계"].copy() 

    df_h_indexed = df_h.set_index("gu")[[col for col in types if col in df_h.columns]]
    df_b_indexed = df_b.set_index("gu")[[col for col in types if col in df_b.columns]]
    
    df_h_numeric = df_h_indexed.apply(pd.to_numeric, errors="coerce")
    df_b_numeric = df_b_indexed.apply(pd.to_numeric, errors="coerce")

    common_gus = df_h_numeric.index.intersection(df_b_numeric.index)
    common_types_heatmap = [t for t in types if t in df_h_numeric.columns and t in df_b_numeric.columns]

    if common_gus.empty or not common_types_heatmap:
        st.warning("draw_avg_beds_heatmap: 공통 자치구 또는 유형이 없어 히트맵을 생성할 수 없습니다.")
        return None

    df_h_common = df_h_numeric.loc[common_gus, common_types_heatmap]
    df_b_common = df_b_numeric.loc[common_gus, common_types_heatmap]
    
    df_avg = df_b_common.divide(df_h_common.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    if df_avg.empty:
        st.info("평균 병상 수 계산 결과가 비어 히트맵을 그릴 수 없습니다.")
        return None

    fig, ax = plt.subplots(figsize=(10, 12 if len(df_avg.index) > 15 else 9), dpi=120) 
    
    cmap = plt.cm.get_cmap("Blues", 10) 
    cmap.set_bad(color='whitesmoke') 

    valid_data_for_scale = df_avg.values[~np.isnan(df_avg.values) & ~np.isinf(df_avg.values)]
    vmin_val = 0 
    vmax_val = np.max(valid_data_for_scale) if valid_data_for_scale.size > 0 else 1
    if vmin_val == vmax_val and valid_data_for_scale.size > 0: 
        vmax_val = vmin_val + 1 

    im = ax.imshow(
        df_avg.values, 
        cmap=cmap,
        aspect="auto", 
        vmin=vmin_val, 
        vmax=vmax_val 
    )

    ax.set_xticks(np.arange(len(df_avg.columns)))
    ax.set_xticklabels(df_avg.columns, fontsize=11, fontweight='bold') 
    ax.set_yticks(np.arange(len(df_avg.index)))
    ax.set_yticklabels(df_avg.index, fontsize=10) 
    ax.set_xlabel("기관 유형", fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel("자치구", fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title("서울시 자치구별 기관 유형별 평균 병상 수", fontsize=16, fontweight='bold', pad=15)

    for i in range(len(df_avg.index)):
        for j in range(len(df_avg.columns)):
            val = df_avg.iloc[i, j]
            text_to_display = "-" 
            if pd.notna(val) and not np.isinf(val):
                text_to_display = f"{val:.1f}" 
            
            cell_color_value = val
            text_color = "black" 
            if pd.notna(cell_color_value) and not np.isinf(cell_color_value) and vmax_val > vmin_val :
                normalized_val = (cell_color_value - vmin_val) / (vmax_val - vmin_val)
                if normalized_val > 0.55: 
                    text_color = "white"
            
            ax.text(j, i, text_to_display,
                    ha="center", va="center",
                    color=text_color, fontsize=9, fontweight='normal')

    ax.set_xticks(np.arange(len(df_avg.columns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(df_avg.index) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5) 
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.05, aspect=30) 
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("평균 병상 수", fontsize=11, fontweight='bold', labelpad=10)

    plt.tight_layout(pad=0.5)
    st.pyplot(fig)
    return df_avg


def draw_sheet0_charts(
    df_metrics,
    figsize1: tuple = (14, 6), 
    figsize2: tuple = (14, 6), 
    figsize3: tuple = (14, 7), 
    dpi: int = 100
) -> None:
    if df_metrics is None or df_metrics.empty:
        st.info("노인주거복지시설 데이터가 없어 차트를 그릴 수 없습니다.")
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
    ax1.set_xlabel("자치구", fontsize=12)
    ax1.set_ylabel('인원 수 (명)', fontsize=12)
    ax1.set_title('서울시 자치구별 노인주거복지시설 정원·현원·추가 수용 가능 인원', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val):,}'))
    fig1.tight_layout()
    st.pyplot(fig1)

    num_groups_fig2 = 2
    bar_width_fig2 = base_width / num_groups_fig2 * 0.7
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    ax2.bar(x - bar_width_fig2/2, df_metrics['facility'], width=bar_width_fig2, label='시설 수 (개소)', color='skyblue')
    ax2b = ax2.twinx()
    ax2b.bar(x + bar_width_fig2/2, df_metrics['staff'],    width=bar_width_fig2, label='종사자 수 (명)', color='lightcoral')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax2.set_xlabel("자치구", fontsize=12)
    ax2.set_title('서울시 자치구별 노인주거복지시설 시설수 및 종사자수', fontsize=15, fontweight='bold')
    ax2.set_ylabel('시설 수 (개소)', fontsize=12, color='black')
    ax2b.set_ylabel('종사자 수 (명)', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2b.tick_params(axis='y', labelcolor='black')
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=figsize3, dpi=dpi)
    ax3.bar(x - bar_width_fig2/2, df_metrics['cap_per_staff'], width=bar_width_fig2, label='종사자 1인당 정원', color='mediumseagreen')
    ax3.bar(x + bar_width_fig2/2, df_metrics['occ_per_staff'], width=bar_width_fig2, label='종사자 1인당 현원', color='mediumpurple')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax3.set_xlabel("자치구", fontsize=12)
    ax3.set_ylabel('담당 인원 수 (명/종사자 1인)', fontsize=12)
    ax3.set_title('서울시 자치구별 노인주거복지시설 종사자 1인당 담당 인원 비교', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(axis='y', linestyle=':', alpha=0.7)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.1f}'))
    fig3.tight_layout()
    st.pyplot(fig3)

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
    ax1.set_xlabel("자치구", fontsize=12)
    ax1.set_ylabel('인원 수 (명)', fontsize=12)
    ax1.set_title('서울시 자치구별 노인의료복지시설 정원·현원·추가 수용 가능 인원', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val):,}'))
    fig1.tight_layout()
    st.pyplot(fig1)


    num_groups_fig2 = 2
    bar_width_fig2 = base_width / num_groups_fig2 * 0.7
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    ax2.bar(x - bar_width_fig2/2, df_metrics['facility'], width=bar_width_fig2, label='시설 수 (개소)', color='skyblue')
    ax2b = ax2.twinx()
    ax2b.bar(x + bar_width_fig2/2, df_metrics['staff'],    width=bar_width_fig2, label='종사자 수 (명)', color='lightcoral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax2.set_xlabel("자치구", fontsize=12)
    ax2.set_title('서울시 자치구별 노인의료복지시설 시설수 및 종사자수', fontsize=15, fontweight='bold')
    ax2.set_ylabel('시설 수 (개소)', fontsize=12, color='black')
    ax2b.set_ylabel('종사자 수 (명)', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2b.tick_params(axis='y', labelcolor='black')
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=figsize3, dpi=dpi)
    ax3.bar(x - bar_width_fig2/2, df_metrics['cap_per_staff'], width=bar_width_fig2, label='종사자 1인당 정원', color='mediumseagreen')
    ax3.bar(x + bar_width_fig2/2, df_metrics['occ_per_staff'], width=bar_width_fig2, label='종사자 1인당 현원', color='mediumpurple')
    ax3.set_xticks(x)
    ax3.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax3.set_xlabel("자치구", fontsize=12)
    ax3.set_ylabel('담당 인원 수 (명/종사자 1인)', fontsize=12)
    ax3.set_title('서울시 자치구별 노인의료복지시설 종사자 1인당 담당 인원 비교', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(axis='y', linestyle=':', alpha=0.7)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.1f}'))
    fig3.tight_layout()
    st.pyplot(fig3)


def draw_nursing_csv_charts(
    df_welf: pd.DataFrame,
    df_centers: pd.DataFrame,
    figsize1: tuple = (14, 6),
    figsize2: tuple = (14, 5),
    dpi: int = 100
) -> None:
    if (df_welf is None or df_welf.empty) and \
       (df_centers is None or df_centers.empty):
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
        ax1.bar(x_welf - bar_width_welf/2, df_welf['facility'], width=bar_width_welf, label='시설 수 (개소)', color='deepskyblue')
        ax1b = ax1.twinx()
        ax1b.bar(x_welf + bar_width_welf/2, df_welf['staff'],    width=bar_width_welf, label='종사자 수 (명)', color='tomato')
        
        ax1.set_xticks(x_welf)
        ax1.set_xticklabels(regions_welf, rotation=45, ha='right', fontsize=10 if n_regions_welf <= 15 else 8)
        ax1.set_xlabel("자치구", fontsize=12)
        ax1.set_title('서울시 자치구별 노인복지관 시설수 및 종사자수', fontsize=15, fontweight='bold')
        ax1.set_ylabel('시설 수 (개소)', fontsize=12, color='black')
        ax1b.set_ylabel('종사자 수 (명)', fontsize=12, color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1b.tick_params(axis='y', labelcolor='black')
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
        bar_width_centers = 0.6

        fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
        ax2.bar(x_centers, df_centers['facility'], width=bar_width_centers, color='mediumseagreen', label='시설 수 (개소)')
        ax2.set_xticks(x_centers)
        ax2.set_xticklabels(regions_centers, rotation=45, ha='right', fontsize=10 if n_regions_centers <= 15 else 8)
        ax2.set_xlabel("자치구", fontsize=12)
        ax2.set_ylabel('시설 수 (개소)', fontsize=12)
        ax2.set_title('서울시 자치구별 경로당 및 노인교실 총 시설수', fontsize=15, fontweight='bold')
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
    
    num_groups_fig1 = 2
    bar_width_fig1 = base_width / num_groups_fig1 * 0.8

    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    ax1.bar(x - bar_width_fig1/2, df_metrics['capacity'],   width=bar_width_fig1, label='정원', color='cornflowerblue')
    ax1.bar(x + bar_width_fig1/2, df_metrics['occupancy'], width=bar_width_fig1, label='현원', color='salmon')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax1.set_xlabel("자치구", fontsize=12)
    ax1.set_ylabel('인원 수 (명)', fontsize=12)
    ax1.set_title('서울시 자치구별 재가노인복지시설 정원·현원 인원수', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val):,}'))
    fig1.tight_layout()
    st.pyplot(fig1)

    num_groups_fig2 = 2
    bar_width_fig2 = base_width / num_groups_fig2 * 0.7
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    ax2.bar(x - bar_width_fig2/2, df_metrics['facility'], width=bar_width_fig2, label='시설 수 (개소)', color='skyblue')
    ax2b = ax2.twinx()
    ax2b.bar(x + bar_width_fig2/2, df_metrics['staff'],    width=bar_width_fig2, label='종사자 수 (명)', color='lightcoral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax2.set_xlabel("자치구", fontsize=12)
    ax2.set_title('서울시 자치구별 재가노인복지시설 시설수 및 종사자수', fontsize=15, fontweight='bold')
    ax2.set_ylabel('시설 수 (개소)', fontsize=12, color='black')
    ax2b.set_ylabel('종사자 수 (명)', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2b.tick_params(axis='y', labelcolor='black')
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=figsize3, dpi=dpi)
    ax3.bar(x - bar_width_fig2/2, df_metrics['cap_per_staff'], width=bar_width_fig2, label='종사자 1인당 정원', color='mediumseagreen')
    ax3.bar(x + bar_width_fig2/2, df_metrics['occ_per_staff'], width=bar_width_fig2, label='종사자 1인당 현원', color='mediumpurple')
    ax3.set_xticks(x)
    ax3.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax3.set_xlabel("자치구", fontsize=12)
    ax3.set_ylabel('담당 인원 수 (명/종사자 1인)', fontsize=12)
    ax3.set_title('서울시 자치구별 재가노인복지시설 종사자 1인당 담당 인원 비교', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(axis='y', linestyle=':', alpha=0.7)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.1f}'))
    fig3.tight_layout()
    st.pyplot(fig3)


def draw_sheet4_charts(
    df_metrics,
    figsize1: tuple = (14, 6), 
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
    ax1.bar(x - bar_width/2, df_metrics['facility'], width=bar_width, label='시설 수 (개소)', color='skyblue')
    ax1b = ax1.twinx() 
    ax1b.bar(x + bar_width/2, df_metrics['staff'],    width=bar_width, label='종사자 수 (명)', color='lightcoral')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax1.set_xlabel("자치구", fontsize=12)
    ax1.set_title('서울시 자치구별 노인일자리지원기관 시설수 및 종사자수', fontsize=15, fontweight='bold')
    ax1.set_ylabel('시설 수 (개소)', fontsize=12, color='black')
    ax1b.set_ylabel('종사자 수 (명)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1b.tick_params(axis='y', labelcolor='black')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1b.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)
    ax1.grid(axis='y', linestyle=':', alpha=0.3) 
    
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
    ax1.set_xlabel("자치구", fontsize=12)
    ax1.set_ylabel('인원 수 (명)', fontsize=12)
    ax1.set_title('서울시 자치구별 치매전담형 장기요양기관 정원·현원·추가 수용 가능 인원', 
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val):,}'))
    fig1.tight_layout()
    st.pyplot(fig1)

    num_groups_fig2 = 2
    bar_width_fig2 = base_width / num_groups_fig2 * 0.7
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    ax2.bar(x - bar_width_fig2/2, df_metrics['facility'], width=bar_width_fig2, label='시설 수 (개소)', color='skyblue')
    ax2b = ax2.twinx()
    ax2b.bar(x + bar_width_fig2/2, df_metrics['staff'],    width=bar_width_fig2, label='종사자 수 (명)', color='lightcoral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax2.set_xlabel("자치구", fontsize=12)
    ax2.set_title('서울시 자치구별 치매전담형 장기요양기관 시설수 및 종사자수', 
                  fontsize=15, fontweight='bold')
    ax2.set_ylabel('시설 수 (개소)', fontsize=12, color='black')
    ax2b.set_ylabel('종사자 수 (명)', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2b.tick_params(axis='y', labelcolor='black')
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=figsize3, dpi=dpi)
    ax3.bar(x - bar_width_fig2/2, df_metrics['cap_per_staff'], width=bar_width_fig2, label='종사자 1인당 정원', color='mediumseagreen')
    ax3.bar(x + bar_width_fig2/2, df_metrics['occ_per_staff'], width=bar_width_fig2, label='종사자 1인당 현원', color='mediumpurple')
    ax3.set_xticks(x)
    ax3.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax3.set_xlabel("자치구", fontsize=12)
    ax3.set_ylabel('담당 인원 수 (명/종사자 1인)', fontsize=12)
    ax3.set_title('서울시 자치구별 치매전담형 장기요양기관 종사자 1인당 담당 인원 비교', 
                  fontsize=15, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(axis='y', linestyle=':', alpha=0.7)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.1f}'))
    fig3.tight_layout()
    st.pyplot(fig3)
# --- END OF chart_utils.py ---
