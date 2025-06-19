# --- START OF MODIFIED FILE chart_utils.py ---
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt # Altair 예시 함수를 위해 유지
import pandas as pd
import numpy as np
import seaborn as sns 
from matplotlib.ticker import FuncFormatter

# ——————————————————————————————————————————————————
# 한글 폰트 설정 (Matplotlib용)
# 각 페이지 파일에서 utils.set_korean_font()를 호출하여 전역 설정을 하므로,
# chart_utils.py에서는 plt.rcParams['axes.unicode_minus'] = False 만 유지합니다.
# ——————————————————————————————————————————————————
plt.rcParams['axes.unicode_minus'] = False
# ——————————————————————————————————————————————————


def draw_example_bar_chart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(df["영역"], df["값"], label="값")
    ax.set_xlabel("값(단위)")
    ax.set_ylabel("영역")
    ax.set_title("예시 수평 막대그래프")
    ax.legend(fontsize=10, loc='upper right')
    st.pyplot(fig)


def draw_example_altair_scatter(df: pd.DataFrame):
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
    if df_hosp is None or df_hosp.empty:
        st.info("의료기관 수 데이터가 없어 막대 그래프를 그릴 수 없습니다.")
        return

    df_plot_base = df_hosp.copy()
    if "gu" in df_plot_base.columns:
        df_plot_base = df_plot_base[df_plot_base["gu"] != "소계"].reset_index(drop=True)
    else:
        st.error("draw_hospital_count_bar_charts: df_hosp에 'gu' 컬럼이 없습니다.")
        return

    types = ["소계", "종합병원", "병원", "의원", "요양병원"]
    missing_types = [t for t in types if t not in df_plot_base.columns]
    if missing_types:
        st.warning(f"draw_hospital_count_bar_charts: df_plot_base에 다음 컬럼이 없습니다: {missing_types}. 해당 그래프는 생략될 수 있습니다.")

    for inst in types:
        if inst not in df_plot_base.columns:
            continue

        df_plot_inst = df_plot_base[['gu', inst]].copy()
        df_plot_inst[inst] = pd.to_numeric(df_plot_inst[inst], errors="coerce").fillna(0)
        df_plot_inst_sorted = df_plot_inst.sort_values(by=inst, ascending=False).reset_index(drop=True)

        if df_plot_inst_sorted.empty:
            st.info(f"'{inst}' 유형에 대한 데이터가 없어 그래프를 생략합니다.")
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(df_plot_inst_sorted["gu"], df_plot_inst_sorted[inst], color='skyblue', label=inst, zorder=3)
        ax.set_title(f"서울시 자치구별 {inst} 수", fontsize=15)
        ax.set_xlabel("자치구", fontsize=12)
        ax.set_ylabel("기관 수", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

        if not df_plot_inst_sorted[inst].empty and df_plot_inst_sorted[inst].max() > 0 :
             ax.set_ylim(0, df_plot_inst_sorted[inst].max() * 1.15)

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

    total_hosp_dict = {}
    total_beds_dict = {}
    for t in types:
        total_hosp_dict[t] = pd.to_numeric(df_h.get(t, 0), errors="coerce").sum()
        total_beds_dict[t] = pd.to_numeric(df_b.get(t, 0), errors="coerce").sum()

    avg_beds_dict = {}
    for t in types:
        hosp_count = total_hosp_dict.get(t, 0)
        if hosp_count > 0:
            avg_beds_dict[t] = total_beds_dict.get(t, 0) / hosp_count
        else:
            avg_beds_dict[t] = 0.0

    bar_colors = {'병원 수': 'mediumseagreen', '병상 수': 'cornflowerblue', '평균 병상 수': 'lightcoral'}

    sorted_total_hosp = sorted(total_hosp_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_total_hosp_keys = [item[0] for item in sorted_total_hosp]
    sorted_total_hosp_values = [item[1] for item in sorted_total_hosp]

    sorted_total_beds = sorted(total_beds_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_total_beds_keys = [item[0] for item in sorted_total_beds]
    sorted_total_beds_values = [item[1] for item in sorted_total_beds]

    sorted_avg_beds = sorted(avg_beds_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_avg_beds_keys = [item[0] for item in sorted_avg_beds]
    sorted_avg_beds_values = [item[1] for item in sorted_avg_beds]

    fig_hosp, ax_hosp = plt.subplots(figsize=(8, 4.5))
    ax_hosp.bar(sorted_total_hosp_keys, sorted_total_hosp_values, color=bar_colors['병원 수'], label='병원 수')
    ax_hosp.set_title('의료기관 유형별 전체 병원 수', fontsize=15)
    ax_hosp.set_ylabel('병원 수', fontsize=12)
    ax_hosp.set_xlabel('기관 유형', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10); plt.yticks(fontsize=10)
    ax_hosp.grid(axis='y', linestyle=':', alpha=0.6)
    ax_hosp.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    st.pyplot(fig_hosp)

    fig_beds, ax_beds = plt.subplots(figsize=(8, 4.5))
    ax_beds.bar(sorted_total_beds_keys, sorted_total_beds_values, color=bar_colors['병상 수'], label='병상 수')
    ax_beds.set_title('의료기관 유형별 전체 병상 수', fontsize=15)
    ax_beds.set_ylabel('병상 수', fontsize=12)
    ax_beds.set_xlabel('기관 유형', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10); plt.yticks(fontsize=10)
    ax_beds.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax_beds.grid(axis='y', linestyle=':', alpha=0.6)
    ax_beds.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    st.pyplot(fig_beds)

    fig_avg_beds, ax_avg_beds = plt.subplots(figsize=(8, 4.5))
    ax_avg_beds.bar(sorted_avg_beds_keys, sorted_avg_beds_values, color=bar_colors['평균 병상 수'], label='평균 병상 수')
    ax_avg_beds.set_title('의료기관 유형별 병원당 평균 병상 수', fontsize=15)
    ax_avg_beds.set_ylabel('평균 병상 수', fontsize=12)
    ax_avg_beds.set_xlabel('기관 유형', fontsize=12)
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

def plot_grouped_bar_all_conditions_yearly(all_years_summary_df):
    if all_years_summary_df is None or all_years_summary_df.empty:
        st.info("연도별/질환별 환자수 비교를 위한 데이터가 없습니다.")
        return

    df_to_plot = all_years_summary_df.sort_values(by=['연도', '질환명'])

    plt.figure(figsize=(18, 9))

    # 방법 1: palette 인자 완전 제거 (Seaborn 기본 동작에 맡김)
    sns.barplot(
        data=df_to_plot,
        x='질환명',
        y='총 노인 환자수',
        hue='연도'
        # palette 인자 없음
    )

    # 방법 2: (만약 방법 1이 여전히 보라색이면 시도) Seaborn의 기본 'deep' 팔레트 명시적 사용
    # num_hues = df_to_plot['연도'].nunique() # '연도'의 고유값 개수만큼 색상 필요
    # current_palette = sns.color_palette("deep", n_colors=num_hues) # 기본 'deep' 팔레트 사용
    # sns.barplot(
    #     data=df_to_plot,
    #     x='질환명',
    #     y='총 노인 환자수',
    #     hue='연도',
    #     palette=current_palette # 명시적으로 기본 팔레트 지정
    # )

    plt.title('서울시 연도별/질환별 노인 환자수 비교', fontsize=16, pad=15)
    plt.xlabel('질환명', fontsize=12)
    plt.ylabel('총 노인 환자수 (명)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='연도', bbox_to_anchor=(1.02, 1), loc='upper left', title_fontsize='11', fontsize='10')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.tight_layout(rect=[0,0,0.9,1])
    st.pyplot(plt)
    
# --- Welfare Facilities Charts ---
def draw_sheet0_charts(
    df_metrics_input,
    year: int,
    figsize1: tuple = (14, 6),
    figsize2: tuple = (14, 6),
    figsize3: tuple = (14, 7),
    dpi: int = 100
) -> None:
    if df_metrics_input is None or df_metrics_input.empty:
        st.info(f"노인주거복지시설 데이터가 없어 차트를 그릴 수 없습니다.")
        return

    df_metrics = df_metrics_input.copy()

    df_metrics_sorted_fig1 = df_metrics.sort_values(by='capacity', ascending=False)
    regions_fig1 = df_metrics_sorted_fig1.index.tolist()
    x_fig1 = np.arange(len(regions_fig1))
    n_regions_fig1 = len(regions_fig1)

    df_metrics_sorted_fig2 = df_metrics.sort_values(by=['facility', 'staff'], ascending=[False, False])
    regions_fig2 = df_metrics_sorted_fig2.index.tolist()
    x_fig2 = np.arange(len(regions_fig2))
    n_regions_fig2 = len(regions_fig2)

    df_metrics_sorted_fig3 = df_metrics.sort_values(by='cap_per_staff', ascending=False)
    regions_fig3 = df_metrics_sorted_fig3.index.tolist()
    x_fig3 = np.arange(len(regions_fig3))
    n_regions_fig3 = len(regions_fig3)

    base_width = 0.8
    num_groups_fig1 = 3
    bar_width_fig1 = base_width / num_groups_fig1 * 0.8

    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    ax1.bar(x_fig1 - bar_width_fig1, df_metrics_sorted_fig1['capacity'],   width=bar_width_fig1, label='정원', color='cornflowerblue')
    ax1.bar(x_fig1,                  df_metrics_sorted_fig1['occupancy'], width=bar_width_fig1, label='현원', color='salmon')
    ax1.bar(x_fig1 + bar_width_fig1, df_metrics_sorted_fig1['additional'], width=bar_width_fig1, label='추가 수용', color='lightgreen')
    ax1.set_xticks(x_fig1)
    ax1.set_xticklabels(regions_fig1, rotation=45, ha='right', fontsize=10 if n_regions_fig1 <= 15 else 8)
    ax1.set_xlabel("자치구", fontsize=12)
    ax1.set_ylabel('인원 수 (명)', fontsize=12)
    ax1.set_title('서울시 자치구별 노인주거복지시설 정원 및 현원, 추가 수용 인원', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val):,}'))
    fig1.tight_layout()
    st.pyplot(fig1)

    num_groups_fig2 = 2
    bar_width_fig2 = base_width / num_groups_fig2 * 0.7
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    ax2.bar(x_fig2 - bar_width_fig2/2, df_metrics_sorted_fig2['facility'], width=bar_width_fig2, label='시설 수 (개소)', color='skyblue')
    ax2b = ax2.twinx()
    ax2b.bar(x_fig2 + bar_width_fig2/2, df_metrics_sorted_fig2['staff'],    width=bar_width_fig2, label='종사자 수 (명)', color='lightcoral')
    ax2.set_xticks(x_fig2)
    ax2.set_xticklabels(regions_fig2, rotation=45, ha='right', fontsize=10 if n_regions_fig2 <= 15 else 8)
    ax2.set_xlabel("자치구", fontsize=12)
    ax2.set_title('서울시 자치구별 노인주거복지시설 시설수 및 종사자수', fontsize=15, fontweight='bold')
    ax2.set_ylabel('시설 수 (개소)', fontsize=12, color='black')
    ax2b.set_ylabel('종사자 수 (명)', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2b.tick_params(axis='y', labelcolor='black')
    lines, labels_ax = ax2.get_legend_handles_labels()
    lines2, labels2_ax = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels_ax + labels2_ax, loc='upper right', fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=figsize3, dpi=dpi)
    ax3.bar(x_fig3 - bar_width_fig2/2, df_metrics_sorted_fig3['cap_per_staff'], width=bar_width_fig2, label='종사자 1인당 정원 돌봄 수', color='mediumseagreen')
    ax3.bar(x_fig3 + bar_width_fig2/2, df_metrics_sorted_fig3['occ_per_staff'], width=bar_width_fig2, label='종사자 1인당 현원 돌봄 수', color='mediumpurple')
    ax3.set_xticks(x_fig3)
    ax3.set_xticklabels(regions_fig3, rotation=45, ha='right', fontsize=10 if n_regions_fig3 <= 15 else 8)
    ax3.set_xlabel("자치구", fontsize=12)
    ax3.set_ylabel('담당 인원 수 (명/종사자 1인)', fontsize=12)
    ax3.set_title('서울시 자치구별 노인주거복지시설 종사자 1인당 담당 인원 비교', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(axis='y', linestyle=':', alpha=0.7)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.1f}'))
    fig3.tight_layout()
    st.pyplot(fig3)

def draw_sheet1_charts(
    df_metrics_input,
    year: int,
    figsize1: tuple = (14, 6),
    figsize2: tuple = (14, 6),
    figsize3: tuple = (14, 7),
    dpi: int = 100
) -> None:
    if df_metrics_input is None or df_metrics_input.empty:
        st.info(f"노인의료복지시설 데이터가 없어 차트를 그릴 수 없습니다.")
        return

    df_metrics = df_metrics_input.copy()

    df_metrics_sorted_fig1 = df_metrics.sort_values(by='capacity', ascending=False)
    regions_fig1 = df_metrics_sorted_fig1.index.tolist()
    x_fig1 = np.arange(len(regions_fig1))
    n_regions_fig1 = len(regions_fig1)

    df_metrics_sorted_fig2 = df_metrics.sort_values(by=['facility', 'staff'], ascending=[False, False])
    regions_fig2 = df_metrics_sorted_fig2.index.tolist()
    x_fig2 = np.arange(len(regions_fig2))
    n_regions_fig2 = len(regions_fig2)

    df_metrics_sorted_fig3 = df_metrics.sort_values(by='cap_per_staff', ascending=False)
    regions_fig3 = df_metrics_sorted_fig3.index.tolist()
    x_fig3 = np.arange(len(regions_fig3))
    n_regions_fig3 = len(regions_fig3)

    base_width = 0.8
    num_groups_fig1 = 3
    bar_width_fig1 = base_width / num_groups_fig1 * 0.8

    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    ax1.bar(x_fig1 - bar_width_fig1, df_metrics_sorted_fig1['capacity'],   width=bar_width_fig1, label='정원', color='cornflowerblue')
    ax1.bar(x_fig1,                  df_metrics_sorted_fig1['occupancy'], width=bar_width_fig1, label='현원', color='salmon')
    ax1.bar(x_fig1 + bar_width_fig1, df_metrics_sorted_fig1['additional'], width=bar_width_fig1, label='추가 수용', color='lightgreen')
    ax1.set_xticks(x_fig1)
    ax1.set_xticklabels(regions_fig1, rotation=45, ha='right', fontsize=10 if n_regions_fig1 <= 15 else 8)
    ax1.set_xlabel("자치구", fontsize=12)
    ax1.set_ylabel('인원 수 (명)', fontsize=12)
    ax1.set_title('서울시 자치구별 노인의료복지시설 정원 및 현원, 추가 수용 인원', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val):,}'))
    fig1.tight_layout()
    st.pyplot(fig1)

    num_groups_fig2 = 2
    bar_width_fig2 = base_width / num_groups_fig2 * 0.7
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    ax2.bar(x_fig2 - bar_width_fig2/2, df_metrics_sorted_fig2['facility'], width=bar_width_fig2, label='시설 수 (개소)', color='skyblue')
    ax2b = ax2.twinx()
    ax2b.bar(x_fig2 + bar_width_fig2/2, df_metrics_sorted_fig2['staff'],    width=bar_width_fig2, label='종사자 수 (명)', color='lightcoral')
    ax2.set_xticks(x_fig2)
    ax2.set_xticklabels(regions_fig2, rotation=45, ha='right', fontsize=10 if n_regions_fig2 <= 15 else 8)
    ax2.set_xlabel("자치구", fontsize=12)
    ax2.set_title('서울시 자치구별 노인의료복지시설 시설수 및 종사자수', fontsize=15, fontweight='bold')
    ax2.set_ylabel('시설 수 (개소)', fontsize=12, color='black')
    ax2b.set_ylabel('종사자 수 (명)', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2b.tick_params(axis='y', labelcolor='black')
    lines, labels_ax = ax2.get_legend_handles_labels()
    lines2, labels2_ax = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels_ax + labels2_ax, loc='upper right', fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=figsize3, dpi=dpi)
    ax3.bar(x_fig3 - bar_width_fig2/2, df_metrics_sorted_fig3['cap_per_staff'], width=bar_width_fig2, label='종사자 1인당 정원 돌봄 수', color='mediumseagreen')
    ax3.bar(x_fig3 + bar_width_fig2/2, df_metrics_sorted_fig3['occ_per_staff'], width=bar_width_fig2, label='종사자 1인당 현원 돌봄 수', color='mediumpurple')
    ax3.set_xticks(x_fig3)
    ax3.set_xticklabels(regions_fig3, rotation=45, ha='right', fontsize=10 if n_regions_fig3 <= 15 else 8)
    ax3.set_xlabel("자치구", fontsize=12)
    ax3.set_ylabel('담당 인원 수 (명/종사자 1인)', fontsize=12)
    ax3.set_title('서울시 자치구별 노인의료복지시설 종사자 1인당 담당 인원 비교', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(axis='y', linestyle=':', alpha=0.7)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.1f}'))
    fig3.tight_layout()
    st.pyplot(fig3)

def draw_nursing_csv_charts(
    df_welf_input: pd.DataFrame,
    df_centers_input: pd.DataFrame,
    year: int,
    figsize1: tuple = (14, 6),
    figsize2: tuple = (14, 5),
    dpi: int = 100
) -> None:
    if (df_welf_input is None or df_welf_input.empty) and \
       (df_centers_input is None or df_centers_input.empty):
        st.info(f"노인여가복지시설(CSV) 데이터가 없어 차트를 그릴 수 없습니다.")
        return

    if df_welf_input is not None and not df_welf_input.empty:
        df_welf = df_welf_input.copy()
        df_welf_sorted = df_welf.sort_values(by=['facility', 'staff'], ascending=[False, False])
        regions_welf = df_welf_sorted.index.tolist()
        x_welf = np.arange(len(regions_welf))
        n_regions_welf = len(regions_welf)
        base_width_welf = 0.8
        num_groups_welf = 2
        bar_width_welf = base_width_welf / num_groups_welf * 0.7

        fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
        ax1.bar(x_welf - bar_width_welf/2, df_welf_sorted['facility'], width=bar_width_welf, label='시설 수 (개소)', color='deepskyblue')
        ax1b = ax1.twinx()
        ax1b.bar(x_welf + bar_width_welf/2, df_welf_sorted['staff'],    width=bar_width_welf, label='종사자 수 (명)', color='tomato')
        ax1.set_xticks(x_welf)
        ax1.set_xticklabels(regions_welf, rotation=45, ha='right', fontsize=10 if n_regions_welf <= 15 else 8)
        ax1.set_xlabel("자치구", fontsize=12)
        ax1.set_title('서울시 자치구별 노인복지관 시설수 및 종사자수', fontsize=15, fontweight='bold')
        ax1.set_ylabel('시설 수 (개소)', fontsize=12, color='black')
        ax1b.set_ylabel('종사자 수 (명)', fontsize=12, color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1b.tick_params(axis='y', labelcolor='black')
        lines, labels_ax = ax1.get_legend_handles_labels()
        lines2, labels2_ax = ax1b.get_legend_handles_labels()
        ax1b.legend(lines + lines2, labels_ax + labels2_ax, loc='upper right', fontsize=10)
        ax1.grid(axis='y', linestyle=':', alpha=0.3)
        fig1.tight_layout()
        st.pyplot(fig1)
    else:
        st.info(f"노인복지관(CSV) 데이터가 없어 '시설수 vs 종사자수' 차트를 그릴 수 없습니다.")

    if df_centers_input is not None and not df_centers_input.empty:
        df_centers = df_centers_input.copy()
        df_centers_sorted = df_centers.sort_values(by='facility', ascending=False)
        regions_centers = df_centers_sorted.index.tolist()
        x_centers = np.arange(len(regions_centers))
        n_regions_centers = len(regions_centers)
        bar_width_centers = 0.6

        fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
        ax2.bar(x_centers, df_centers_sorted['facility'], width=bar_width_centers, color='mediumseagreen', label='시설 수 (개소)')
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
        st.info(f"경로당+노인교실(CSV) 데이터가 없어 '총 시설수' 차트를 그릴 수 없습니다.")

def draw_sheet3_charts(
    df_metrics_input,
    year: int,
    figsize1: tuple = (14, 6),
    figsize2: tuple = (14, 6),
    figsize3: tuple = (14, 7),
    dpi: int = 100
) -> None:
    if df_metrics_input is None or df_metrics_input.empty:
        st.info(f"재가노인복지시설 데이터가 없어 차트를 그릴 수 없습니다.")
        return

    df_metrics = df_metrics_input.copy()

    df_metrics_sorted_fig1 = df_metrics.sort_values(by='capacity', ascending=False)
    regions_fig1 = df_metrics_sorted_fig1.index.tolist()
    x_fig1 = np.arange(len(regions_fig1))
    n_regions_fig1 = len(regions_fig1)

    df_metrics_sorted_fig2 = df_metrics.sort_values(by=['facility', 'staff'], ascending=[False, False])
    regions_fig2 = df_metrics_sorted_fig2.index.tolist()
    x_fig2 = np.arange(len(regions_fig2))
    n_regions_fig2 = len(regions_fig2)

    df_metrics_sorted_fig3 = df_metrics.sort_values(by='cap_per_staff', ascending=False)
    regions_fig3 = df_metrics_sorted_fig3.index.tolist()
    x_fig3 = np.arange(len(regions_fig3))
    n_regions_fig3 = len(regions_fig3)

    base_width = 0.8
    num_groups_fig1 = 2
    bar_width_fig1 = base_width / num_groups_fig1 * 0.8

    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    ax1.bar(x_fig1 - bar_width_fig1/2, df_metrics_sorted_fig1['capacity'],   width=bar_width_fig1, label='정원', color='cornflowerblue')
    ax1.bar(x_fig1 + bar_width_fig1/2, df_metrics_sorted_fig1['occupancy'], width=bar_width_fig1, label='현원', color='salmon')
    ax1.set_xticks(x_fig1)
    ax1.set_xticklabels(regions_fig1, rotation=45, ha='right', fontsize=10 if n_regions_fig1 <= 15 else 8)
    ax1.set_xlabel("자치구", fontsize=12)
    ax1.set_ylabel('인원 수 (명)', fontsize=12)
    ax1.set_title('서울시 자치구별 재가노인복지시설 정원 및 현원 인원수', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val):,}'))
    fig1.tight_layout()
    st.pyplot(fig1)

    num_groups_fig2 = 2
    bar_width_fig2 = base_width / num_groups_fig2 * 0.7
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    ax2.bar(x_fig2 - bar_width_fig2/2, df_metrics_sorted_fig2['facility'], width=bar_width_fig2, label='시설 수 (개소)', color='skyblue')
    ax2b = ax2.twinx()
    ax2b.bar(x_fig2 + bar_width_fig2/2, df_metrics_sorted_fig2['staff'],    width=bar_width_fig2, label='종사자 수 (명)', color='lightcoral')
    ax2.set_xticks(x_fig2)
    ax2.set_xticklabels(regions_fig2, rotation=45, ha='right', fontsize=10 if n_regions_fig2 <= 15 else 8)
    ax2.set_xlabel("자치구", fontsize=12)
    ax2.set_title('서울시 자치구별 재가노인복지시설 시설수 및 종사자수', fontsize=15, fontweight='bold')
    ax2.set_ylabel('시설 수 (개소)', fontsize=12, color='black')
    ax2b.set_ylabel('종사자 수 (명)', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2b.tick_params(axis='y', labelcolor='black')
    lines, labels_ax = ax2.get_legend_handles_labels()
    lines2, labels2_ax = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels_ax + labels2_ax, loc='upper right', fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=figsize3, dpi=dpi)
    ax3.bar(x_fig3 - bar_width_fig2/2, df_metrics_sorted_fig3['cap_per_staff'], width=bar_width_fig2, label='종사자 1인당 정원 돌봄 수', color='mediumseagreen')
    ax3.bar(x_fig3 + bar_width_fig2/2, df_metrics_sorted_fig3['occ_per_staff'], width=bar_width_fig2, label='종사자 1인당 현원 돌봄 수', color='mediumpurple')
    ax3.set_xticks(x_fig3)
    ax3.set_xticklabels(regions_fig3, rotation=45, ha='right', fontsize=10 if n_regions_fig3 <= 15 else 8)
    ax3.set_xlabel("자치구", fontsize=12)
    ax3.set_ylabel('담당 인원 수 (명/종사자 1인)', fontsize=12)
    ax3.set_title('서울시 자치구별 재가노인복지시설 종사자 1인당 담당 인원 비교', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(axis='y', linestyle=':', alpha=0.7)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.1f}'))
    fig3.tight_layout()
    st.pyplot(fig3)

def draw_sheet4_charts(
    df_metrics_input,
    year: int,
    figsize1: tuple = (14, 6),
    dpi: int = 100
) -> None:
    if df_metrics_input is None or df_metrics_input.empty:
        st.info(f"노인일자리지원기관 데이터가 없어 차트를 그릴 수 없습니다.")
        return

    df_metrics = df_metrics_input.copy()
    df_metrics_sorted = df_metrics.sort_values(by=['facility', 'staff'], ascending=[False, False])
    regions = df_metrics_sorted.index.tolist()
    x = np.arange(len(regions))
    n_regions = len(regions)

    base_width = 0.8
    num_groups = 2
    bar_width = base_width / num_groups * 0.7

    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    ax1.bar(x - bar_width/2, df_metrics_sorted['facility'], width=bar_width, label='시설 수 (개소)', color='skyblue')
    ax1b = ax1.twinx()
    ax1b.bar(x + bar_width/2, df_metrics_sorted['staff'],    width=bar_width, label='종사자 수 (명)', color='lightcoral')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45, ha='right', fontsize=10 if n_regions <= 15 else 8)
    ax1.set_xlabel("자치구", fontsize=12)
    ax1.set_title('서울시 자치구별 노인일자리지원기관 시설수 및 종사자수', fontsize=15, fontweight='bold')
    ax1.set_ylabel('시설 수 (개소)', fontsize=12, color='black')
    ax1b.set_ylabel('종사자 수 (명)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1b.tick_params(axis='y', labelcolor='black')
    lines, labels_ax = ax1.get_legend_handles_labels()
    lines2, labels2_ax = ax1b.get_legend_handles_labels()
    ax1b.legend(lines + lines2, labels_ax + labels2_ax, loc='upper right', fontsize=10)
    ax1.grid(axis='y', linestyle=':', alpha=0.3)
    fig1.tight_layout()
    st.pyplot(fig1)

def draw_sheet5_charts(
    df_metrics_input,
    year: int,
    figsize1: tuple = (14, 6),
    figsize2: tuple = (14, 6),
    figsize3: tuple = (14, 7),
    dpi: int = 100
) -> None:
    if df_metrics_input is None or df_metrics_input.empty:
        st.info(f"치매전담형 장기요양기관 데이터가 없어 차트를 그릴 수 없습니다.")
        return

    df_metrics = df_metrics_input.copy()

    df_metrics_sorted_fig1 = df_metrics.sort_values(by='capacity', ascending=False)
    regions_fig1 = df_metrics_sorted_fig1.index.tolist()
    x_fig1 = np.arange(len(regions_fig1))
    n_regions_fig1 = len(regions_fig1)

    df_metrics_sorted_fig2 = df_metrics.sort_values(by=['facility', 'staff'], ascending=[False, False])
    regions_fig2 = df_metrics_sorted_fig2.index.tolist()
    x_fig2 = np.arange(len(regions_fig2))
    n_regions_fig2 = len(regions_fig2)

    df_metrics_sorted_fig3 = df_metrics.sort_values(by='cap_per_staff', ascending=False)
    regions_fig3 = df_metrics_sorted_fig3.index.tolist()
    x_fig3 = np.arange(len(regions_fig3))
    n_regions_fig3 = len(regions_fig3)

    base_width = 0.8
    num_groups_fig1 = 3
    bar_width_fig1 = base_width / num_groups_fig1 * 0.8

    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    ax1.bar(x_fig1 - bar_width_fig1, df_metrics_sorted_fig1['capacity'],   width=bar_width_fig1, label='정원', color='cornflowerblue')
    ax1.bar(x_fig1,                  df_metrics_sorted_fig1['occupancy'], width=bar_width_fig1, label='현원', color='salmon')
    ax1.bar(x_fig1 + bar_width_fig1, df_metrics_sorted_fig1['additional'], width=bar_width_fig1, label='추가 수용', color='lightgreen')
    ax1.set_xticks(x_fig1)
    ax1.set_xticklabels(regions_fig1, rotation=45, ha='right', fontsize=10 if n_regions_fig1 <= 15 else 8)
    ax1.set_xlabel("자치구", fontsize=12)
    ax1.set_ylabel('인원 수 (명)', fontsize=12)
    ax1.set_title('서울시 자치구별 치매전담형 장기요양기관 정원 및 현원, 추가 수용 인원',
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val):,}'))
    fig1.tight_layout()
    st.pyplot(fig1)

    num_groups_fig2 = 2
    bar_width_fig2 = base_width / num_groups_fig2 * 0.7
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    ax2.bar(x_fig2 - bar_width_fig2/2, df_metrics_sorted_fig2['facility'], width=bar_width_fig2, label='시설 수 (개소)', color='skyblue')
    ax2b = ax2.twinx()
    ax2b.bar(x_fig2 + bar_width_fig2/2, df_metrics_sorted_fig2['staff'],    width=bar_width_fig2, label='종사자 수 (명)', color='lightcoral')
    ax2.set_xticks(x_fig2)
    ax2.set_xticklabels(regions_fig2, rotation=45, ha='right', fontsize=10 if n_regions_fig2 <= 15 else 8)
    ax2.set_xlabel("자치구", fontsize=12)
    ax2.set_title('서울시 자치구별 치매전담형 장기요양기관 시설수 및 종사자수',
                  fontsize=15, fontweight='bold')
    ax2.set_ylabel('시설 수 (개소)', fontsize=12, color='black')
    ax2b.set_ylabel('종사자 수 (명)', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2b.tick_params(axis='y', labelcolor='black')
    lines, labels_ax = ax2.get_legend_handles_labels()
    lines2, labels2_ax = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels_ax + labels2_ax, loc='upper right', fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=figsize3, dpi=dpi)
    ax3.bar(x_fig3 - bar_width_fig2/2, df_metrics_sorted_fig3['cap_per_staff'], width=bar_width_fig2, label='종사자 1인당 정원 돌봄 수', color='mediumseagreen')
    ax3.bar(x_fig3 + bar_width_fig2/2, df_metrics_sorted_fig3['occ_per_staff'], width=bar_width_fig2, label='종사자 1인당 현원 돌봄 수', color='mediumpurple')
    ax3.set_xticks(x_fig3)
    ax3.set_xticklabels(regions_fig3, rotation=45, ha='right', fontsize=10 if n_regions_fig3 <= 15 else 8)
    ax3.set_xlabel("자치구", fontsize=12)
    ax3.set_ylabel('담당 인원 수 (명/종사자 1인)', fontsize=12)
    ax3.set_title('서울시 자치구별 치매전담형 장기요양기관 종사자 1인당 담당 인원 비교',
                  fontsize=15, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(axis='y', linestyle=':', alpha=0.7)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.1f}'))
    fig3.tight_layout()
    st.pyplot(fig3)

# --- Mental Health Charts ---
def plot_total_elderly_trend(total_patients_df, condition_name):
    if total_patients_df is None or total_patients_df.empty:
        st.info(f"노인 {condition_name} 환자수 총계 추이 데이터를 그릴 수 없습니다.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=total_patients_df, x='연도', y='총 노인 환자수', marker='o', ax=ax, color='steelblue', label=f'{condition_name} 총 환자수')
    ax.set_title(f'서울시 노인 {condition_name} 환자수 추이', fontsize=15)
    ax.set_xlabel('연도')
    ax.set_ylabel('총 노인 환자수 (명)')
    ax.grid(True)
    if not total_patients_df.empty:
        ax.set_xticks(total_patients_df['연도'].unique())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.legend(fontsize=10)
    st.pyplot(fig)

def plot_gender_elderly_trend(patients_gender_df, condition_name):
    if patients_gender_df is None or patients_gender_df.empty:
        st.info(f"노인 {condition_name} 성별 환자수 추이 데이터를 그릴 수 없습니다.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    if '남' in patients_gender_df.columns:
        ax.plot(patients_gender_df['연도'], patients_gender_df['남'], marker='o', label='남성', color='dodgerblue')
    if '여' in patients_gender_df.columns:
        ax.plot(patients_gender_df['연도'], patients_gender_df['여'], marker='s', label='여성', color='hotpink')
    ax.set_title(f'서울시 노인 {condition_name} 환자수 추이 (성별)', fontsize=15)
    ax.set_xlabel('연도')
    ax.set_ylabel('노인 환자수 (명)')
    ax.legend()
    ax.grid(True)
    if not patients_gender_df.empty:
        ax.set_xticks(patients_gender_df['연도'].unique())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    st.pyplot(fig)

def plot_subgroup_gender_elderly_trend(patients_subgroup_df, condition_name):
    if patients_subgroup_df is None or patients_subgroup_df.empty:
        st.info(f"노인 {condition_name} 세부 연령대 및 성별 환자수 추이 데이터를 그릴 수 없습니다.")
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=patients_subgroup_df, x='연도', y='환자수', hue='세부연령그룹', style='성별', marker='o', markersize=7, ax=ax)
    ax.set_title(f'서울시 {condition_name} 노인 환자수 추이 (세부 연령대 및 성별)', fontsize=15)
    ax.set_xlabel('연도')
    ax.set_ylabel('환자 수 (명)')
    if not patients_subgroup_df.empty:
        ax.set_xticks(sorted(patients_subgroup_df['연도'].unique()))
    ax.legend(title='구분', bbox_to_anchor=(1.02, 1), loc='upper left', title_fontsize='11', fontsize='10')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.tight_layout(rect=[0,0,0.85,1])
    st.pyplot(fig)

def plot_all_conditions_yearly_comparison(all_conditions_summary_df, selected_year_int):
    if all_conditions_summary_df.empty:
        st.info("종합 비교를 위한 데이터가 없습니다.")
        return
    year_df_to_plot = all_conditions_summary_df[all_conditions_summary_df['연도'] == selected_year_int].copy()
    if year_df_to_plot.empty:
        st.info(f"데이터가 없어 종합 비교 그래프를 생성할 수 없습니다.")
        return
    year_df_to_plot = year_df_to_plot.sort_values(by='총 노인 환자수', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=year_df_to_plot, x='질환명', y='총 노인 환자수', color='teal', ax=ax, label='총 노인 환자수')
    ax.set_title(f'서울시 노인 정신질환별 환자수 비교', fontsize=15)
    ax.set_xlabel('질환명')
    ax.set_ylabel('총 노인 환자수 (명)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

def plot_pie_chart_by_year(all_conditions_summary_df, selected_year_int):
    if all_conditions_summary_df.empty:
        st.info("파이차트 생성을 위한 데이터가 없습니다.")
        return
    year_data_for_pie = all_conditions_summary_df[all_conditions_summary_df['연도'] == selected_year_int].copy()
    if year_data_for_pie.empty or year_data_for_pie['총 노인 환자수'].sum() == 0:
        st.info(f"질환별 환자 수 비율을 계산할 데이터가 없습니다.")
        return
    year_data_for_pie = year_data_for_pie.sort_values(by='총 노인 환자수', ascending=False)
    fig, ax = plt.subplots(figsize=(9, 9))
    wedges, texts, autotexts = ax.pie(
        year_data_for_pie['총 노인 환자수'], labels=year_data_for_pie['질환명'],
        autopct=lambda p: '{:.1f}% ({:,.0f}명)'.format(p, p * np.sum(year_data_for_pie['총 노인 환자수']) / 100),
        startangle=140, pctdistance=0.75
    )
    for text in texts: text.set_fontsize(10)
    for autotext in autotexts: autotext.set_fontsize(9); autotext.set_color('black')
    centre_circle = plt.Circle((0,0),0.60,fc='white'); fig.gca().add_artist(centre_circle)
    ax.set_title(f'서울시 전체 노인 정신질환별 환자 수 비율', fontsize=16)
    ax.axis('equal')
    ax.legend(year_data_for_pie['질환명'], title="질환명", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

def plot_sigungu_mental_patients_by_condition_year(df_sigungu_mental_total, selected_condition, selected_year):
    if df_sigungu_mental_total.empty:
        st.info(f"<{selected_condition}>에 대한 구별 정신질환자 수 데이터가 없습니다.")
        return

    df_plot = df_sigungu_mental_total[
        (df_sigungu_mental_total['질환명'] == selected_condition) &
        (df_sigungu_mental_total['연도'] == selected_year)
    ].copy()

    if df_plot.empty or ('질환별_노인_환자수_총합' in df_plot.columns and df_plot['질환별_노인_환자수_총합'].sum() == 0):
        st.info(f"<{selected_condition}>에 대한 유의미한 구별 환자 수 데이터가 없습니다.")
        return
    if '질환별_노인_환자수_총합' not in df_plot.columns:
        st.warning(f"<{selected_condition}> 데이터에 '질환별_노인_환자수_총합' 컬럼이 없습니다.")
        return

    df_plot = df_plot.sort_values(by='질환별_노인_환자수_총합', ascending=False)

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.barplot(data=df_plot, x='시군구', y='질환별_노인_환자수_총합', color='lightcoral', ax=ax, label=f'{selected_condition} 환자수')
    ax.set_title(f'서울시 자치구별 {selected_condition} 노인 환자 수', fontsize=16)
    ax.set_xlabel('자치구', fontsize=14)
    ax.set_ylabel(f'{selected_condition} 노인 환자 수 (명)', fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

# 새로운 함수: 모든 질환 연도별 환자수 추이 (꺾은선)
def plot_all_conditions_trend_lineplot(all_years_summary_df):
    if all_years_summary_df is None or all_years_summary_df.empty:
        st.info("꺾은선 그래프를 위한 종합 데이터가 없습니다.")
        return
    plt.figure(figsize=(15, 8))
    sns.lineplot(
        data=all_years_summary_df,
        x='연도',
        y='총 노인 환자수',
        hue='질환명',
        marker='o',
        palette='tab10'
    )
    plt.title('서울시 5대 정신질환별 노인 환자수 연도별 추이', fontsize=16)
    plt.xlabel('연도', fontsize=12)
    plt.ylabel('총 노인 환자수 (명)', fontsize=12)
    if not all_years_summary_df.empty:
        unique_years_in_data = sorted(all_years_summary_df['연도'].unique())
        plt.xticks(unique_years_in_data, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='질환명', bbox_to_anchor=(1.02, 1), loc='upper left', title_fontsize='11', fontsize='10')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0,0,0.85,1])
    st.pyplot(plt)

# 새로운 함수: 모든 질환 연도별 전체 노인 인구 대비 환자 비율 추이 (꺾은선)
def plot_elderly_population_ratio_trend_lineplot(df_merged_ratio_seoul_total):
    if df_merged_ratio_seoul_total is None or df_merged_ratio_seoul_total.empty or '노인 인구 대비 환자 비율 (%)' not in df_merged_ratio_seoul_total.columns:
        st.info("노인 인구 대비 환자 비율 추이 데이터를 그릴 수 없거나 필요한 컬럼이 없습니다.")
        return
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=df_merged_ratio_seoul_total, x='연도', y='노인 인구 대비 환자 비율 (%)', hue='질환명', marker='o', palette='tab10')
    plt.title('서울시 연도별 정신질환별 노인 환자 비율 (전체 노인 인구 대비)', fontsize=16)
    plt.xlabel('연도', fontsize=12)
    plt.ylabel('노인 인구 대비 환자 비율 (%)', fontsize=12)
    if not df_merged_ratio_seoul_total.empty:
        plt.xticks(sorted(df_merged_ratio_seoul_total['연도'].unique()), fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}%')) # Y축 % 포맷
    plt.legend(title='질환명', bbox_to_anchor=(1.02, 1), loc='upper left', title_fontsize='11', fontsize='10')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0,0,0.85,1])
    st.pyplot(plt)

# 새로운 함수: 구별 <질환명> 노인 환자 "비율" (전체 노인 인구 대비)
def plot_sigungu_mental_patients_ratio_by_condition_year(df_sigungu_ratio_data, selected_condition, selected_year):
    if df_sigungu_ratio_data.empty:
        st.info(f"<{selected_condition}>에 대한 구별 정신질환자 비율 데이터가 없습니다.")
        return

    df_plot = df_sigungu_ratio_data[
        (df_sigungu_ratio_data['질환명'] == selected_condition) &
        (df_sigungu_ratio_data['연도'] == selected_year)
    ].copy()

    ratio_col_name = '전체노인인구_대비_질환자_비율(%)'
    if df_plot.empty or (ratio_col_name in df_plot.columns and df_plot[ratio_col_name].sum() == 0):
        st.info(f"<{selected_condition}>에 대한 유의미한 구별 환자 비율 데이터가 없습니다.")
        return
    if ratio_col_name not in df_plot.columns:
        st.warning(f"<{selected_condition}> 데이터에 '{ratio_col_name}' 컬럼이 없습니다.")
        return

    df_plot = df_plot.sort_values(by=ratio_col_name, ascending=False)

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.barplot(data=df_plot, x='시군구', y=ratio_col_name, color='mediumpurple', ax=ax, label=f'{selected_condition} 환자 비율(%)')
    ax.set_title(f'서울시 자치구별 {selected_condition} 노인 환자 비율', fontsize=16)
    ax.set_xlabel('자치구', fontsize=14)
    ax.set_ylabel(f'{selected_condition} 환자 비율 (%)', fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}%'))
    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

# --- END OF chart_utils.py ---
