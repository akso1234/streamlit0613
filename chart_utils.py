from matplotlib import font_manager
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import pandas as pd

# ——————————————————————————————————————————————————
# 한글 폰트 설정 (Matplotlib용)
# ——————————————————————————————————————————————————
font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
font_name = font_manager.FontProperties(fname=font_path).get_name()

font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False
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

    '소계'라는 gu(구 이름)가 포함되어 있다면, 그 행을 제외하고 그래프를 그립니다.
    """
    # -----------------------------
    # 1) '소계' 구 이름이 있다면 제거
    # -----------------------------
    df_plot = df_hosp.copy()
    if "gu" in df_plot.columns:
        df_plot = df_plot[df_plot["gu"] != "소계"].reset_index(drop=True)

    # -----------------------------
    # 2) 숫자형으로 변환 (NaN 방지)
    # -----------------------------
    types = ["소계", "종합병원", "병원", "의원", "요양병원"]
    for inst in types:
        df_plot[inst] = pd.to_numeric(df_plot[inst], errors="coerce").fillna(0)

    # -----------------------------
    # 3) inst별로 Figure를 하나씩 생성
    # -----------------------------
    for inst in types:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(df_plot["gu"], df_plot[inst])
        ax.set_title(f"구별 {inst} 수", fontsize=14)
        ax.set_xlabel("자치구", fontsize=12)
        ax.set_ylabel("기관 수", fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=90)
        plt.tight_layout()
        st.pyplot(fig)

def draw_aggregate_hospital_bed_charts(df_hosp: pd.DataFrame, df_beds: pd.DataFrame):
    """
    df_hosp: load_raw_data()에서 반환된 DataFrame (컬럼: 'gu', '종합병원', '병원', '의원', '요양병원', ...)
    df_beds: load_raw_data()에서 반환된 DataFrame (컬럼: 'gu', '종합병원', '병원', '의원', '요양병원', ...)
    
    - (A) 전체 의료기관 유형별 전체 병상 수 막대그래프
    - (B) 전체 의료기관 유형별 전체 병원 수 막대그래프
    - (C) 의료기관 유형별 병원당 평균 병상 수 막대그래프
    """
    # 1) '소계' 구 이름을 제거하고, 숫자로 변환
    df_h = df_hosp.copy()
    df_b = df_beds.copy()
    # '소계' 행 제거
    if "gu" in df_h.columns:
        df_h = df_h[df_h["gu"] != "소계"].reset_index(drop=True)
    if "gu" in df_b.columns:
        df_b = df_b[df_b["gu"] != "소계"].reset_index(drop=True)

    types = ["종합병원", "병원", "의원", "요양병원"]

    # 2) 전체 병원 수 total_hosp 및 전체 병상 수 total_beds 계산
    total_hosp = {}
    total_beds = {}
    for t in types:
        # df_h[t]와 df_b[t]를 숫자형으로 변환 후 합산
        total_hosp[t] = pd.to_numeric(df_h[t], errors="coerce").sum()
        total_beds[t] = pd.to_numeric(df_b[t], errors="coerce").sum()

    # 3) 평균 병상 수 avg_beds 계산 (병상수/병원수)
    avg_beds = {}
    for t in types:
        # 병원 수가 0일 경우를 방지
        if total_hosp[t] and total_hosp[t] > 0:
            avg_beds[t] = total_beds[t] / total_hosp[t]
        else:
            avg_beds[t] = 0

    # ——————————————————————————————————
    # (A) 의료기관 유형별 전체 병원 수
    # ——————————————————————————————————
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(total_hosp.keys(), total_hosp.values(), color='tab:green')
    ax2.set_title('의료기관 유형별 전체 병원 수', fontsize=14)
    ax2.set_ylabel('병원 수', fontsize=12)
    ax2.set_xlabel('기관 유형', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig2)

    # ——————————————————————————————————
    # (B) 의료기관 유형별 전체 병상 수
    # ——————————————————————————————————
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(total_beds.keys(), total_beds.values(), color='tab:blue')
    ax1.set_title('의료기관 유형별 전체 병상 수', fontsize=14)
    ax1.set_ylabel('병상 수', fontsize=12)
    ax1.set_xlabel('기관 유형', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig1)

    # ——————————————————————————————————
    # (C) 의료기관 유형별 병원당 평균 병상 수
    # ——————————————————————————————————
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(avg_beds.keys(), avg_beds.values(), color='tab:orange')
    ax3.set_title('의료기관 유형별 병원당 평균 병상 수', fontsize=14)
    ax3.set_ylabel('평균 병상 수', fontsize=12)
    ax3.set_xlabel('기관 유형', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig3)

# --- (1) 더 잘 보이도록 폰트 사이즈 조정 및 대비 설정 함수 ---
def draw_avg_beds_heatmap(df_hosp: pd.DataFrame, df_beds: pd.DataFrame):
    """
    df_hosp: 'gu', '종합병원', '병원', '요양병원' 컬럼을 포함
    df_beds: 'gu', '종합병원', '병원', '요양병원' 컬럼을 포함
    """

    # 1) '소계' 구 이름 제거
    df_h = df_hosp.copy().reset_index(drop=True)
    df_b = df_beds.copy().reset_index(drop=True)
    if "gu" in df_h.columns:
        df_h = df_h[df_h["gu"] != "소계"].reset_index(drop=True)
    if "gu" in df_b.columns:
        df_b = df_b[df_b["gu"] != "소계"].reset_index(drop=True)

    # 2) 숫자형으로 변환
    types = ["종합병원", "병원", "요양병원"]
    for t in types:
        df_h[t] = pd.to_numeric(df_h[t], errors="coerce").fillna(0)
        df_b[t] = pd.to_numeric(df_b[t], errors="coerce").fillna(0)

    # 3) 구 × 유형별 평균병상/병원 계산
    records = []
    for idx, row in df_h.iterrows():
        gu = row["gu"]
        for t in types:
            hosp_cnt = row[t]
            bed_cnt = df_b.loc[idx, t]
            avg_bed = bed_cnt / hosp_cnt if hosp_cnt and hosp_cnt > 0 else np.nan
            records.append({
                "gu": gu,
                "type": t,
                "평균병상/병원": avg_bed
            })
    stats = pd.DataFrame(records)

    # 4) Pivot
    pivot = stats.pivot(index="gu", columns="type", values="평균병상/병원")

    # 5) Matplotlib으로 히트맵 그리기
    heat_data = pivot.values  # shape: (n_gu, 3)

    # ———— 5.1) Figure 크기 및 DPI 키우기 ————
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)

    # ———— 5.2) 컬러맵을 대비가 조금 더 진한 Blues로 설정 ————
    cmap = plt.cm.Blues

    im = ax.imshow(heat_data, cmap=cmap, aspect="auto", vmin=np.nanmin(heat_data), vmax=np.nanmax(heat_data))

    # 축 눈금 설정
    ax.set_xticks(np.arange(len(types)))
    ax.set_xticklabels(types, fontsize=14, fontweight='bold')
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=12)

    # 축 레이블(제목) 설정
    ax.set_xlabel("기관 유형", fontsize=16, fontweight='bold')
    ax.set_ylabel("자치구", fontsize=16, fontweight='bold')
    ax.set_title("구별 기관 유형별 평균 병상 수", fontsize=18, fontweight='bold')

    # ———— 5.3) 셀 내부 값(annot)을 배경 대비에 따라 색상 동적으로 조정 ————
    # pivot 값에서 최댓값/최솟값에 따라 텍스트 색을 결정합니다.
    vmax = np.nanmax(heat_data)
    vmin = np.nanmin(heat_data)

    for i in range(heat_data.shape[0]):
        for j in range(heat_data.shape[1]):
            val = heat_data[i, j]
            if np.isnan(val):
                text = "-"
            else:
                text = f"{val:.1f}"

            # 대비 기준(텍스트 색상): 
            # 값이 전체 범위의 절반보다 크면 흰색, 작으면 검정색
            if not np.isnan(val) and (val > (vmin + vmax) / 2):
                text_color = "white"
            else:
                text_color = "black"

            ax.text(
                j, i, text,
                ha="center", va="center",
                color=text_color,
                fontsize=12,  # annot 폰트 크기
                fontweight='bold'  # annot 폰트 굵기
            )

    # ———— 5.4) 축 눈금을 그리드처럼 보이도록 그리기 ————
    ax.set_xticks(np.arange(len(types) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(pivot.index) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # ———— 5.5) 컬러바 추가 ————
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("평균 병상 수", fontsize=14, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

    return pivot


def draw_sheet0_charts(
    df_metrics,
    figsize1: tuple = (12, 5),
    figsize2: tuple = (15, 5),
    figsize3: tuple = (14, 6),
    dpi: int = 100
) -> None:
    regions = df_metrics.index.tolist()
    x = np.arange(len(regions))
    width = 0.35
    w = 0.2

    # (1) 정원·현원·추가 수용
    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    ax1.bar(x - width, df_metrics['capacity'],   width=width, label='정원', color='#55A868')
    ax1.bar(x,         df_metrics['occupancy'], width=width, label='현원', color='#C44E52')
    ax1.bar(x + width, df_metrics['additional'], width=width, label='추가 수용', color='#EE8454')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('명', fontsize=12)
    ax1.set_title('구별 노인주거복지시설: 정원·현원·추가 수용 가능 인원수', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig1)

    # (2) 시설수·종사자수
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    ax2.bar(x - w/2, df_metrics['facility'], width=w, label='시설수', color='#4C72B0')
    ax2.bar(x + w/2, df_metrics['staff'],    width=w, label='종사자수', color='#C44E52')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax2.set_title('구별 노인주거복지시설 시설수·종사자수', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('명', fontsize=12)
    ax2.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)

    # (3) 정원/종사자 vs 현원/종사자 비율
    fig3, ax3 = plt.subplots(figsize=figsize3, dpi=dpi)
    ax3.bar(x - w/2, df_metrics['cap_per_staff'], width=w, label='정원/종사자수 비율', color='#55A868')
    ax3.bar(x + w/2, df_metrics['occ_per_staff'], width=w, label='현원/종사자수 비율', color='#8172B2')
    ax3.set_xticks(x)
    ax3.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('1명당 케어 인원수', fontsize=12)
    ax3.set_title('구별 노인주거복지시설 정원 vs 현원 1명당 케어 인원수', 
                  fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig3)

def draw_sheet1_charts(
    df_metrics,
    figsize1: tuple = (12, 5),
    figsize2: tuple = (15, 5),
    figsize3: tuple = (14, 6),
    dpi: int = 100
) -> None:
    regions = df_metrics.index.tolist()
    x = np.arange(len(regions))
    width = 0.35
    w = 0.2

    # (1) 정원·현원·추가 수용
    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    ax1.bar(x - width, df_metrics['capacity'],   width=width, label='정원', color='#55A868')
    ax1.bar(x,         df_metrics['occupancy'], width=width, label='현원', color='#C44E52')
    ax1.bar(x + width, df_metrics['additional'], width=width, label='추가 수용', color='#EE8454')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('명', fontsize=12)
    ax1.set_title('구별 노인의료복지시설: 정원·현원·추가 수용 가능 인원수', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig1)

    # (2) 시설수·종사자수
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    ax2.bar(x - w/2, df_metrics['facility'], width=w, label='시설수', color='#4C72B0')
    ax2.bar(x + w/2, df_metrics['staff'],    width=w, label='종사자수', color='#C44E52')
    ax2.set_xticks(x)
    ax2.set_ylabel('명', fontsize=12)
    ax2.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax2.set_title('구별 노인의료복지시설 시설수·종사자수', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)

    # (3) 정원/종사자 vs 현원/종사자 비율
    fig3, ax3 = plt.subplots(figsize=figsize3, dpi=dpi)
    ax3.bar(x - w/2, df_metrics['cap_per_staff'], width=w, label='정원/종사자수 비율', color='#55A868')
    ax3.bar(x + w/2, df_metrics['occ_per_staff'], width=w, label='현원/종사자수 비율', color='#8172B2')
    ax3.set_xticks(x)
    ax3.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('1명당 케어 인원수', fontsize=12)
    ax3.set_title('구별 노인의료복지시설 정원 vs 현원 1명당 케어 인원수', 
                  fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig3)


def draw_nursing_csv_charts(
    df_welf: pd.DataFrame,
    df_centers: pd.DataFrame,
    figsize1: tuple = (14, 5),
    figsize2: tuple = (14, 5),
    dpi: int = 100
) -> None:
    """
    load_nursing_sheet1()이 반환한 두 DataFrame(df_welf, df_centers)을 
    입력받아 다음 두 차트를 Streamlit에 렌더링합니다:
      1) 노인복지관 시설수 vs 종사자수
      2) 경로당+노인교실 합계(시설수)
    """
    regions = df_welf.index.tolist()
    x = np.arange(len(regions))
    w = 0.4

    # (1) 노인복지관: 시설수 vs 종사자수
    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    ax1.bar(x - w/2, df_welf['facility'], width=w, label='시설수', color='#4C72B0')
    ax1.bar(x + w/2, df_welf['staff'],    width=w, label='종사자수', color='#C44E52')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('개소 / 명', fontsize=12)
    ax1.set_title('구별 노인복지관 시설수 vs 종사자수', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig1)

    # (2) 경로당+노인교실 합계
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    ax2.bar(x, df_centers['facility'], width=w, color='tab:green')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('개소', fontsize=12)
    ax2.set_title('구별 경로당+노인교실 합계', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)



def draw_sheet3_charts(
    df_metrics,
    figsize1: tuple = (12, 5),
    figsize2: tuple = (15, 5),
    figsize3: tuple = (14, 6),
    dpi: int = 100
) -> None:
    regions = df_metrics.index.tolist()
    x = np.arange(len(regions))
    width = 0.35
    w = 0.2

    # (1) 정원·현원·추가 수용
    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    ax1.bar(x - width/2, df_metrics['capacity'],   width=width, label='정원', color='#55A868')
    ax1.bar(x + width/2,         df_metrics['occupancy'], width=width, label='현원', color='#C44E52')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('명', fontsize=12)
    ax1.set_title('구별 재가노인복지시설: 정원·현원 인원수', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig1)

    # (2) 시설수·종사자수
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    ax2.bar(x - w/2, df_metrics['facility'], width=w, label='시설수', color='#4C72B0')
    ax2.bar(x + w/2, df_metrics['staff'],    width=w, label='종사자수', color='#C44E52')
    ax2.set_xticks(x)
    ax2.set_ylabel('명', fontsize=12)
    ax2.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax2.set_title('구별 재가노인복지시설 시설수·종사자수', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)

    # (3) 정원/종사자 vs 현원/종사자 비율
    fig3, ax3 = plt.subplots(figsize=figsize3, dpi=dpi)
    ax3.bar(x - w/2, df_metrics['cap_per_staff'], width=w, label='정원/종사자수 비율', color='#55A868')
    ax3.bar(x + w/2, df_metrics['occ_per_staff'], width=w, label='현원/종사자수 비율', color='#8172B2')
    ax3.set_xticks(x)
    ax3.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('1명당 케어 인원수', fontsize=12)
    ax3.set_title('구별 재가노인복지시설 정원 vs 현원 1명당 케어 인원수', 
                  fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig3)


def draw_sheet4_charts(
    df_metrics,
    figsize1: tuple = (12, 5),
    dpi: int = 100
) -> None:
    regions = df_metrics.index.tolist()
    x = np.arange(len(regions))
    width = 0.35
    w = 0.2

    # (1) 정원·현원·추가 수용
    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    ax1.bar(x - w/2, df_metrics['facility'],   width=w)
    ax1.bar(x + w/2, df_metrics['staff'],      width=w)
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax1.legend(['시설수','종사자'])
    ax1.set_ylabel('명', fontsize=12)
    ax1.set_title('구별 노인일자리지원 시설수·종사자수')
    plt.tight_layout()
    st.pyplot(fig1)

def draw_sheet5_charts(
    df_metrics,
    figsize1: tuple = (12, 5),
    figsize2: tuple = (15, 5),
    figsize3: tuple = (14, 6),
    dpi: int = 100
) -> None:
    regions = df_metrics.index.tolist()
    x = np.arange(len(regions))
    width = 0.35
    w = 0.2

    # (1) 정원·현원·추가 수용
    fig1, ax1 = plt.subplots(figsize=figsize1, dpi=dpi)
    ax1.bar(x - width, df_metrics['capacity'],   width=width, label='정원', color='#55A868')
    ax1.bar(x,         df_metrics['occupancy'], width=width, label='현원', color='#C44E52')
    ax1.bar(x + width, df_metrics['additional'], width=width, label='추가 수용', color='#EE8454')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('명', fontsize=12)
    ax1.set_title('구별 치매전담형 장기요양기관: 정원·현원·추가 수용 가능 인원수', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig1)

    # (2) 시설수·종사자수
    fig2, ax2 = plt.subplots(figsize=figsize2, dpi=dpi)
    ax2.bar(x - w/2, df_metrics['facility'], width=w, label='시설수', color='#4C72B0')
    ax2.bar(x + w/2, df_metrics['staff'],    width=w, label='종사자수', color='#C44E52')
    ax2.set_xticks(x)
    ax2.set_ylabel('명', fontsize=12)
    ax2.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax2.set_title('구별 치매전담형 장기요양기관 시설수·종사자수', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)

    # (3) 정원/종사자 vs 현원/종사자 비율
    fig3, ax3 = plt.subplots(figsize=figsize3, dpi=dpi)
    ax3.bar(x - w/2, df_metrics['cap_per_staff'], width=w, label='정원/종사자수 비율', color='#55A868')
    ax3.bar(x + w/2, df_metrics['occ_per_staff'], width=w, label='현원/종사자수 비율', color='#8172B2')
    ax3.set_xticks(x)
    ax3.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('1명당 케어 인원수', fontsize=12)
    ax3.set_title('구별 치매전담형 장기요양기관 정원 vs 현원 1명당 케어 인원수', 
                  fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig3)