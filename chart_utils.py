# --- START OF MODIFIED FILE chart_utils.py ---
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt # Altair 예시 함수를 위해 유지
import pandas as pd
import numpy as np

# ... (이전 코드들은 동일하게 유지) ...

# --- 기존 시각화 함수 (정신질환 관련 부분만 수정) ---
def plot_total_elderly_trend(total_patients_df, condition_name):
    if total_patients_df is None or total_patients_df.empty: 
        st.info(f"노인 {condition_name} 환자수 총계 추이 데이터를 그릴 수 없습니다.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=total_patients_df, x='연도', y='총 노인 환자수', marker='o', ax=ax, color='steelblue', label=f'{condition_name} 총 환자수')
    ax.set_title(f'서울시 노인 {condition_name} 환자수 추이', fontsize=15)
    ax.set_xlabel('연도')
    ax.set_ylabel('총 노인 환자수 (명)') # 요청사항 2: (명) 추가
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
    ax.set_ylabel('노인 환자수 (명)') # 요청사항 2: (명) 추가
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
    ax.set_ylabel('환자 수 (명)') # 요청사항 2: (명) 추가
    if not patients_subgroup_df.empty: 
        ax.set_xticks(sorted(patients_subgroup_df['연도'].unique()))
    ax.legend(title='구분', bbox_to_anchor=(1.02, 1), loc='upper left', title_fontsize='11', fontsize='10')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.tight_layout(rect=[0,0,0.85,1])
    st.pyplot(fig)

def plot_all_conditions_yearly_comparison(all_conditions_summary_df, selected_year_int): # selected_year_int는 데이터 필터링에만 사용
    if all_conditions_summary_df.empty: 
        st.info("종합 비교를 위한 데이터가 없습니다.")
        return
    year_df_to_plot = all_conditions_summary_df[all_conditions_summary_df['연도'] == selected_year_int].copy()
    if year_df_to_plot.empty: 
        st.info(f"{selected_year_int}년 데이터가 없어 종합 비교 그래프를 생성할 수 없습니다.")
        return
    year_df_to_plot = year_df_to_plot.sort_values(by='총 노인 환자수', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=year_df_to_plot, x='질환명', y='총 노인 환자수', color='teal', ax=ax, label='총 노인 환자수')
    ax.set_title(f'서울시 노인 정신질환별 환자수 비교', fontsize=15) # 요청사항 7: 연도 제거
    ax.set_xlabel('질환명')
    ax.set_ylabel('총 노인 환자수 (명)') # (명) 추가
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

def plot_pie_chart_by_year(all_conditions_summary_df, selected_year_int): # selected_year_int는 데이터 필터링에만 사용
    if all_conditions_summary_df.empty: 
        st.info("파이차트 생성을 위한 데이터가 없습니다.")
        return
    year_data_for_pie = all_conditions_summary_df[all_conditions_summary_df['연도'] == selected_year_int].copy()
    if year_data_for_pie.empty or year_data_for_pie['총 노인 환자수'].sum() == 0: 
        st.info(f"{selected_year_int}년 질환별 환자 수 비율을 계산할 데이터가 없습니다.")
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
    ax.set_title(f'서울시 전체 노인 정신질환별 환자 수 비율', fontsize=16) # 요청사항 7: 연도 제거
    ax.axis('equal')
    ax.legend(year_data_for_pie['질환명'], title="질환명", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

def plot_sigungu_mental_patients_by_condition_year(df_sigungu_mental_total, selected_condition, selected_year): # selected_year는 데이터 필터링에만 사용
    if df_sigungu_mental_total.empty:
        st.info(f"{selected_condition}에 대한 구별 정신질환자 수 데이터가 없습니다.") # 연도 정보 제거
        return

    df_plot = df_sigungu_mental_total[
        (df_sigungu_mental_total['질환명'] == selected_condition) &
        (df_sigungu_mental_total['연도'] == selected_year) # 데이터 필터링 시에는 연도 사용
    ].copy()

    if df_plot.empty or ('질환별_노인_환자수_총합' in df_plot.columns and df_plot['질환별_노인_환자수_총합'].sum() == 0):
        st.info(f"{selected_condition}에 대한 유의미한 구별 환자 수 데이터가 없습니다.") # 연도 정보 제거
        return
    if '질환별_노인_환자수_총합' not in df_plot.columns:
        st.warning(f"{selected_condition} 데이터에 '질환별_노인_환자수_총합' 컬럼이 없습니다.") # 연도 정보 제거
        return

    df_plot = df_plot.sort_values(by='질환별_노인_환자수_총합', ascending=False)

    fig, ax = plt.subplots(figsize=(20, 10))
    bars = sns.barplot(data=df_plot, x='시군구', y='질환별_노인_환자수_총합', color='lightcoral', ax=ax, label=f'{selected_condition} 환자수')
    ax.set_title(f'서울시 구별 <{selected_condition}> 노인 환자 수', fontsize=16) # 요청사항 9: 연도 제거
    ax.set_xlabel('자치구', fontsize=14)
    ax.set_ylabel(f'{selected_condition} 노인 환자 수 (명)', fontsize=14) # (명) 추가
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    for bar_patch in bars.patches:
        if bar_patch.get_height() > 0:
            y_offset = bar_patch.get_height() * 0.01 
            if y_offset < 1 and bar_patch.get_height() > 0: y_offset = 3 
            elif y_offset < 1 and bar_patch.get_height() <=0 : y_offset = -3

            ax.text(
                x=bar_patch.get_x() + bar_patch.get_width() / 2,
                y=bar_patch.get_height() + y_offset, 
                s=f"{int(bar_patch.get_height()):,}", 
                ha='center',
                va='bottom' if bar_patch.get_height() >=0 else 'top',
                fontsize=8
            )
    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

# ... (이하 다른 sheet_charts 함수들은 이전과 동일하게 유지) ...
def draw_sheet0_charts(df_metrics_input, year: int, figsize1: tuple = (14, 6), figsize2: tuple = (14, 6), figsize3: tuple = (14, 7), dpi: int = 100 ): None
def draw_sheet1_charts(df_metrics_input, year: int, figsize1: tuple = (14, 6), figsize2: tuple = (14, 6), figsize3: tuple = (14, 7), dpi: int = 100 ): None
def draw_nursing_csv_charts(df_welf_input: pd.DataFrame, df_centers_input: pd.DataFrame, year: int, figsize1: tuple = (14, 6), figsize2: tuple = (14, 5), dpi: int = 100 ): None
def draw_sheet3_charts(df_metrics_input, year: int, figsize1: tuple = (14, 6), figsize2: tuple = (14, 6), figsize3: tuple = (14, 7), dpi: int = 100 ): None
def draw_sheet4_charts(df_metrics_input, year: int, figsize1: tuple = (14, 6), dpi: int = 100 ): None
def draw_sheet5_charts(df_metrics_input, year: int, figsize1: tuple = (14, 6), figsize2: tuple = (14, 6), figsize3: tuple = (14, 7), dpi: int = 100 ): None

# --- END OF chart_utils.py ---
