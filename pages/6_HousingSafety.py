import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import set_korean_font, load_csv # 해당 페이지에 맞는 utils 함수 임포트
import os

# --- 데이터 전처리 함수 ---
@st.cache_data
def preprocess_rescue_data_cached(df_rescue_raw):
    if df_rescue_raw is None: 
        st.warning("구조활동현황 원본 데이터가 없습니다.")
        return pd.DataFrame()
    # 간단한 전처리 예시 (실제 필요한 전처리에 따라 수정)
    # 여기서는 원본을 그대로 사용하거나, 필요한 최소한의 처리만 가정
    df_processed = df_rescue_raw.copy()
    # 예시: '발생장소_구' 컬럼이 있는지 확인하고, 없다면 경고 후 빈 DF 반환
    if '발생장소_구' not in df_processed.columns:
        st.warning("구조활동현황 데이터에 '발생장소_구' 컬럼이 없습니다.")
        return pd.DataFrame()
    df_processed['발생장소_구'] = df_processed['발생장소_구'].astype(str).str.strip()
    return df_processed

@st.cache_data
def preprocess_elderly_data_for_housing_cached(df_elderly_raw):
    if df_elderly_raw is None: 
        st.warning("고령자현황 원본 데이터가 없습니다.")
        return pd.DataFrame()
    df_elderly_processed = df_elderly_raw.copy()
    try:
        # 컬럼명 정제 (따옴표 및 공백 제거)
        new_columns = []
        for ct in df_elderly_processed.columns:
            if isinstance(ct, tuple):
                new_columns.append(tuple(str(s).replace('"', '').strip() if isinstance(s, str) else str(s) for s in ct))
            else: # 단일 레벨 컬럼인 경우
                new_columns.append(str(ct).replace('"', '').strip() if isinstance(ct, str) else str(ct))
        
        # MultiIndex 또는 일반 Index로 설정
        if all(isinstance(col, tuple) for col in new_columns):
             # 모든 컬럼명이 튜플 형태이면 MultiIndex로 간주
            if len(new_columns) == len(df_elderly_processed.columns): # 컬럼 개수 일치 확인
                 df_elderly_processed.columns = pd.MultiIndex.from_tuples(new_columns) # MultiIndex names는 생략하거나 필요시 추가
            else:
                st.error("고령자현황: MultiIndex 컬럼명 생성 중 길이 불일치.")
                return pd.DataFrame()
        else:
            df_elderly_processed.columns = new_columns


        # 2023년 65세 이상 인구 총계 컬럼 찾기
        target_pop_col_tuple = None
        # MultiIndex 컬럼인 경우
        if isinstance(df_elderly_processed.columns, pd.MultiIndex) and len(df_elderly_processed.columns.levels) == 4:
            for col_tuple in df_elderly_processed.columns:
                # 컬럼 튜플의 각 요소가 문자열인지 확인하고 strip() 적용
                c0 = str(col_tuple[0]).strip()
                c1 = str(col_tuple[1]).strip()
                c2 = str(col_tuple[2]).strip()
                c3 = str(col_tuple[3]).strip()
                if c0.startswith('2023') and '65세이상 인구' in c1 and c2 == '소계' and c3 == '소계':
                    target_pop_col_tuple = col_tuple
                    break
        else: # 단일 레벨 컬럼이거나 MultiIndex 레벨이 다른 경우, 다른 방식으로 컬럼 찾기 (이 부분은 실제 데이터 구조에 맞게 조정 필요)
            st.warning("고령자현황: 예상된 4레벨 MultiIndex가 아니거나, 컬럼 구조가 다릅니다. 2023년 노인 인구 컬럼을 찾는 로직 수정이 필요할 수 있습니다.")
            # 임시로 컬럼 위치 기반 접근 시도 (매우 불안정하므로 실제 데이터 확인 후 수정 권장)
            if len(df_elderly_processed.columns) > 55 : # 예시 인덱스 (실제 파일 보고 결정)
                 potential_col_name_as_tuple = df_elderly_processed.columns[55] # 이 인덱스는 파일에 따라 달라짐
                 if isinstance(potential_col_name_as_tuple, tuple) and len(potential_col_name_as_tuple)==4:
                    if str(potential_col_name_as_tuple[0]).strip().startswith('2023') and \
                       '65세이상 인구' in str(potential_col_name_as_tuple[1]).strip() and \
                       '소계' == str(potential_col_name_as_tuple[2]).strip() and \
                       '소계' == str(potential_col_name_as_tuple[3]).strip():
                        target_pop_col_tuple = potential_col_name_as_tuple


        if target_pop_col_tuple is None:
            st.error("고령자현황: 2023년 65세 이상 인구 '소계' 컬럼을 찾지 못했습니다. 컬럼명을 확인해주세요.")
            # print("DEBUG (Elderly): Available columns:", df_elderly_processed.columns.tolist()[:10]) # 디버깅용
            return pd.DataFrame()

        # 자치구 정보 컬럼 (두 번째 컬럼으로 가정)
        if len(df_elderly_processed.columns) < 2:
            st.error("고령자현황: 자치구 정보 컬럼이 부족합니다.")
            return pd.DataFrame()
        gu_info_col_tuple_elderly = df_elderly_processed.columns[1]

        # '소계' 행 제외 및 필요한 컬럼 선택
        df_filtered_elderly = df_elderly_processed[df_elderly_processed[gu_info_col_tuple_elderly].astype(str).str.strip() != '소계'].copy()
        
        if gu_info_col_tuple_elderly in df_filtered_elderly.columns and target_pop_col_tuple in df_filtered_elderly.columns:
            df_final_elderly = df_filtered_elderly[[gu_info_col_tuple_elderly, target_pop_col_tuple]].copy()
            df_final_elderly.columns = ['발생장소_구', '노인인구수']
            df_final_elderly['발생장소_구'] = df_final_elderly['발생장소_구'].astype(str).str.strip()
            df_final_elderly['노인인구수'] = pd.to_numeric(df_final_elderly['노인인구수'].astype(str).str.replace(',','', regex=False), errors='coerce').fillna(0).astype(int)
            df_final_elderly.dropna(subset=['발생장소_구', '노인인구수'], inplace=True)
            return df_final_elderly
        else:
            st.error("고령자현황: 필요한 컬럼(자치구 또는 인구수)을 필터링된 데이터에서 찾을 수 없습니다.")
            return pd.DataFrame()
    except Exception as e: st.error(f"고령자현황 데이터 전처리 중 예외: {e}"); return pd.DataFrame()

@st.cache_data
def preprocess_housing_data_cached(df_housing_raw):
    if df_housing_raw is None: 
        st.warning("노후주택현황 원본 데이터가 없습니다.")
        return pd.DataFrame()
    df_housing_processed = df_housing_raw.copy()
    try:
        # 컬럼명 정제
        new_cols_h = []
        for ct_h in df_housing_processed.columns:
            if isinstance(ct_h, tuple):
                 new_cols_h.append(tuple(str(s).replace('"', '').strip() if isinstance(s, str) else str(s) for s in ct_h))
            else:
                 new_cols_h.append(str(ct_h).replace('"', '').strip() if isinstance(ct_h, str) else str(ct_h))

        if all(isinstance(col, tuple) for col in new_cols_h):
            if len(new_cols_h) == len(df_housing_processed.columns):
                df_housing_processed.columns = pd.MultiIndex.from_tuples(new_cols_h)
            else:
                st.error("노후주택: MultiIndex 컬럼명 생성 중 길이 불일치.")
                return pd.DataFrame()
        else:
             df_housing_processed.columns = new_cols_h
        
        # 2023년 30년 이상 노후주택 '계' 컬럼 찾기
        target_housing_col_tuple = None
        # MultiIndex 컬럼인 경우
        if isinstance(df_housing_processed.columns, pd.MultiIndex) and len(df_housing_processed.columns.levels) == 3:
            for col_tuple_h in df_housing_processed.columns:
                c0_h = str(col_tuple_h[0]).strip() # 연도 또는 '계'
                c1_h = str(col_tuple_h[1]).strip() # 노후기간
                c2_h = str(col_tuple_h[2]).strip() # 주택유형 또는 '계'
                if c0_h.startswith('2023') and c1_h in ["30년이상", "30년 이상"] and c2_h == "계":
                    target_housing_col_tuple = col_tuple_h
                    break
        else: # 단일 레벨 컬럼이거나 MultiIndex 레벨이 다른 경우
            st.warning("노후주택: 예상된 3레벨 MultiIndex가 아니거나, 컬럼 구조가 다릅니다. 2023년 30년 이상 노후주택 컬럼을 찾는 로직 수정이 필요할 수 있습니다.")
            # 임시 로직 (실제 데이터 구조 확인 후 수정 필요)
            for col_name_h in df_housing_processed.columns:
                if isinstance(col_name_h, str) and '2023' in col_name_h and ('30년이상' in col_name_h or '30년 이상' in col_name_h) and '계' in col_name_h: # 매우 단순한 가정
                    target_housing_col_tuple = col_name_h # 단일 문자열 컬럼명으로 취급
                    break


        if target_housing_col_tuple is None:
            st.error("노후 주택 데이터: 2023년 '30년 이상 계' 컬럼을 찾지 못했습니다. 컬럼명을 확인해주세요.")
            # print("DEBUG (Housing): Available columns:", df_housing_processed.columns.tolist()[:10]) # 디버깅용
            return pd.DataFrame()

        if len(df_housing_processed.columns) < 2:
            st.error("노후 주택 데이터: 자치구 정보 컬럼이 부족합니다.")
            return pd.DataFrame()
        gu_info_col_tuple_housing = df_housing_processed.columns[1] # 두 번째 컬럼을 자치구 정보로 가정

        df_filtered_housing = df_housing_processed[df_housing_processed[gu_info_col_tuple_housing].astype(str).str.strip() != '소계'].copy()
        
        if gu_info_col_tuple_housing in df_filtered_housing.columns and target_housing_col_tuple in df_filtered_housing.columns:
            df_final_housing = df_filtered_housing[[gu_info_col_tuple_housing, target_housing_col_tuple]].copy()
            df_final_housing.columns = ['발생장소_구', '노후주택수']
            df_final_housing['발생장소_구'] = df_final_housing['발생장소_구'].astype(str).str.strip()
            df_final_housing['노후주택수'] = pd.to_numeric(df_final_housing['노후주택수'].astype(str).str.replace(',','', regex=False), errors='coerce').fillna(0).astype(int)
            df_final_housing.dropna(subset=['발생장소_구', '노후주택수'], inplace=True)
            return df_final_housing
        else:
            st.error("노후 주택 데이터: 필요한 컬럼(자치구 또는 노후주택수)을 필터링된 데이터에서 찾을 수 없습니다.")
            return pd.DataFrame()
    except Exception as e: st.error(f"노후 주택 데이터 전처리 중 예외: {e}"); return pd.DataFrame()

# --- 시각화 함수 (이전과 동일) ---
def plot_gu_incident_counts(df_rescue):
    if df_rescue.empty or '발생장소_구' not in df_rescue.columns: st.info("구별 총 사고 발생 건수 데이터를 그릴 수 없습니다."); return
    gu_incident_counts = df_rescue['발생장소_구'].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=gu_incident_counts.index, y=gu_incident_counts.values, color='steelblue', ax=ax)
    ax.set_title('서울시 구별 총 사고 발생 건수', fontsize=16); ax.set_xlabel('자치구', fontsize=12); ax.set_ylabel('사고 건수', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10); ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.tight_layout(); st.pyplot(fig)

def plot_stacked_bar_incident_causes_by_gu(df_rescue, top_n_causes=7):
    if df_rescue.empty or '발생장소_구' not in df_rescue.columns or '사고원인' not in df_rescue.columns: st.info("구별 사고원인별 발생 건수 데이터를 그릴 수 없습니다."); return
    gu_cause_counts = df_rescue.groupby(['발생장소_구', '사고원인']).size().unstack(fill_value=0)
    gu_cause_counts['총계'] = gu_cause_counts.sum(axis=1)
    gu_cause_counts = gu_cause_counts.sort_values(by='총계', ascending=False).drop(columns='총계')
    if gu_cause_counts.shape[1] > top_n_causes:
        top_causes_sum = gu_cause_counts.sum(axis=0).nlargest(top_n_causes).index
        df_plot = gu_cause_counts[top_causes_sum].copy()
        df_plot['기타원인'] = gu_cause_counts.drop(columns=top_causes_sum).sum(axis=1)
    else:
        df_plot = gu_cause_counts.copy()
        if '기타원인' not in df_plot.columns and df_plot.shape[1] > 0 : df_plot['기타원인'] = 0
    if df_plot.empty: st.info("사고 원인별 집계 데이터가 비어있어 누적 막대 그래프를 생성할 수 없습니다."); return
    fig, ax = plt.subplots(figsize=(18, 10))
    df_plot.plot(kind='bar', stacked=True, ax=ax) 
    ax.set_title(f'서울시 구별 주요 사고원인별 발생 건수 (상위 {top_n_causes}개 및 기타)', fontsize=16, pad=15)
    ax.set_xlabel('자치구', fontsize=12); ax.set_ylabel('사고 건수', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10); ax.tick_params(axis='y', labelsize=10)
    ax.legend(title='사고원인', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.7); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.tight_layout(rect=[0, 0, 0.88, 1]); st.pyplot(fig)

def plot_pie_major_incident_causes(df_rescue, top_n=7):
    if df_rescue.empty or '사고원인' not in df_rescue.columns: st.info("주요 사고원인 비율 파이 차트를 그릴 수 없습니다."); return
    cause_counts = df_rescue['사고원인'].value_counts();
    if cause_counts.empty: st.info("사고원인 데이터가 없어 파이 차트를 생성할 수 없습니다."); return
    if len(cause_counts) > top_n:
        top_causes_pie = cause_counts.nlargest(top_n).copy()
        other_sum_pie = cause_counts.nsmallest(len(cause_counts) - top_n).sum()
        if other_sum_pie > 0: top_causes_pie.loc['기타원인'] = other_sum_pie
    else: top_causes_pie = cause_counts.copy()
    if top_causes_pie.empty: st.info("파이 차트를 위한 최종 사고원인 데이터가 비어있습니다."); return
    fig, ax = plt.subplots(figsize=(10, 8))
    patches, texts, autotexts = ax.pie(top_causes_pie, labels=top_causes_pie.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85, wedgeprops={'edgecolor': 'grey', 'linewidth': 0.7})
    for text in texts: text.set_fontsize(10)
    for autotext in autotexts: autotext.set_fontsize(9); autotext.set_color('black')
    ax.set_title(f'주요 사고원인 비율 (상위 {top_n}개 및 기타)', fontsize=16, pad=20); ax.axis('equal')
    plt.tight_layout(); st.pyplot(fig)

def plot_correlation_scatter_housing(merged_df, x_col, y_col, title_text):
    if merged_df.empty or x_col not in merged_df.columns or y_col not in merged_df.columns: st.info(f"'{title_text}' 산점도를 그릴 데이터가 없습니다."); return
    if not pd.api.types.is_numeric_dtype(merged_df[x_col]) or not pd.api.types.is_numeric_dtype(merged_df[y_col]): st.warning(f"산점도용 컬럼('{x_col}', '{y_col}') 중 숫자형이 아닌 것이 있습니다."); return
    try:
        correlation = merged_df[x_col].corr(merged_df[y_col])
        st.write(f"**상관계수 ({x_col} vs {y_col}): {correlation:.3f}**")
    except Exception as e:
        st.warning(f"상관계수 계산 중 오류 발생: {e}")
        correlation = None # 상관계수 계산 실패 시 None으로 설정

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=x_col, y=y_col, data=merged_df, ax=ax, color='darkcyan', scatter_kws={'s':60, 'alpha':0.65, 'edgecolor':'black'}, line_kws={'color':'red', 'linewidth':1.5})
    ax.set_title(title_text, fontsize=15); ax.set_xlabel(x_col, fontsize=12); ax.set_ylabel(y_col, fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}')); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.tight_layout(); st.pyplot(fig)

def plot_bubble_chart_housing(df_final_merged, target_cause_for_bubble):
    required_cols = ['노후주택수', '노인인구수', f'{target_cause_for_bubble}건수', '발생장소_구']
    if df_final_merged.empty or not all(c in df_final_merged.columns for c in required_cols): st.info(f"버블 차트({target_cause_for_bubble})를 그릴 데이터가 부족합니다."); return
    fig, ax = plt.subplots(figsize=(12, 8))
    bubble_sizes_data = df_final_merged[f'{target_cause_for_bubble}건수']
    min_bubble_size, max_bubble_size = 30, 1200
    if bubble_sizes_data.nunique() <= 1: scaled_bubble_sizes = pd.Series([100] * len(bubble_sizes_data), index=bubble_sizes_data.index)
    else:
        # 분모가 0이 되는 경우 방지
        denominator = bubble_sizes_data.max() - bubble_sizes_data.min()
        if denominator == 0: denominator = 1e-9 # 매우 작은 값으로 대체
        
        scaled_bubble_sizes = (bubble_sizes_data - bubble_sizes_data.min()) / denominator * (max_bubble_size - min_bubble_size) + min_bubble_size
        scaled_bubble_sizes[bubble_sizes_data == 0] = min_bubble_size / 2
    scatter_plot = ax.scatter(x='노후주택수', y='노인인구수', s=scaled_bubble_sizes, c=df_final_merged[f'{target_cause_for_bubble}건수'], cmap='OrRd', alpha=0.7, edgecolors='grey', linewidth=0.5, data=df_final_merged)
    top_n_districts = df_final_merged.sort_values(by=f'{target_cause_for_bubble}건수', ascending=False).head(7)
    for _, row_data in top_n_districts.iterrows(): ax.text(row_data['노후주택수'] * 1.01, row_data['노인인구수'] * 1.01, row_data['발생장소_구'], fontsize=9, color='black', ha='left', va='bottom')
    ax.set_title(f'노후 주택 수, 고령 인구 수와 {target_cause_for_bubble} 발생 건수', fontsize=16, pad=15)
    ax.set_xlabel('30년 이상 노후 주택 수', fontsize=12); ax.set_ylabel('고령 인구 수 (65세 이상)', fontsize=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}')); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    cbar = fig.colorbar(scatter_plot, ax=ax, label=f'{target_cause_for_bubble} 발생 건수'); cbar.ax.tick_params(labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.6); plt.tight_layout(); st.pyplot(fig)


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
        # 이미 각 preprocess 함수에서 st.warning/error를 표시하므로, 여기서는 추가 메시지 없이 반환
        return

    st.write("### 주거 안전사고 현황 (2023년 데이터 기준)") # 데이터 기준 연도 명시

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
            # 시간 형식 '%H:%M' 또는 '%H:%M:%S' 모두 시도
            df_rescue_time_analysis['신고시간_dt'] = pd.to_datetime(df_rescue_time_analysis['신고시각'], format='%H:%M', errors='coerce')
            if df_rescue_time_analysis['신고시간_dt'].isnull().sum() > len(df_rescue_time_analysis) * 0.8: # 대부분 변환 실패 시 다른 형식 시도
                df_rescue_time_analysis['신고시간_dt'] = pd.to_datetime(df_rescue_time_analysis['신고시각'], errors='coerce') # pandas가 자동 추론하도록 시도
            
            if '신고시간_dt' in df_rescue_time_analysis.columns and not df_rescue_time_analysis['신고시간_dt'].isnull().all():
                df_rescue_time_analysis['신고시간(시)'] = df_rescue_time_analysis['신고시간_dt'].dt.hour
                # NaN이나 잘못된 시간 데이터 제거 후 int로 변환
                hourly_incidents_counts = df_rescue_time_analysis.dropna(subset=['신고시간(시)'])
                hourly_incidents_counts = hourly_incidents_counts[pd.to_numeric(hourly_incidents_counts['신고시간(시)'], errors='coerce').notnull()]
                hourly_incidents_counts['신고시간(시)'] = hourly_incidents_counts['신고시간(시)'].astype(int)
                hourly_incidents_counts = hourly_incidents_counts['신고시간(시)'].value_counts().sort_index()
                
                if not hourly_incidents_counts.empty:
                    fig_time, ax_time = plt.subplots(figsize=(12, 6))
                    sns.lineplot(x=hourly_incidents_counts.index, y=hourly_incidents_counts.values, marker='o', color='indigo', ax=ax_time)
                    ax_time.set_title('시간대별 사고 발생 추이', fontsize=15); ax_time.set_xlabel('신고 시간 (0시 ~ 23시)', fontsize=12); ax_time.set_ylabel('사고 건수', fontsize=12)
                    ax_time.set_xticks(ticks=range(0, 24)); ax_time.grid(True, linestyle='--', alpha=0.7)
                    ax_time.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}')); plt.tight_layout(); st.pyplot(fig_time)
                else: st.info("시간대별 사고 발생 추이 분석을 위한 데이터가 부족합니다 (시간 변환 또는 집계 실패).")
            else: st.info("구조활동 데이터의 '신고시각'을 유효한 시간 형식으로 변환하는데 실패했습니다.")
        else: st.info("구조활동 데이터에 '신고시각' 컬럼이 없어 시간대별 분석을 수행할 수 없습니다.")

    # 5. 상관관계 분석 (순서 변경)
    with tabs[4]:
        st.subheader("노인 인구 및 노후 주택과 특정 사고 건수 간 상관관계")
        if '사고원인' in df_rescue_processed.columns:
            unique_causes_list_h_corr = sorted(df_rescue_processed['사고원인'].unique())
            default_idx_corr = unique_causes_list_h_corr.index('화재') if '화재' in unique_causes_list_h_corr else 0
            selected_cause_corr = st.selectbox("상관관계 분석 사고 원인:", unique_causes_list_h_corr, index=default_idx_corr, key="corr_cause_select_housing_main")
            if selected_cause_corr:
                cause_incidents_df = df_rescue_processed[df_rescue_processed['사고원인'] == selected_cause_corr]['발생장소_구'].value_counts().reset_index()
                cause_incidents_df.columns = ['발생장소_구', f'{selected_cause_corr}건수']

                if not df_elderly_processed_h.empty:
                    merged_df_corr_elderly = pd.merge(cause_incidents_df, df_elderly_processed_h, on='발생장소_구', how='inner')
                    if not merged_df_corr_elderly.empty:
                        plot_correlation_scatter_housing(merged_df_corr_elderly, '노인인구수', f'{selected_cause_corr}건수', f"노인 인구수와 {selected_cause_corr} 발생 건수")
                    else: st.info("노인 인구 데이터와 사고 건수 병합 결과 데이터가 없습니다.")
                else: st.info("노인 인구 데이터가 비어있어 상관관계 분석을 수행할 수 없습니다.")
                st.divider()
                if not df_housing_processed_h.empty:
                    merged_df_corr_housing = pd.merge(cause_incidents_df, df_housing_processed_h, on='발생장소_구', how='inner')
                    if not merged_df_corr_housing.empty:
                        plot_correlation_scatter_housing(merged_df_corr_housing, '노후주택수', f'{selected_cause_corr}건수', f"노후 주택수와 {selected_cause_corr} 발생 건수")
                    else: st.info("노후 주택 데이터와 사고 건수 병합 결과 데이터가 없습니다.")
                else: st.info("노후 주택 데이터가 비어있어 상관관계 분석을 수행할 수 없습니다.")
        else: st.info("구조활동 데이터에 '사고원인' 컬럼이 없어 상관관계 분석을 수행할 수 없습니다.")

    # 6. 사고와의 상관관계 분석 (순서 변경) - 버블 차트
    with tabs[5]:
        st.subheader("노후 주택, 고령 인구와 특정 사고 건수의 복합적 관계 (버블 차트)")
        unique_causes_list_h_bubble = sorted(df_rescue_processed['사고원인'].unique()) if '사고원인' in df_rescue_processed.columns else ['화재']
        default_idx_bubble = unique_causes_list_h_bubble.index('화재') if '화재' in unique_causes_list_h_bubble else 0
        cause_for_bubble = st.selectbox("버블 크기 기준 사고 원인:", unique_causes_list_h_bubble, index=default_idx_bubble, key="bubble_cause_select_housing_main")
        if cause_for_bubble:
            if not df_housing_processed_h.empty and not df_elderly_processed_h.empty:
                df_merged_bubble_step1 = pd.merge(df_housing_processed_h, df_elderly_processed_h, on='발생장소_구', how='inner')
                if not df_merged_bubble_step1.empty:
                    safety_accidents_for_bubble = df_rescue_processed[df_rescue_processed['사고원인'] == cause_for_bubble]['발생장소_구'].value_counts().reset_index()
                    safety_accidents_for_bubble.columns = ['발생장소_구', f'{cause_for_bubble}건수']
                    df_final_merged_for_bubble = pd.merge(df_merged_bubble_step1, safety_accidents_for_bubble, on='발생장소_구', how='left')
                    df_final_merged_for_bubble[f'{cause_for_bubble}건수'] = df_final_merged_for_bubble[f'{cause_for_bubble}건수'].fillna(0).astype(int)
                    if not df_final_merged_for_bubble.empty: 
                        plot_bubble_chart_housing(df_final_merged_for_bubble, cause_for_bubble)
                    else: st.info(f"'{cause_for_bubble}' 사고 기준 종합 분석용 최종 데이터가 비어있습니다.")
                else: st.info("노후 주택 데이터와 노인 인구 데이터 병합 결과가 비어있어 버블 차트 분석을 수행할 수 없습니다.")
            else: st.info("노후 주택 또는 노인 인구 데이터가 비어있어 버블 차트 분석을 수행할 수 없습니다.")

if __name__ == "__main__":
    run_housing_safety_page()
