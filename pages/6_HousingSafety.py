# --- START OF 6_HousingSafety.py ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import set_korean_font, load_csv 
import os
from matplotlib.ticker import PercentFormatter, FuncFormatter # FuncFormatter 추가

# --- 데이터 전처리 함수 (이전과 동일하게 유지) ---
@st.cache_data
def preprocess_rescue_data_cached(df_rescue_raw):
    if df_rescue_raw is None: 
        st.warning("구조활동현황 원본 데이터가 없습니다.")
        return pd.DataFrame()
    df_processed = df_rescue_raw.copy()
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
        new_columns = []
        for ct in df_elderly_processed.columns:
            if isinstance(ct, tuple):
                new_columns.append(tuple(str(s).replace('"', '').strip() if isinstance(s, str) else str(s) for s in ct))
            else:
                new_columns.append(str(ct).replace('"', '').strip() if isinstance(ct, str) else str(ct))
        
        if all(isinstance(col, tuple) for col in new_columns):
            if len(new_columns) == len(df_elderly_processed.columns):
                 df_elderly_processed.columns = pd.MultiIndex.from_tuples(new_columns)
            else:
                st.error("고령자현황: MultiIndex 컬럼명 생성 중 길이 불일치.")
                return pd.DataFrame()
        else:
            df_elderly_processed.columns = new_columns

        target_pop_col_tuple = None 
        total_pop_col_tuple = None  

        if isinstance(df_elderly_processed.columns, pd.MultiIndex) and len(df_elderly_processed.columns.levels) >= 4: 
            for col_tuple in df_elderly_processed.columns:
                if len(col_tuple) == 4: 
                    c0, c1, c2, c3 = (str(col_tuple[i]).strip() for i in range(4))
                    if c0.startswith('2023'): 
                        if '65세이상 인구' in c1 and c2 == '소계' and c3 == '소계':
                            target_pop_col_tuple = col_tuple
                        if '전체인구' in c1 and c2 == '소계' and c3 == '소계': 
                            total_pop_col_tuple = col_tuple 
            if not (target_pop_col_tuple and total_pop_col_tuple) :
                st.error("고령자현황: 2023년 65세 이상 인구 '소계' 또는 '전체인구' 컬럼을 자동 탐색하지 못했습니다.")
                return pd.DataFrame()
        else:
            found_target = False
            found_total = False
            for col_name_str in df_elderly_processed.columns:
                if isinstance(col_name_str, str): 
                    if '2023' in col_name_str and '65세이상 인구' in col_name_str and '소계' in col_name_str: 
                        target_pop_col_tuple = col_name_str 
                        found_target = True
                    if '2023' in col_name_str and '전체인구' in col_name_str and '소계' in col_name_str:
                        total_pop_col_tuple = col_name_str
                        found_total = True
            if not (found_target and found_total):
                st.warning("고령자현황: 예상된 컬럼 구조가 아니거나, 컬럼 구조가 다릅니다. '2023년 65세 이상 인구 소계' 또는 '2023년 전체인구 소계' 컬럼을 찾지 못했습니다.")
                return pd.DataFrame()

        if target_pop_col_tuple is None or total_pop_col_tuple is None:
             st.error("고령자현황: 2023년 65세 이상 인구 '소계' 또는 '전체인구' 컬럼을 찾지 못했습니다.")
             return pd.DataFrame()

        gu_info_col_name = None
        if isinstance(df_elderly_processed.columns, pd.MultiIndex):
            gu_info_col_name = df_elderly_processed.columns[1] 
        else: 
            if len(df_elderly_processed.columns) > 1:
                 gu_info_col_name = df_elderly_processed.columns[1] 
            else:
                st.error("고령자현황: 자치구 정보 컬럼이 부족합니다.")
                return pd.DataFrame()

        df_filtered_elderly = df_elderly_processed[df_elderly_processed[gu_info_col_name].astype(str).str.strip() != '소계'].copy()
        
        if gu_info_col_name in df_filtered_elderly.columns and \
           target_pop_col_tuple in df_filtered_elderly.columns and \
           total_pop_col_tuple in df_filtered_elderly.columns:
            
            df_final_elderly = df_filtered_elderly[[gu_info_col_name, target_pop_col_tuple, total_pop_col_tuple]].copy()
            df_final_elderly.columns = ['발생장소_구', '노인인구수', '전체인구수']
            df_final_elderly['발생장소_구'] = df_final_elderly['발생장소_구'].astype(str).str.strip()
            df_final_elderly['노인인구수'] = pd.to_numeric(df_final_elderly['노인인구수'].astype(str).str.replace(',','', regex=False), errors='coerce').fillna(0).astype(int)
            df_final_elderly['전체인구수'] = pd.to_numeric(df_final_elderly['전체인구수'].astype(str).str.replace(',','', regex=False), errors='coerce').fillna(0).astype(int)
            df_final_elderly.dropna(subset=['발생장소_구', '노인인구수', '전체인구수'], inplace=True)

            df_final_elderly['고령인구비율'] = np.where(
                df_final_elderly['전체인구수'] > 0,
                (df_final_elderly['노인인구수'] / df_final_elderly['전체인구수']),0
            ).astype(float)
            return df_final_elderly
        else:
            st.error("고령자현황: 필요한 컬럼(자치구, 노인인구수, 전체인구수)을 필터링된 데이터에서 찾을 수 없습니다.")
            return pd.DataFrame()
    except Exception as e: st.error(f"고령자현황 데이터 전처리 중 예외: {e}"); return pd.DataFrame()

@st.cache_data
def preprocess_housing_data_cached(df_housing_raw):
    if df_housing_raw is None: 
        st.warning("노후주택현황 원본 데이터가 없습니다.")
        return pd.DataFrame()
    df_housing_processed = df_housing_raw.copy()
    try:
        new_cols_h = []
        for ct_h in df_housing_processed.columns:
            if isinstance(ct_h, tuple) and len(ct_h) == 3:
                 new_cols_h.append(tuple(str(s).replace('"', '').strip() if isinstance(s, str) else str(s) for s in ct_h))
            elif isinstance(ct_h, tuple) and len(ct_h) < 3:
                new_cols_h.append(tuple(str(s).replace('"', '').strip() if isinstance(s, str) else str(s) for s in ct_h) + ('',) * (3 - len(ct_h)))
            else: 
                 new_cols_h.append((str(ct_h).replace('"', '').strip() if isinstance(ct_h, str) else str(ct_h), '', ''))

        if len(new_cols_h) == len(df_housing_processed.columns):
            df_housing_processed.columns = pd.MultiIndex.from_tuples(new_cols_h)
        else:
            st.error("노후주택: MultiIndex 컬럼명 생성 중 길이 불일치.")
            return pd.DataFrame()
        
        target_old_housing_col_tuple = None
        col_20_to_30_total_tuple = None     
        year_prefix_housing = '2023'      

        if isinstance(df_housing_processed.columns, pd.MultiIndex) and len(df_housing_processed.columns.levels) >= 3:
            for col_tuple_h in df_housing_processed.columns:
                if len(col_tuple_h) == 3:
                    c0_h, c1_h, c2_h = (str(col_tuple_h[i]).strip() for i in range(3))
                    if c0_h.startswith(year_prefix_housing):
                        if c1_h in ["30년이상", "30년 이상"] and c2_h == "계":
                            target_old_housing_col_tuple = col_tuple_h
                        elif ("20년이상" in c1_h and "30년미만" in c1_h and c2_h == "계") or \
                             ("20년~30년미만" in c1_h and c2_h == "계"): 
                            col_20_to_30_total_tuple = col_tuple_h
            
            if target_old_housing_col_tuple is None:
                 st.error(f"노후 주택 데이터: {year_prefix_housing}년 '30년 이상 계' 컬럼을 찾지 못했습니다.")
                 return pd.DataFrame()
            if col_20_to_30_total_tuple is None :
                 st.error(f"노후 주택 데이터: {year_prefix_housing}년 '20년~30년미만 계' 또는 '20년이상 30년미만 계' 컬럼을 찾지 못했습니다.")
                 return pd.DataFrame()
        else:
            found_target_housing = False
            found_20_30_housing = False
            for col_name_str_h in df_housing_processed.columns:
                if isinstance(col_name_str_h, str):
                    if year_prefix_housing in col_name_str_h and ("30년이상" in col_name_str_h or "30년 이상" in col_name_str_h) and "계" in col_name_str_h:
                        target_old_housing_col_tuple = col_name_str_h
                        found_target_housing = True
                    if year_prefix_housing in col_name_str_h and ("20년~30년미만" in col_name_str_h or ("20년이상" in col_name_str_h and "30년미만" in col_name_str_h)) and "계" in col_name_str_h:
                        col_20_to_30_total_tuple = col_name_str_h
                        found_20_30_housing = True
            if not (found_target_housing and found_20_30_housing):
                st.warning("노후주택: 예상된 컬럼 구조가 아니거나, 컬럼 구조가 다릅니다. 필요한 컬럼을 찾지 못했습니다.")
                return pd.DataFrame()

        gu_name_column_actual_name = None
        if isinstance(df_housing_processed.columns, pd.MultiIndex):
             gu_name_column_actual_name = df_housing_processed.columns[1]
        else: 
            if len(df_housing_processed.columns) > 1:
                 gu_name_column_actual_name = df_housing_processed.columns[1]
            else:
                st.error("노후 주택 데이터: 자치구 정보 컬럼이 부족합니다.")
                return pd.DataFrame()

        df_filtered_housing = df_housing_processed[
            df_housing_processed[gu_name_column_actual_name].astype(str).str.strip() != '소계'
        ].copy()
        
        data_to_extract = {
            '발생장소_구': df_filtered_housing[gu_name_column_actual_name].astype(str).str.strip(),
            '노후주택수': pd.to_numeric(df_filtered_housing[target_old_housing_col_tuple].astype(str).str.replace(',','', regex=False), errors='coerce').fillna(0).astype(int),
            '주택수_20년_30년미만': pd.to_numeric(df_filtered_housing[col_20_to_30_total_tuple].astype(str).str.replace(',','', regex=False), errors='coerce').fillna(0).astype(int)
        }
        df_final_housing = pd.DataFrame(data_to_extract)
        df_final_housing['전체주택수'] = df_final_housing['주택수_20년_30년미만'] + df_final_housing['노후주택수']
        df_final_housing['노후주택비율'] = np.where(
            df_final_housing['전체주택수'] > 0,
            (df_final_housing['노후주택수'] / df_final_housing['전체주택수']), 0
        ).astype(float)
        
        cols_to_return = ['발생장소_구', '노후주택수', '전체주택수', '노후주택비율']
        if not all(col in df_final_housing.columns for col in cols_to_return):
            st.error("노후주택: 최종 DataFrame에 필요한 컬럼 중 일부가 누락되었습니다.")
            return pd.DataFrame()

        df_final_housing = df_final_housing[cols_to_return]
        df_final_housing.dropna(subset=cols_to_return, inplace=True)
        return df_final_housing

    except KeyError as ke: 
        st.error(f"노후 주택 데이터 전처리 중 KeyError: '{ke}'. 컬럼명이 파일과 일치하는지 확인해주세요.")
        return pd.DataFrame()
    except Exception as e: 
        st.error(f"노후 주택 데이터 전처리 중 예외: {e}")
        return pd.DataFrame()


# --- 시각화 함수 ---
def plot_gu_incident_counts(df_rescue):
    if df_rescue.empty or '발생장소_구' not in df_rescue.columns: 
        st.info("구별 총 사고 발생 건수 데이터를 그릴 수 없습니다.")
        return

    df_rescue_filtered = df_rescue[df_rescue['발생장소_구'] != '서울특별시'].copy()
    if df_rescue_filtered.empty:
        st.info("자치구별 사고 건수 데이터가 없습니다 ('서울특별시' 제외 후).")
        return
        
    gu_incident_counts = df_rescue_filtered['발생장소_구'].value_counts().sort_values(ascending=False)
    
    if gu_incident_counts.empty:
        st.info("자치구별 사고 건수 집계 결과가 없습니다.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=gu_incident_counts.index, y=gu_incident_counts.values, color='steelblue', ax=ax, label='사고 건수')
    ax.set_title('서울시 구별 총 사고 발생 건수', fontsize=16)
    ax.set_xlabel('자치구', fontsize=12)
    ax.set_ylabel('사고 건수 (개)', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10); ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7); ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.legend(fontsize=10)
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
    ax.set_xlabel('자치구', fontsize=12)
    ax.set_ylabel('사고 건수 (개)', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10); ax.tick_params(axis='y', labelsize=10)
    ax.legend(title='사고원인', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.7); ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
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
    ax.set_title('주요 사고원인 비율', fontsize=16, pad=20)
    ax.axis('equal')
    ax.legend(top_causes_pie.index, title="사고원인", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    plt.tight_layout(rect=[0,0,0.85,1]); st.pyplot(fig)

def plot_correlation_scatter_ratio(merged_df, x_col_ratio, y_col_incident, title_text, x_label_text, selected_cause):
    if merged_df.empty or x_col_ratio not in merged_df.columns or y_col_incident not in merged_df.columns: st.info(f"'{title_text}' 산점도를 그릴 데이터가 없습니다."); return
    if not pd.api.types.is_numeric_dtype(merged_df[x_col_ratio]) or not pd.api.types.is_numeric_dtype(merged_df[y_col_incident]): st.warning(f"산점도용 컬럼('{x_col_ratio}', '{y_col_incident}') 중 숫자형이 아닌 것이 있습니다."); return
    
    correlation_label_x = x_label_text.replace(" (65세 이상)", "")

    try:
        correlation = merged_df[x_col_ratio].corr(merged_df[y_col_incident])
        st.write(f"**상관계수 ({correlation_label_x} vs {selected_cause}건수): {correlation:.3f}**")
    except Exception as e:
        st.warning(f"상관계수 계산 중 오류 발생: {e}")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=x_col_ratio, y=y_col_incident, data=merged_df, ax=ax, color='darkcyan',
                scatter_kws={'s':60, 'alpha':0.65, 'edgecolor':'black'}, 
                line_kws={'color':'red', 'linewidth':1.5}, 
                label=f'{selected_cause}건수의 합') 
    ax.set_title(title_text, fontsize=15)
    ax.set_xlabel(x_label_text, fontsize=12)
    ax.set_ylabel(f'{selected_cause}건수 (건)', fontsize=12) 
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)) 
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.legend(fontsize=10)
    plt.tight_layout(); st.pyplot(fig)

def plot_bubble_chart_ratio(df_final_merged, target_cause_for_bubble):
    required_cols = ['노후주택비율', '고령인구비율', f'{target_cause_for_bubble}건수', '발생장소_구']
    if df_final_merged.empty or not all(c in df_final_merged.columns for c in required_cols): st.info(f"버블 차트({target_cause_for_bubble})를 그릴 데이터가 부족합니다."); return
    fig, ax = plt.subplots(figsize=(12, 8))
    bubble_sizes_data = df_final_merged[f'{target_cause_for_bubble}건수']
    min_bubble_size, max_bubble_size = 30, 1200
    if bubble_sizes_data.nunique() <= 1: scaled_bubble_sizes = pd.Series([100] * len(bubble_sizes_data), index=bubble_sizes_data.index)
    else:
        denominator = bubble_sizes_data.max() - bubble_sizes_data.min()
        if denominator == 0: denominator = 1e-9 # 분모가 0이 되는 것 방지
        scaled_bubble_sizes = min_bubble_size + (bubble_sizes_data - bubble_sizes_data.min()) / denominator * (max_bubble_size - min_bubble_size)
        scaled_bubble_sizes[bubble_sizes_data == 0] = min_bubble_size / 2
    
    scatter_plot = ax.scatter(x='고령인구비율', y='노후주택비율', s=scaled_bubble_sizes, c=df_final_merged[f'{target_cause_for_bubble}건수'], cmap='YlOrRd', alpha=0.7, edgecolors='grey', linewidth=0.5, data=df_final_merged, label='자치구별 데이터')
    
    top_n_districts = df_final_merged.sort_values(by=f'{target_cause_for_bubble}건수', ascending=False).head(7)
    for _, row_data in top_n_districts.iterrows(): 
        ax.text(row_data['고령인구비율'] + 0.002, row_data['노후주택비율'] + 0.001, row_data['발생장소_구'], fontsize=9, color='black', ha='left', va='bottom')
    
    ax.set_title(f'고령 인구 비율, 노후 주택 비율과 {target_cause_for_bubble} 발생 건수', fontsize=16, pad=15)
    ax.set_xlabel('고령 인구 비율 (%)', fontsize=12) 
    ax.set_ylabel('30년 이상 노후 주택 비율 (%)', fontsize=12) 
    
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    cbar = fig.colorbar(scatter_plot, ax=ax, label=f'{target_cause_for_bubble} 발생 건수'); cbar.ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6); plt.tight_layout(); st.pyplot(fig)

def plot_heatmap_housing_elderly_incident_ratio(df_merged_ratio, accident_col_name, target_safety_accident_cause):
    if df_merged_ratio.empty or not all(col in df_merged_ratio.columns for col in ['노후주택비율', '고령인구비율', accident_col_name]):
        st.info("히트맵 생성에 필요한 데이터가 부족합니다.")
        return
    num_bins = 5
    try:
        df_merged_ratio_copy = df_merged_ratio.copy() 
        df_merged_ratio_copy['고령인구비율_bin_label'] = pd.cut(
            df_merged_ratio_copy['고령인구비율'], bins=num_bins, precision=2, include_lowest=True,
            labels=[f'{i*100/num_bins:.0f}-{(i+1)*100/num_bins:.0f}%' for i in range(num_bins)]
        )
        df_merged_ratio_copy['노후주택비율_bin_label'] = pd.cut(
            df_merged_ratio_copy['노후주택비율'], bins=num_bins, precision=2, include_lowest=True,
            labels=[f'{i*100/num_bins:.0f}-{(i+1)*100/num_bins:.0f}%' for i in range(num_bins)]
        )

        heatmap_data_ratio = df_merged_ratio_copy.pivot_table(
            values=accident_col_name,
            index='노후주택비율_bin_label', 
            columns='고령인구비율_bin_label', 
            aggfunc='mean',
            observed=True 
        ).sort_index(ascending=False)

        if not heatmap_data_ratio.empty:
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 8))
            sns.heatmap(heatmap_data_ratio, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=.5, 
                        cbar_kws={'label': f'평균 {target_safety_accident_cause} 건수'}, ax=ax_heatmap)
            ax_heatmap.set_title(f'고령 인구 비율 및 노후 주택 비율 구간별 평균 {target_safety_accident_cause} 발생 건수', fontsize=16, pad=15)
            ax_heatmap.set_xlabel('고령 인구 비율 구간 (%)', fontsize=12) 
            ax_heatmap.set_ylabel('30년 이상 노후 주택 비율 구간 (%)', fontsize=12) 
            plt.setp(ax_heatmap.get_xticklabels(), rotation=45, ha="right")
            plt.setp(ax_heatmap.get_yticklabels(), rotation=0)
            plt.tight_layout()
            st.pyplot(fig_heatmap)
        else:
            st.info("히트맵 생성을 위한 집계 데이터가 부족합니다 (구간화 후 데이터 없음).")
    except ValueError as ve:
        st.warning(f"히트맵 구간화 중 오류 발생: {ve}. 데이터 분포를 확인하거나 bin 수를 조정해주세요.")
    except Exception as ex:
        st.error(f"히트맵 생성 중 예외 발생: {ex}")

# 새로운 함수: 노후주택 및 고령인구 비율 비교 막대 그래프
def plot_housing_elderly_ratio_comparison(df_housing, df_elderly):
    if (df_housing is None or df_housing.empty or '노후주택비율' not in df_housing.columns or '발생장소_구' not in df_housing.columns) or \
       (df_elderly is None or df_elderly.empty or '고령인구비율' not in df_elderly.columns or '발생장소_구' not in df_elderly.columns):
        st.info("노후 주택 비율 또는 고령 인구 비율 데이터가 없어 비교 그래프를 생성할 수 없습니다.")
        return

    df_merged_ratios = pd.merge(
        df_housing[['발생장소_구', '노후주택비율']],
        df_elderly[['발생장소_구', '고령인구비율']],
        on='발생장소_구',
        how='inner'
    )
    df_merged_ratios = df_merged_ratios[df_merged_ratios['발생장소_구'] != '서울특별시']

    if df_merged_ratios.empty:
        st.info("병합된 노후 주택 및 고령 인구 비율 데이터가 없습니다.")
        return

    df_plot_melted = df_merged_ratios.melt(
        id_vars='발생장소_구',
        value_vars=['노후주택비율', '고령인구비율'],
        var_name='지표종류',
        value_name='비율'
    )
    df_plot_melted.rename(columns={'발생장소_구': '자치구'}, inplace=True)

    ordered_gus = df_merged_ratios.sort_values(by='노후주택비율', ascending=False)['발생장소_구'].tolist()

    fig, ax = plt.subplots(figsize=(18, 10))
    
    sns.barplot(x='자치구', y='비율', hue='지표종류', data=df_plot_melted, order=ordered_gus,
                palette={'노후주택비율':'coral', '고령인구비율':'skyblue'}, ax=ax)
    
    ax.set_title('서울시 구별 30년 이상 노후 주택 비율 및 고령 인구 비율 비교', fontsize=18, pad=20) # 제목 추가
    ax.set_xlabel('자치구', fontsize=14)
    ax.set_ylabel('비율 (%)', fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=10) # x축 레이블 회전 및 크기
    ax.tick_params(axis='y', labelsize=10) # y축 레이블 크기
    
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    if not df_plot_melted['비율'].empty: # Y축 범위 설정
        max_ratio = df_plot_melted['비율'].max()
        ax.set_ylim(0, max_ratio * 1.15 if max_ratio > 0 else 0.1) # 15% 여유분 또는 최소 0.1

    ax.legend(title='지표 종류', fontsize=11, title_fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout() # plt.tight_layout() 대신 fig 객체 사용
    st.pyplot(fig)


# --- Streamlit 페이지 레이아웃 ---
def run_housing_safety_page():
    st.title("주거 안전사고와 노후 주택 현황")
    set_korean_font()

    # 데이터 로드
    df_rescue_raw = load_csv("data/서울특별시_구조활동현황.csv")
    df_elderly_raw_h = load_csv("data/고령자현황_20250531210628.csv", header_config=[0,1,2,3])
    df_housing_raw_h = load_csv("data/노후기간별+주택현황_20250601054647.csv", header_config=[0,1,2])

    if df_rescue_raw is None or df_elderly_raw_h is None or df_housing_raw_h is None:
        st.error("필수 데이터 파일을 로드하지 못했습니다. 'data' 폴더 내용을 확인해주세요."); return

    # 데이터 전처리
    df_rescue_processed = preprocess_rescue_data_cached(df_rescue_raw)
    df_elderly_processed_h = preprocess_elderly_data_for_housing_cached(df_elderly_raw_h)
    df_housing_processed_h = preprocess_housing_data_cached(df_housing_raw_h)

    if df_rescue_processed is None or df_rescue_processed.empty : st.error("구조활동 데이터 처리 실패."); return
    if df_elderly_processed_h is None or df_elderly_processed_h.empty: st.error("고령자 현황 데이터 처리 실패."); return
    if df_housing_processed_h is None or df_housing_processed_h.empty: st.error("노후 주택 현황 데이터 처리 실패."); return

    # 탭 구성
    tab_titles_ordered = ["구별 사고 건수", "구별 사고 원인", "주요 사고 원인", "시간대별 사고", "노후주택 및 고령인구 비율", "상관관계 분석", "통합 상관관계 분석"]
    tabs = st.tabs(tab_titles_ordered)

    with tabs[0]: 
        st.subheader(tab_titles_ordered[0])
        plot_gu_incident_counts(df_rescue_processed)

    with tabs[1]: 
        st.subheader(tab_titles_ordered[1])
        top_n_causes_stacked = st.slider("표시할 상위 사고원인 개수:", 3, 15, 7, key="stacked_bar_top_n_slider_housing_main_v5")
        plot_stacked_bar_incident_causes_by_gu(df_rescue_processed, top_n_causes=top_n_causes_stacked)

    with tabs[2]: 
        st.subheader(tab_titles_ordered[2])
        top_n_causes_pie = st.slider("표시할 상위 사고원인 개수:", 3, 10, 7, key="pie_chart_top_n_slider_housing_main_v5")
        plot_pie_major_incident_causes(df_rescue_processed, top_n=top_n_causes_pie)

    with tabs[3]: 
        st.subheader(tab_titles_ordered[3])
        if '신고시각' in df_rescue_processed.columns:
            df_rescue_time_analysis = df_rescue_processed.copy()
            try:
                df_rescue_time_analysis['신고시간_dt'] = pd.to_datetime(df_rescue_time_analysis['신고시각'], errors='coerce')
            except Exception:
                 df_rescue_time_analysis['신고시간_dt'] = pd.NaT

            if '신고시간_dt' in df_rescue_time_analysis.columns and not df_rescue_time_analysis['신고시간_dt'].isnull().all():
                df_rescue_time_analysis['신고시간(시)'] = df_rescue_time_analysis['신고시간_dt'].dt.hour
                hourly_incidents_counts_df = df_rescue_time_analysis.dropna(subset=['신고시간(시)'])
                hourly_incidents_counts_df = hourly_incidents_counts_df[pd.to_numeric(hourly_incidents_counts_df['신고시간(시)'], errors='coerce').notnull()]
                if not hourly_incidents_counts_df.empty:
                    hourly_incidents_counts_df['신고시간(시)'] = hourly_incidents_counts_df['신고시간(시)'].astype(int)
                    hourly_incidents_final_counts = hourly_incidents_counts_df['신고시간(시)'].value_counts().sort_index()
                    if not hourly_incidents_final_counts.empty:
                        fig_time, ax_time = plt.subplots(figsize=(12, 6))
                        sns.lineplot(x=hourly_incidents_final_counts.index, y=hourly_incidents_final_counts.values, marker='o', color='indigo', ax=ax_time, label='사고 건수')
                        ax_time.set_title('시간대별 사고 발생 추이', fontsize=15)
                        ax_time.set_xlabel('신고 시간 (0시 ~ 23시)', fontsize=12)
                        ax_time.set_ylabel('사고 건수 (개)', fontsize=12)
                        ax_time.set_xticks(ticks=range(0, 24)); ax_time.grid(True, linestyle='--', alpha=0.7)
                        ax_time.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
                        ax_time.legend(fontsize=10)
                        plt.tight_layout(); st.pyplot(fig_time)
                    else: st.info("시간대별 사고 발생 추이 분석을 위한 데이터가 부족합니다 (집계 후 데이터 없음).")
                else: st.info("시간대별 사고 발생 추이 분석을 위한 데이터가 부족합니다 (시간 변환 또는 유효 데이터 부족).")
            else: st.info("구조활동 데이터의 '신고시각'을 유효한 시간 형식으로 변환하는데 실패했습니다.")
        else: st.info("구조활동 데이터에 '신고시각' 컬럼이 없어 시간대별 분석을 수행할 수 없습니다.")
    
    with tabs[4]: 
        st.subheader(tab_titles_ordered[4])
        plot_housing_elderly_ratio_comparison(df_housing_processed_h, df_elderly_processed_h) # 수정된 함수 호출

    with tabs[5]: 
        st.subheader(tab_titles_ordered[5])
        if '사고원인' in df_rescue_processed.columns and \
           not df_elderly_processed_h.empty and '고령인구비율' in df_elderly_processed_h.columns and \
           not df_housing_processed_h.empty and '노후주택비율' in df_housing_processed_h.columns:
            
            unique_causes_list_h_corr_ratio = sorted(df_rescue_processed['사고원인'].unique())
            default_idx_corr_ratio = unique_causes_list_h_corr_ratio.index('화재') if '화재' in unique_causes_list_h_corr_ratio else 0
            selected_cause_corr_ratio = st.selectbox(
                "상관관계 분석 사고 원인:", 
                unique_causes_list_h_corr_ratio, 
                index=default_idx_corr_ratio, 
                key="corr_ratio_cause_select_housing_v5"
            )
            
            if selected_cause_corr_ratio:
                cause_incidents_df_ratio = df_rescue_processed[df_rescue_processed['사고원인'] == selected_cause_corr_ratio]['발생장소_구'].value_counts().reset_index()
                cause_incidents_df_ratio.columns = ['발생장소_구', f'{selected_cause_corr_ratio}건수']

                merged_df_corr_elderly_ratio = pd.merge(cause_incidents_df_ratio, df_elderly_processed_h[['발생장소_구', '고령인구비율']], on='발생장소_구', how='inner')
                if not merged_df_corr_elderly_ratio.empty:
                    plot_correlation_scatter_ratio(
                        merged_df_corr_elderly_ratio, 
                        '고령인구비율', 
                        f'{selected_cause_corr_ratio}건수', 
                        f"고령 인구 비율과 {selected_cause_corr_ratio} 발생 건수", 
                        "고령 인구 비율", 
                        selected_cause_corr_ratio
                    )
                else: st.info("고령 인구 비율 데이터와 사고 건수 병합 결과 데이터가 없습니다.")
                
                merged_df_corr_housing_ratio = pd.merge(cause_incidents_df_ratio, df_housing_processed_h[['발생장소_구', '노후주택비율']], on='발생장소_구', how='inner')
                if not merged_df_corr_housing_ratio.empty:
                    plot_correlation_scatter_ratio(
                        merged_df_corr_housing_ratio, 
                        '노후주택비율', 
                        f'{selected_cause_corr_ratio}건수', 
                        f"노후 주택 비율과 {selected_cause_corr_ratio} 발생 건수", 
                        "30년 이상 노후 주택 비율",
                        selected_cause_corr_ratio
                    )
                else: st.info("노후 주택 비율 데이터와 사고 건수 병합 결과 데이터가 없습니다.")
        else: 
            missing_info = []
            if '사고원인' not in df_rescue_processed.columns: missing_info.append("'사고원인' 컬럼")
            if df_elderly_processed_h.empty or '고령인구비율' not in df_elderly_processed_h.columns: missing_info.append("'고령인구비율' 컬럼")
            if df_housing_processed_h.empty or '노후주택비율' not in df_housing_processed_h.columns: missing_info.append("'노후주택비율' 컬럼")
            st.info(f"상관관계 분석을 수행하기 위한 데이터({', '.join(missing_info)})가 부족합니다.")


    with tabs[6]: 
        st.subheader(tab_titles_ordered[6])
        unique_causes_list_h_bubble_ratio = sorted(df_rescue_processed['사고원인'].unique()) if '사고원인' in df_rescue_processed.columns else ['화재']
        default_idx_bubble_ratio = unique_causes_list_h_bubble_ratio.index('화재') if '화재' in unique_causes_list_h_bubble_ratio else 0
        cause_for_bubble_ratio = st.selectbox(
            "버블/히트맵 기준 사고 원인:",
            unique_causes_list_h_bubble_ratio, 
            index=default_idx_bubble_ratio, 
            key="bubble_ratio_cause_select_housing_v5"
        )
        
        if cause_for_bubble_ratio:
            if not df_housing_processed_h.empty and '노후주택비율' in df_housing_processed_h.columns and \
               not df_elderly_processed_h.empty and '고령인구비율' in df_elderly_processed_h.columns:
                
                df_merged_bubble_ratio_step1 = pd.merge(
                    df_housing_processed_h[['발생장소_구', '노후주택비율']], 
                    df_elderly_processed_h[['발생장소_구', '고령인구비율']], 
                    on='발생장소_구', how='inner'
                )
                
                if not df_merged_bubble_ratio_step1.empty:
                    safety_accidents_for_bubble_ratio = df_rescue_processed[df_rescue_processed['사고원인'] == cause_for_bubble_ratio]['발생장소_구'].value_counts().reset_index()
                    safety_accidents_for_bubble_ratio.columns = ['발생장소_구', f'{cause_for_bubble_ratio}건수']
                    
                    df_final_merged_for_bubble_ratio = pd.merge(df_merged_bubble_ratio_step1, safety_accidents_for_bubble_ratio, on='발생장소_구', how='left')
                    df_final_merged_for_bubble_ratio[f'{cause_for_bubble_ratio}건수'] = df_final_merged_for_bubble_ratio[f'{cause_for_bubble_ratio}건수'].fillna(0).astype(int)
                    
                    if not df_final_merged_for_bubble_ratio.empty: 
                        st.markdown("#### 버블 차트")
                        plot_bubble_chart_ratio(df_final_merged_for_bubble_ratio, cause_for_bubble_ratio)
                        st.divider()
                        st.markdown("#### 히트맵")
                        plot_heatmap_housing_elderly_incident_ratio(df_final_merged_for_bubble_ratio, f'{cause_for_bubble_ratio}건수', cause_for_bubble_ratio)
                    else: st.info(f"'{cause_for_bubble_ratio}' 사고 기준 종합 분석용 최종 데이터가 비어있습니다.")
                else: st.info("노후 주택 비율과 고령 인구 비율 데이터 병합 결과가 비어있어 분석을 수행할 수 없습니다.")
            else: 
                missing_info_bubble = []
                if df_housing_processed_h.empty or '노후주택비율' not in df_housing_processed_h.columns: missing_info_bubble.append("노후주택비율")
                if df_elderly_processed_h.empty or '고령인구비율' not in df_elderly_processed_h.columns: missing_info_bubble.append("고령인구비율")
                st.info(f"다음 데이터가 준비되지 않아 버블 차트/히트맵 분석을 수행할 수 없습니다: {', '.join(missing_info_bubble)}")

if __name__ == "__main__":
    run_housing_safety_page()
# --- END OF 6_HousingSafety.py ---
