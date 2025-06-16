    # ---!!! 중요 디버깅 코드 시작 !!!---
    print("DEBUG (Dokgo): df_gu_data.columns (first 30 columns after filtering for Seoul and Gu):")
    for i, col_tuple in enumerate(df_gu_data.columns):
        if i < 30 : 
            print(f"  Col {i}: {col_tuple} (Type: {type(col_tuple)})")
        else:
            break
    print("DEBUG (Dokgo): End of df_gu_data.columns print for filtered data")
    # ---!!! 중요 디버깅 코드 끝 !!!---
