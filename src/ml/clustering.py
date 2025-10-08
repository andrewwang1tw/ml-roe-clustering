import os
import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.db import load_data_from_csv


def run_analysis(params):    
    print("\n===============================================================================================================>")
    print("=======================================> [run_analysis], Start !!! ==============================================>")
    print("===============================================================================================================>\n\n")    
    print("\n\n======> [run_analysis], params ===>", params)
    
      
    #--- 1. 取資料 --------------------------------------------------------------------------------------------------    
    #engine = make_engine(params['server'], params['database'], trusted=params['trusted'])
    #df_raw = load_sales_data(engine, year=params['year'], month_from=params['month_from'])  
    
    selected_data_value = params['selected_data_value']
    df_raw = load_data_from_csv(selected_data_value)
    print("\n\n======> [run_analysis 1], df_raw ===>\n\n", df_raw)

    #--- 2. cleaned data: df_cleaned 和  去高度相關前的相關矩陣 : corr_before -----------------------------------------
    df_cleaned, corr_before = clean_and_correlate(df_raw)
    print("\n\n======> [run_analysis 2], cleaned data ===========>\n\n", df_cleaned.head())
    print("\n\n======> [run_analysis 2], 去高度相關前的相關矩陣 ===>\n\n", corr_before) 
           
    # 因之後會做 PCA, 所以不用 去高度相關, 還是用 df_cleaned      
    df_original = df_cleaned.copy()
           
    features_to_remove = highly_correlated_features(corr_before, threshold=0.9)
    print("\n\n======> [run_analysis 2], 高度相關特徵 ===>\n", 
          f"共 {len(features_to_remove)} 個高度相關特徵 (r >= 0.9)。", "高度相關特徵 ===> ", features_to_remove)
        
    df_filtered = df_cleaned.drop(columns=features_to_remove)   
    print("\n======> [run_analysis 2], df_filtered ===> \n", df_filtered)
    
    corr_after = df_filtered.corr()    
    print("\n\n======> [run_analysis 2], ===> ", f"原始維度: {corr_before.shape[1]}, 移除後維度: {corr_after.shape[1]}",
          "\n\n======> [run_analysis 2], 去高度相關後, 相關矩陣 ===>\n\n", corr_after) 
      
    # 因之後會做 PCA, 用沒有 去高度相關的
    df_filtered = df_original  
      
    #--- 3. 標準化 --------------------------------------------------------------------------------------------------  
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_filtered)
    print("\n======> [run_analysis 3], 原始特徵數量 =====================> ", df_filtered.shape[1],
          "\n======> [run_analysis 3], 標準化數據形狀 (樣本數, 特徵數) ===> ", X_scaled.shape,
          "\n======> [run_analysis 3], Z-Score 標準化, X_scaled ===>\n\n", X_scaled)
    
    #--- 4. PCA  ----------------------------------------------------------------------------------------------------
    # 執行 2D PCA, 繪圖用
    pca_2d = PCA(n_components=2)
    principal_components_2d = pca_2d.fit_transform(X_scaled)
    df_pca_2d = pd.DataFrame(data=principal_components_2d, columns=['PC1', 'PC2'], index=df_filtered.index)
    print("\n======> [run_analysis 4], df_pca_2d 累積解釋變異量 ====>", pca_2d.explained_variance_ratio_.sum())
    
    # 執行 3D PCA, 繪圖用
    pca_3d = PCA(n_components=3)
    principal_components_3d = pca_3d.fit_transform(X_scaled)
    df_pca_3d = pd.DataFrame(data=principal_components_3d, columns=['PC1', 'PC2', 'PC3'], index=df_filtered.index)    
    print(f"======> [run_analysis 4], df_pca_3d 累積解釋變異量: {pca_3d.explained_variance_ratio_.sum():.4f}")

    # 加 df_company_info 給 df_pca_2d, df_pca_3d, 繪圖用
    df_company_info = df_raw[['公司代號', '名稱']].loc[df_filtered.index]
    df_pca_2d = df_pca_2d.join(df_company_info[['名稱', '公司代號']]) 
    df_pca_3d = df_pca_3d.join(df_company_info[['名稱', '公司代號']]) 
    
    # --- 5. PCA 降維 和 Elbow Method 測試不同 k ---------------------------------------------------------------------
    # PCA 降維
    manual_pca = params['manual_pca'] 
    pca = PCA(n_components=manual_pca)
    X_pca = pca.fit_transform(X_scaled)
    print("\n\n\n======> [run_analysis 5], PCA 降維後形狀: ", X_pca.shape)
    print("======> [run_analysis 5], PCA 解釋變異量比例: ", pca.explained_variance_ratio_)
    print("======> [run_analysis 5], PCA 累積解釋變異量: ", pca.explained_variance_ratio_.cumsum())
    
    
    # K-Means 參數設置
    max_k = 15 # 測試 K 值從 1 到 15
    wcss = [] # 儲存每個 K 值下的群內平方和

    # 計算不同 K 值下的 WCSS
    for i in range(1, max_k + 1):
        # n_init=10：進行 10 次初始化以找到更優的結果
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X_pca)   
        wcss.append(kmeans.inertia_) # inertia_ 屬性即為 WCSS
    
    # --- 6. 執行 K-Means 分群  -------------------------------------------------------------------------------------- 
    manual_k = params['manual_k']   
    kmeans_final = KMeans(n_clusters=manual_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans_final.fit(X_pca)
    cluster_labels = kmeans_final.labels_
    # 也可以這樣做
    # cluster_labels = kmeans_final.fit_predict(X_pca) 
    print("\n\n\n======> [run_analysis 6] ===> kmeans_final\n", kmeans_final)
    print("\n======> [run_analysis 6] ===> cluster_labels\n", cluster_labels)

    # 分群結果  
    df_results = df_filtered.copy() 
    df_results['Cluster'] = cluster_labels  
    print("\n======> [run_analysis 6], 分群結果 ===>\n", df_results)
    
    # 將 分群結果 合併 PCA 2D/3D  
    df_pca_2d['Cluster'] = cluster_labels
    df_pca_3d['Cluster'] = cluster_labels

    # 將公司代號和名稱加入結果 DataFrame (用於後續解讀)
    df_results = df_company_info.merge(df_results, left_index=True, right_index=True)   
    df_results_cluster_counts = df_results['Cluster'].value_counts().sort_index()  
    print("\n======> [run_analysis 6], K-Means 分群完成 ", {manual_k}, "個群組")
    print("\n======> [run_analysis 6], 每群組數量 ===>\n", df_results_cluster_counts)
  
    # --- 7. 群集 特徵 -----------------------------------------------------------------------------------
    id_cols_to_exclude = ['公司代號', '名稱']
    df_numeric_results = df_results.drop(columns=id_cols_to_exclude)
    cluster_profile = df_numeric_results.groupby('Cluster').mean()

    # 4. 取得關鍵特徵的摘要 
    # key_features=['ROE','EPS','FCF','毛利率','盈餘成長率','總資產周轉率','現金殖利率','高登','PER','PBR','股價標準差','股價變異數']    
    key_features = df_results.columns.drop(id_cols_to_exclude).drop('Cluster').to_list()
    
    # 只存在 df_numeric_results 中的欄位
    key_features_filtered = [col for col in key_features if col in df_numeric_results.columns]
    cluster_summary = cluster_profile[key_features_filtered]
    
    # 加 cluster counts
    cluster_summary['counts'] = df_results_cluster_counts
    print("\n======> [run_analysis 7], 群組財務特徵平均值 ===>\n", cluster_summary)


    # 7. --- 最後存到 session_state ---------------------------------------------------------------------------
    st.session_state['df_raw'] = df_raw
    # corr before/after
    st.session_state['corr_before'] = corr_before
    st.session_state['corr_after'] = corr_after
    st.session_state['features_to_remove'] = features_to_remove
    st.session_state['df_filtered'] = df_filtered   
    # PCA before
    st.session_state['X_scaled'] = X_scaled
    st.session_state['df_pca_2d'] = df_pca_2d
    st.session_state['df_pca_3d'] = df_pca_3d
    st.session_state['pca_2d'] = pca_2d
    st.session_state['pca_3d'] = pca_3d
    # Elbow method
    st.session_state['wcss'] = wcss    
    #
    st.session_state['cluster_summary'] = cluster_summary    
    st.session_state['df_results'] = df_results   
    # 
     
    print("\n===============================================================================================================>")
    print("=======================================> [run_analysis], copmpleted !!! =======================================>")
    print("===============================================================================================================>\n\n\n")


#===================================================================
# 2. 資料清理 與 相關矩陣計算
#===================================================================
def clean_and_correlate(df):
    # 1. 隔離識別欄位
    id_cols = ['公司代號', '名稱', '產業別']           
    df_features = df.drop(columns=id_cols)

    # 2. 資料清理：移除任何包含 NaN 的列 (公司)
    original_rows = len(df_features)
    df_cleaned = df_features.dropna()
    cleaned_rows = len(df_cleaned)
      
    print(f"原始公司數量: {original_rows}")
    print(f"清理後公司數量: {cleaned_rows}")
    if original_rows != cleaned_rows:
        print(f"注意: 已移除 {original_rows - cleaned_rows} 筆包含缺失值的公司資料。")

    # 3. 計算相關矩陣
    correlation_matrix = df_cleaned.corr(numeric_only=True)    
    return df_cleaned, correlation_matrix


#===================================================================
# 2-1 highly_correlated_features
#===================================================================
def highly_correlated_features(corr_matrix, threshold=0.9):    
    # 使用 numpy.triu 確保只檢查上三角矩陣，避免重複檢查
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()

    # 迭代檢查欄位
    for column in upper.columns:
        # 找出與當前欄位高度相關的其他欄位
        high_corr_features = upper.index[abs(upper[column]) >= threshold].tolist()
        
        # 移除策略：保留 column (較早出現)，移除 high_corr_features 中的所有對應特徵
        for feature2 in high_corr_features:
            # 確保不會重複移除或移除已經被移除的特徵
            if feature2 not in to_drop:
                 to_drop.add(feature2)

    return to_drop
