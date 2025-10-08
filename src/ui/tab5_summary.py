import io
import json
import streamlit as st
import pandas as pd
from agent.langGraph_gemini import stream_graph_updates2, stream_graph_updates
from ui.ui_main import get_cluster_labels


#-----------------------------------------------------------------------------------
# Tab 5, cluster summary
#-----------------------------------------------------------------------------------
def tab5_show_cluster_summary(cluster_summary, df_results):
    st.markdown("**各群集特徵(K-Means 平均值)**")
    
    all_cols = cluster_summary.columns.tolist()
    format_mapping = {}

    # 為所有財務特徵設置 .1f 格式
    # for col in all_cols:
    #     if col == 'counts':
    #         format_mapping[col] = "{:.0f}" # 欄位 'counts' 顯示整數
    #     else:
    #         # 其他所有欄位（如 ROE, EPS, FCF, PBR 等）顯示2位小數
    #         format_mapping[col] = "{:.2f}" 
    
    format_mapping = {
        # 獲利/比率 (建議 4 位小數，確保精確度)
        'ROE': '{:.1%}',
        '毛利率': '{:.1%}',
        '純益率': '{:.1%}',
        '營業利益率': '{:.1%}',
        '現金殖利率': '{:.1%}',
        '高登': '{:.1%}',
        '總資產周轉率': '{:.1%}',        
        # 成長率/較大比率
        '盈餘成長率': '{:.1%}',
        '營收年成長率': '{:.1%}',        
        # 金額/較大數值 (建議 2 位小數)
        'EPS': '{:.2f}',
        'FCF': '{:.2f}',
        '每股淨值': '{:.2f}',
        '權益乘數': '{:.1%}',    
        '股本億元': '{:,.0f}',  
        # 倍數 (建議 2 位小數)
        'PBR': '{:.1f}',
        'PER': '{:.1f}',        
        # 波動性 (數值較大，建議 2 位小數)
        '股價標準差': '{:.1f}',
        '股價變異數': '{:.1f}',
        '平均價': '{:.1f}',
        'counts': '{:d}'
    } 
    
    new_order = ['ROE','毛利率','營業利益率','純益率', '總資產周轉率', '權益乘數','盈餘成長率','高登','營收年成長率','現金殖利率','EPS']
    cluster_summary = cluster_summary.reindex(columns=new_order + [col for col in cluster_summary.columns if col not in new_order])
        
    # colors
    excluded_cols = ['股價標準差', '股價變異數', 'Cluster', 'counts']
    gradient_subset = [col for col in all_cols if col not in excluded_cols]
        
    # 熱度背景格式化        
    styled_df = cluster_summary.style.background_gradient(
        cmap="RdYlGn",  # 雙向顏色 (紅→黃→綠)
        axis=1,         # 0(Row), 1(column), None 整張表統一顏色比例
        #subset=gradient_subset,
        vmin=-cluster_summary[gradient_subset].abs().max().max(),  # 負到正對稱
        vmax=cluster_summary[gradient_subset].abs().max().max()
    ).format(format_mapping)
    
    st.dataframe(styled_df, width='stretch')
                
    # ----------------------------------------------------------------------
    # Radar 圖,  計算各群組數目並準備新的標籤
    # ----------------------------------------------------------------------

    # --- 步驟 1. 計算每個群組的樣本數 (N) ---
    cluster_counts = df_results['Cluster'].value_counts().sort_index()

    # 創建一個映射字典，例如 {0: 'Cluster 0 (N=450)', 1: 'Cluster 1 (N=300)', ...}
    cluster_name_map = {}
    for cluster_id, count in cluster_counts.items():
        original_name = f'Cluster {cluster_id}'
        new_name = f'Cluster {cluster_id} (股票數目={count})'
        cluster_name_map[original_name] = new_name
        
    print("\nCluster Summary, Tab 5 --- 群組數目分析 ---", cluster_counts)
    print("\nCluster Summary, Tab 5 --- 新的群組標籤映射 ---", cluster_name_map)

 
    # --- 步驟 2: 數據標準化和轉換 (保持不變) ---    
    summary = cluster_summary.copy()
    #df_radar = summary.T.copy()
    df_radar = summary.drop(columns='counts', axis=1).T.copy()     
    df_radar.columns = [f'Cluster {i}' for i in range(len(df_radar.columns))]

    inverting_features = ['PER', 'PBR', '股價標準差', '股價變異數'] 
    normalized_df = df_radar.copy()
    features = df_radar.index.tolist()

    for feature in features:
        data_series = df_radar.loc[feature]
        min_val = data_series.min()
        max_val = data_series.max()
        
        if (max_val - min_val) > 0:
            normalized_values = (data_series - min_val) / (max_val - min_val)
        else:
            # 異常情況：分母為零
            normalized_values = pd.Series(0.5, index=df_radar.columns)
            
        if feature in inverting_features:
            # PER (本益比)、PBR (股價淨值比)、股價標準差/變異數 數值越高反而代表「風險越高」或「估值越貴」(通常被視為較差的傾向)。
            normalized_df.loc[feature] = 1 - normalized_values
        else:
            # ROE、EPS、FCF 這樣的獲利指標，數值越高代表公司越好
            normalized_df.loc[feature] = normalized_values

    # 重命名 df_normalized_wide 中的 'Cluster' 欄位值
    df_normalized_wide = normalized_df.T.reset_index().rename(columns={'index': 'Cluster'})

    # 應用新的群組標籤
    df_normalized_wide['Cluster'] = df_normalized_wide['Cluster'].map(cluster_name_map)

    df_long = df_normalized_wide.melt(id_vars='Cluster', var_name='Feature', value_name='Score')



    # 步驟 3: 使用 Plotly 繪製交互式雷達圖 (增強顏色對比)
    #cjk_font_family = "Noto Sans CJK JP, Microsoft JhengHei, Arial Unicode MS, sans-serif"
    import plotly.express as px
    fig = px.line_polar(
        df_long, 
        r='Score', 
        theta='Feature', 
        color='Cluster', 
        line_close=True, 
        title='各群集財務體質雷達圖：標準化比較', 
        color_discrete_sequence=px.colors.qualitative.Light24_r 
    )
    # 調整線條和填色
    fig.update_traces(
        fill='toself', 
        opacity=1, # 稍微調低不透明度，讓圖表更清晰
        line=dict(width=1) 
    )
    # 調整佈局
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["20%", "40%", "60%", "80%", "100%"],
                #tickfont_family=cjk_font_family 
            ),
            #angularaxis=dict(#tickfont_family=cjk_font_family)
        ),
        # title_font_family=cjk_font_family,
        # legend_font_family=cjk_font_family,
        # font_family=cjk_font_family
    )

    #fig.show()        
    st.plotly_chart(fig, width='stretch', key="key_tab5_show_radar")   
    
    #----------------------------------------------------------------
    # AI 解釋 cluster summary
    # ---------------------------------------------------------------
    input = f"""
                Please explain the cluster summary: {cluster_summary}.
                And answer in Traditional Chinese.
            """            
    if st.button("AI 各群集特徵總結", key="ai_cluster_summary"): 
        with st.status("📂 AI 解釋 cluster summary  中...", expanded=True) as status:  
            from agent.langGraph_gemini import stream_llm_updates     
            response = stream_llm_updates(input)     
            if response:                                  
                st.markdown(response)             
                status.update(label="📂 AI 解釋 cluster summary ", state="complete", expanded=True)  
                
            status.update(label="📂 AI 解釋 cluster summary ", state="complete", expanded=True) 