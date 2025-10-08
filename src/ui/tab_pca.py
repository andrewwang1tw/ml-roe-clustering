import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from typing import List, Optional
from ui.ui_main import llm_chat
      
#-----------------------------------------------------------------------------------
# Tab 2, PCA before
#-----------------------------------------------------------------------------------
def tab2_show_pca_before(X_scaled, columns):  
    st.markdown("""**主成分分析 (Principal Component Analysis, PCA)** """)
    
    # 繪圖時, 不降維, 顯示全部主成分的累積解釋變異量
    #N_COMPONENTS = 8
    #pca = PCA(n_components=N_COMPONENTS)    
    pca_full = PCA()
    pca_full.fit(X_scaled)
        
    # 計算累積解釋變異量, 找出保留 80% 變異量所需的主成分數量
    cumulative_variance_ratio = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_80 = np.where(cumulative_variance_ratio >= 0.8)[0][0] + 1
    print(f"\nTab2, 保留 80% 總變異量所需的主成分數量: {n_components_80} 個 \n")

    # 將結果轉換為 Plotly 易於處理的 DataFrame
    variance_df = pd.DataFrame({
        '主成分數量': range(1, len(cumulative_variance_ratio) + 1),
        '累積解釋變異量比例': cumulative_variance_ratio
    })

    # --- 使用 Plotly Express 繪製圖表 ---
    fig = px.line(
        variance_df,
        x='主成分數量',
        y='累積解釋變異量比例',
        title='<b>累積解釋變異量 (Cumulative Explained Variance)</b>',
        markers=True, 
        template='plotly_white'
    )
    # 1. 繪製 80% 門檻的水平參考線 (紅線)
    fig.add_hline(
        y=0.8, 
        line_dash="dash", 
        line_color="red", 
        annotation_text="80% Threshold",
        annotation_position="top left",
        annotation_font_size=10
    )
    # 2. 繪製對應主成分數量的垂直參考線 (綠線)
    fig.add_vline(
        x=n_components_80, 
        line_dash="dot", 
        line_color="green",
        annotation_text=f"{n_components_80} Components",
        annotation_position="bottom right",
        annotation_font_size=10
    )
    # 3. 調整圖表樣式 (邊框、網格、大小)
    fig.update_xaxes(
        tickmode='linear',
        dtick=1, # X 軸間隔為 1
        # 邊框線
        showline=True, 
        linewidth=1.5, 
        linecolor='black',
        # 網格線
        showgrid=True, 
        gridcolor='lightgray', 
        #griddash='dot'
    )
    fig.update_yaxes(
        tickformat=".0%", # Y 軸顯示為百分比格式 (0.8 會顯示為 80%)
        # 邊框線
        showline=True, 
        linewidth=1.5, 
        linecolor='black',
        # 網格線
        showgrid=True, 
        gridcolor='lightgray', 
        #griddash='dot'
    )
    # 4. 設定圖表大小和字體
    fig.update_layout(
        width=800,
        height=500,
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor='rgba(245,245,245,1)', 
        #font=dict(family="Arial, Noto Sans CJK TC, sans-serif", size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 顯示圖表, fig.show()
    st.plotly_chart(fig, width='stretch', key="key_tab2_show_pca_before")
    
    #--- 各主成分變異比例 ---
    st.write("**各主成分變異比例 (Explained Variance Ratio)**") 
    # df = pd.DataFrame({
    #     "PC1": [pca_full.explained_variance_ratio_[0]], 
    #     "PC2": [pca_full.explained_variance_ratio_[1]], 
    #     "PC3": [pca_full.explained_variance_ratio_[2]], 
    #     "PC4": [pca_full.explained_variance_ratio_[3]], 
    #     "PC5": [pca_full.explained_variance_ratio_[4]],  
    #     "PC6": [pca_full.explained_variance_ratio_[5]], 
    #     "PC7": [pca_full.explained_variance_ratio_[6]], 
    #     "PC8": [pca_full.explained_variance_ratio_[7]], 
    #     "PC9": [pca_full.explained_variance_ratio_[8]], 
    #     "PC10": [pca_full.explained_variance_ratio_[9]], 
    #     "PC11": [pca_full.explained_variance_ratio_[10]],  
    #     "PC12": [pca_full.explained_variance_ratio_[11]], 
    #     "PC13": [pca_full.explained_variance_ratio_[12]], 
    #     "PC14": [pca_full.explained_variance_ratio_[13]], 
    #     "PC15": [pca_full.explained_variance_ratio_[14]], 
    #     "PC16": [pca_full.explained_variance_ratio_[15]],        
    #     "總解釋比例": [sum(pca_full.explained_variance_ratio_[:16])]
    # })    
    
    n_components = len(pca_full.explained_variance_ratio_)
    pc_names = [f'PC{i+1}' for i in range(n_components)]     
    data = {
        pc_names[i]: [pca_full.explained_variance_ratio_[i]] for i in range(n_components)       
    }
    
    data['總解釋比例'] = [sum(pca_full.explained_variance_ratio_[:n_components])]   
    df = pd.DataFrame(data)  
    
    styled_df = df.style.format("{:.4f}")
    st.dataframe(styled_df, hide_index=True, width='stretch')
    
    #--- 主成分組成 ---     
    #comp_df = pd.DataFrame(pca_full.components_, columns=columns, index=["PC1", "PC2", "PC3","PC4", "PC5", "PC6","PC7", "PC8", "PC9", "PC10","PC11", "PC12", "PC13","PC14", "PC15", "PC16"])
    st.write("**主成分組成 (Components)**")
    pc_indexes = [f"PC{i+1}" for i in range(n_components)] 
    comp_df = pd.DataFrame(
        pca_full.components_, 
        columns=columns, # 這是原始特徵欄位名稱
        index=pc_indexes  # 使用動態生成的索引
    )
    
    # 熱度背景格式化       
    #excluded_cols = ['股價標準差', '股價變異數', 'Cluster', 'counts']
    gradient_subset = [col for col in comp_df.columns]
         
    styled_df = comp_df.style.background_gradient(
        cmap="RdYlGn",  # 雙向顏色 (紅→黃→綠)
        axis=None,         # 0(Row), 1(column), None 整張表統一顏色比例
        subset=gradient_subset,
        vmin=-comp_df[gradient_subset].abs().max().max(),  # 負到正對稱
        vmax=comp_df[gradient_subset].abs().max().max()
    ).format("{:.3f}")
        
    #st.dataframe(comp_df.style.format("{:.3f}"), width='stretch')    
    st.dataframe(styled_df, width='stretch')
    print("\nTab2, comp_df ===>\n", comp_df)
      
    #--- AI  ---     
    #And {cluster_summary} is the cluster summary associated with the PCA result.
    #cluster_summary = st.session_state['cluster_summary'] 
    
    input = f"Please explain the PCA result: {df} and {comp_df} and answer in Traditional Chinese."            
     
    if st.button("AI 解釋 主成分分析 (PCA)", key="ai_explain_pca"):       
        #llm_chat(input) 
        with st.status("📂 AI 解釋 主成分分析 (PCA) 中...", expanded=True) as status:  
            from agent.langGraph_gemini import stream_llm_updates     
            response = stream_llm_updates(input)     
            if response:                                  
                st.markdown(response)             
                status.update(label="📂 AI 解釋 主成分分析 (PCA)", state="complete", expanded=True)  
                
            status.update(label="📂 AI 解釋 主成分分析 (PCA)", state="complete", expanded=True)        
   

     

#--------------------------------------------------------------------------------------------
# tab2_show_pca_2d_before
#--------------------------------------------------------------------------------------------    
def tab2_show_pca_2d_before(df_pca_2d, pca_2d, columns):
    fig_2d = px.scatter(
        df_pca_2d,
        x='PC1',
        y='PC2',
        hover_name='名稱', # 滑鼠懸停時顯示公司名稱
        title='<b>2D PCA 視覺化 (PC1 vs PC2)</b>',
        labels={
            'PC1': f'主成分 1 (PC1, 變異量: {pca_2d.explained_variance_ratio_[0]*100:.2f}%)',
            'PC2': f'主成分 2 (PC2, 變異量: {pca_2d.explained_variance_ratio_[1]*100:.2f}%)'
        },
        template='plotly_white',
        width=800,
        height=700
    )

    # 加入公司名稱標籤 (使用 add_trace 或 add_annotations，Plotly 通常傾向 hover_name)  
    for i in range(min(100, len(df_pca_2d))):
        fig_2d.add_annotation(
            x=df_pca_2d['PC1'].iloc[i],
            y=df_pca_2d['PC2'].iloc[i],
            text=df_pca_2d['名稱'].iloc[i],
            showarrow=True,
            arrowhead=1,
            ax=5, # 水平偏移
            ay=-10, # 垂直偏移
            font=dict(size=9, color='black'),
            bgcolor="rgba(255, 255, 255, 0.7)", # 讓標籤背景半透明，不遮擋點
            bordercolor="black",
            borderwidth=0.5,
            borderpad=2
        )

    # 調整軸線、網格和邊框
    fig_2d.update_xaxes(
        showline=True, linewidth=1.5, linecolor='black',
        showgrid=True, gridcolor='lightgray', griddash='dot'
    )
    fig_2d.update_yaxes(
        showline=True, linewidth=1.5, linecolor='black',
        showgrid=True, gridcolor='lightgray', griddash='dot'
    )
    fig_2d.update_layout(
        #font=dict(family="Arial, Noto Sans CJK TC, sans-serif", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor='rgba(245,245,245,1)', 
        hovermode="closest" # 讓 hover 效果更精確
    )

    #fig_2d.show() 
    st.plotly_chart(fig_2d, width='stretch', key="key_tab2_show_pca_2d_before")   
    
    #st.write("**主成分組成 (Components)**")
    comp_df = pd.DataFrame(pca_2d.components_, columns=columns, index=["PC1", "PC2"])
    st.dataframe(comp_df.style.format("{:.3f}"), width='stretch')
 
    
    
#--------------------------------------------------------------------------------------------
# tab2_show_pca_3d_before
#--------------------------------------------------------------------------------------------    
def tab2_show_pca_3d_before(df_pca_3d, pca_3d, columns):    
    import plotly.graph_objects as go # 用於 3D 圖標籤

    # --- Plotly 3D PCA 視覺化 ----------------------------------------------------------------------------------
    fig_3d = px.scatter_3d(
        df_pca_3d,
        x='PC1',
        y='PC2',
        z='PC3',
        hover_name='名稱', # 滑鼠懸停時顯示公司名稱
        title='<b>3D PCA 視覺化</b>',
        labels={
            'PC1': f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.2f}%)',
            'PC2': f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.2f}%)',
            'PC3': f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.2f}%)'
        },
        template='plotly_white',
        width=800, # 3D 圖通常可以設置更寬一些
        height=700,
    )

    fig_3d.update_traces(marker=dict(size=2)) # 將 size 調整為你認為合適的數值

    # 在 3D 圖中加入公司名稱標籤 (使用 go.Scatter3d 並設定 text)
    # 這種方法會將標籤視為資料點的一部分，可以在 hover 時顯示。    
    trace_annotations = []
    for i in range(min(10, len(df_pca_3d))):
        trace_annotations.append(
            go.Scatter3d(
                x=[df_pca_3d['PC1'].iloc[i]],
                y=[df_pca_3d['PC2'].iloc[i]],
                z=[df_pca_3d['PC3'].iloc[i]],
                mode='text', # 顯示文字
                text=[df_pca_3d['名稱'].iloc[i]],
                textfont=dict(size=9, color='black'),
                showlegend=False,
                hoverinfo='none', # 避免重複 hover 資訊
            )
        )
    # 將這些文字標籤加入到圖中
    for trace in trace_annotations:
        fig_3d.add_trace(trace)

    # 調整 3D 軸線、網格和邊框
    fig_3d.update_layout(scene=dict(
        xaxis=dict(
            showline=True, linewidth=1.5, linecolor='black',
            showgrid=True, gridcolor='lightgray', 
            #griddash='dot',
            title=f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.2f}%)' # 因為 px.scatter_3d 的 labels 參數只影響 hover
        ),
        yaxis=dict(
            showline=True, 
            linewidth=1.5, 
            linecolor='black',
            showgrid=True, 
            gridcolor='lightgray', 
            #griddash='dot',
            title=f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.2f}%)'
        ),
        zaxis=dict(
            showline=True, 
            linewidth=1.5, 
            linecolor='black',
            showgrid=True, 
            gridcolor='lightgray', 
            #griddash='dot',
            title=f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.2f}%)'
        ),
        aspectmode='cube' # 讓 XYZ 軸比例相等，避免變形
    ))
    # 字體設定
    fig_3d.update_layout(
        #font=dict(family="Arial, Noto Sans CJK TC, sans-serif", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor='rgba(245,245,245,1)',
        hovermode="closest"
    )
    #fig_3d.show()
    st.plotly_chart(fig_3d, width='stretch', key="key_tab2_show_pca_3d_before")
    
    #st.write("**主成分組成 (Components)**")
    comp_df = pd.DataFrame(pca_3d.components_, columns=columns, index=["PC1", "PC2", "PC3"])
    st.dataframe(comp_df.style.format("{:.3f}"), width='stretch')  