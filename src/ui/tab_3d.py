import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from ui.ui_main import llm_chat
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

    
#--------------------------------------------------------------------------------------------
# tab4_show_pca_3d
#--------------------------------------------------------------------------------------------    
def tab4_show_pca_3d(df_pca_3d, pca_3d, columns, manual_k):    
    import plotly.graph_objects as go # 用於 3D 圖標籤

    # --- Plotly 3D PCA 視覺化 ---
    fig_3d = px.scatter_3d(
        df_pca_3d,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Cluster',  # 使用 Cluster 欄位來決定顏色
        hover_name='名稱', # 滑鼠懸停時顯示公司名稱
        title='<b>3D PCA 視覺化</b>',
        labels={
            'PC1': f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.2f}%)',
            'PC2': f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.2f}%)',
            'PC3': f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.2f}%)',
            'Cluster': '群組'
        },
        template='plotly_white',
        width=800, # 3D 圖通常可以設置更寬一些
        height=700,
    )

    fig_3d.update_traces(marker=dict(size=2)) # 將 size 調整為你認為合適的數值

    # 在 3D 圖中加入公司名稱標籤 (使用 go.Scatter3d 並設定 text)
    trace_annotations = []
    for i in range(min(50, len(df_pca_3d))):
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
                textposition="top center", # 文字位置
                scene='scene', # 確保文字在正確的 3D 場景中
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
        hovermode="closest",
        coloraxis_colorbar=dict(
            title="群組", 
            # 關鍵修正: 設定 tickvals 為整數
            tickvals=list(range(manual_k)),
            # ticktext=[str(val) for val in cluster_tick_vals], 
            dtick=1 # 確保刻度間隔為 1，進一步強調整數刻度
        ),
    )
    #fig_3d.show()
    st.plotly_chart(fig_3d, width='stretch', key="key_tab4_show_pca_3d")
    
    #st.write("**主成分組成 (Components)**")
    #comp_df = pd.DataFrame(pca_3d.components_, columns=columns, index=["PC1", "PC2", "PC3"])
    #st.dataframe(comp_df.style.format("{:.3f}"), width='stretch')  




#--------------------------------------------------------------------------------------------
# def tab4_show_cluster_feature_boxplots(df_clustered, last_n_months):
#     print("\ndf_clustered, Tab 42 ===>\n", df_clustered)    
#     st.write('📋 **各群特徵差異**')
    
#     #
#     fig, ax = plt.subplots(figsize=(6,3))
#     sns.barplot(data=df_clustered, x='cluster', y='continuous_flag', estimator='mean', ax=ax)
    
#     #ax.set_title(f'最近{last_n_months}個月,月增與年增同時成長', fontsize=10)
#     #ax.set_ylabel('月增/年增同時連續成長', fontsize=10)
#     ax.set_title(f'mom/yoy growth for recent {last_n_months} months', fontsize=10)
#     ax.set_ylabel('mom/yoy growth', fontsize=10)
#     ax.set_xlabel('cluster', fontsize=10)    
    
#     ax.tick_params(axis='x', labelsize=8)
#     ax.tick_params(axis='y', labelsize=8)
#     st.pyplot(fig)

#     #
#     #for col, title in [('avg_mom','平均月增成長'), ('std_mom','月增標準差'), ('avg_yoy','平均年增成長'), ('std_yoy','年增標準差')]:
#     for col, title in [('avg_mom','avg_mom'), ('std_mom','std_mom'), ('avg_yoy','avg_yoy'), ('std_yoy','std_yoy')]:
#         fig_c, ax_c = plt.subplots(figsize=(6,3))
#         sns.boxplot(data=df_clustered, x='cluster', y=col, ax=ax_c)
#         #
#         ax_c.set_title(f'{title}', fontsize=10)
#         ax_c.set_ylabel(f'{title}', fontsize=10)
#         ax_c.set_xlabel('cluster', fontsize=10)        
#         #
#         ax_c.tick_params(axis='x', labelsize=8)
#         ax_c.tick_params(axis='y', labelsize=8)
#         st.pyplot(fig_c)
