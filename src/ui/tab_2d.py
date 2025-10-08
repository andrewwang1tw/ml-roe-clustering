import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from ui.ui_main import llm_chat
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

#--------------------------------------------------------------------------------------------
# tab4_show_pca_2d
#--------------------------------------------------------------------------------------------    
def tab4_show_pca_2d(df_pca_2d, pca_2d, columns, manual_k):
    fig_2d = px.scatter(
        df_pca_2d,
        x='PC1',
        y='PC2',
        color='Cluster',  # 使用 Cluster 欄位來決定顏色
        hover_name='名稱', # 滑鼠懸停時顯示公司名稱
        title='<b>2D PCA 視覺化 (PC1 vs PC2)</b>',
        labels={
            'PC1': f'主成分 1 (PC1, 變異量: {pca_2d.explained_variance_ratio_[0]*100:.2f}%)',
            'PC2': f'主成分 2 (PC2, 變異量: {pca_2d.explained_variance_ratio_[1]*100:.2f}%)',
            'Cluster': '群組'
        },
        template='plotly_white',
        width=800,
        height=700,
        opacity=0.8,
    )
    
    # 調整點的大小 (讓點小一點，避免遮擋)
    fig_2d.update_traces(marker=dict(size=6))

    # 加入公司名稱標籤 (使用 add_trace 或 add_annotations，Plotly 通常傾向 hover_name)  
    for i in range(min(50, len(df_pca_2d))):
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
        hovermode="closest", # 讓 hover 效果更精確
        coloraxis_colorbar=dict(
            title="群組", 
            # 關鍵修正: 設定 tickvals 為整數
            tickvals=list(range(manual_k)),
            # ticktext=[str(val) for val in cluster_tick_vals], 
            dtick=1 # 確保刻度間隔為 1，進一步強調整數刻度
        ),
    )
    #fig_2d.show() 
    st.plotly_chart(fig_2d, width='stretch', key="key_tab4_show_pca_2d")   
    
    #st.write("**主成分組成 (Components)**")
    #comp_df = pd.DataFrame(pca_2d.components_, columns=columns, index=["PC1", "PC2"])
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
