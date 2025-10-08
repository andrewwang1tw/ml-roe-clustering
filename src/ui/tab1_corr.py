import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

#-----------------------------------------------------------------------------------
# Tab 1, show_correlation_heatmap
#-----------------------------------------------------------------------------------
def tab1_show_correlation_heatmap(corr_before, corr_after, dropped=None):
    #print("\ncorr_before, Tab 1 ===>\n", corr_before)       
    #option = st.selectbox("📊 選擇", ["Before, 去高度相關前", "After, 去高度相關後"], key='select_corr_option')
    #corr_df = corr_before if option.startswith("Before") else corr_after   
   
    before, after = st.columns(2)
    with before:
        plot_corr_matrix(corr_before, plot_key="plot_key_before")
        
    with after:
        plot_corr_matrix(corr_after, plot_key="plot_key_after")
        
    corr_df = corr_before
    # 顯示高相關對
    st.write("⚠️ 高相關特徵 Pairs")
    threshold = 0.9
    high_corr_pairs = []
    cols = corr_df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr_df.iloc[i, j] > threshold:
                high_corr_pairs.append((cols[i], cols[j], round(corr_df.iloc[i, j], 3)))

    if high_corr_pairs:
        st.dataframe(pd.DataFrame(high_corr_pairs, columns=["Feature A","Feature B","Correlation"]))
    else:
        st.success("沒有超過相關係數閾值的特徵 pair ✅")
        
    # 被移除的高相關特徵
    if dropped is not None:
        st.write("🗑️ 被移除的高相關特徵")
        st.write(dropped if dropped else "✅ 沒有特徵被移除")    


#---------------------------------------------------------------------
#
#---------------------------------------------------------------------
def plot_corr_matrix(df,  plot_key=None):
        
    title = "<b>原始相關矩陣圖</b>"
    if plot_key=="plot_key_after":          
        title = "<b>去高度相關特徵後 - 相關矩陣圖</b>"
    
    correlation_matrix = df.copy()    
    fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",  # 在每個方塊中自動顯示相關係數，保留兩位小數
        aspect="equal",   # 保持方塊為正方形
        color_continuous_scale=px.colors.diverging.RdBu_r, # 使用紅藍色系，'_r'表示反轉，讓紅色表示負相關，藍色表示正相關
        color_continuous_midpoint=0,  # 確保白色/中點在 0.0 相關係數處
        labels=dict(x="特徵 (X軸)", y="特徵 (Y軸)", color="相關係數"),
        title=title
    )

    # 設定邊界線 (Marker Line)
    fig.update_traces(
        xgap=1,  # 設定水平間隔 (像素)，數值越大邊界越粗
        ygap=1   # 設定垂直間隔 (像素)
    )

    # 設定 X 和 Y 軸的標籤 (使用矩陣的索引名稱)
    fig.update_xaxes(
        tickangle=-45, # 旋轉 X 軸標籤，避免重疊
        tickmode='array',
        tickvals=list(range(len(correlation_matrix.columns))),
        ticktext=correlation_matrix.columns.tolist(),
        showgrid=False
    )

    fig.update_yaxes(
        tickmode='array',
        tickvals=list(range(len(correlation_matrix.index))),
        ticktext=correlation_matrix.index.tolist(),
        showgrid=False
    )

    # 調整佈局和字體
    fig.update_layout(
        width=900,
        height=900,
        title_x=0.5, # 標題置中
        font=dict(
            #family=cjk_font_family,
            size=12
        ),
        # 調整顏色條 (Colorbar) 設置
        coloraxis_colorbar=dict(
            title="相關係數",
            tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0],
            #title_font_family=cjk_font_family
        )
    )
    
    #fig.show()
    st.plotly_chart(fig, use_container_width=True, key=plot_key)
    