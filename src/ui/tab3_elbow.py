import streamlit as st
import pandas as pd
import plotly.express as px
from ui.ui_main import llm_chat


def tab3_show_elbow_plot(min_k, max_k, wcss):
    st.markdown("""**Elbow method/K值** """)
    
    # 將結果轉換成 Plotly 易於處理的 DataFrame
    elbow_df = pd.DataFrame({
        '分群數量 K': range(1, max_k + 1),
        '群內平方和 (WCSS/SSE)': wcss
    })
    
    # --- 使用 Plotly Express 繪製圖表 ---
    fig = px.line(
        elbow_df,
        x='分群數量 K',
        y='群內平方和 (WCSS/SSE)',
        title='<b>肘部法則 (Elbow Method) - 決定最佳 K 值</b>',
        markers=True, # 顯示數據點
        template='plotly_white' # 使用簡潔的白色背景模板
    )
    fig.update_yaxes(
        # 這裡沒有設定 dtick，讓 Plotly 自動優化間距 (效果通常不錯)
        # 如果您需要手動設定，可以改成：dtick=500 （依據您的資料範圍而定）
        showgrid=True, 
        gridcolor='lightgray', 
        #dtick = 2000, # 確保 X 軸間隔為 1
        #griddash='dot'
        showline=True, 
        linewidth=1.5, 
        linecolor='black',
    )
    # X 軸設定：確保每個 K 值都有刻度
    fig.update_xaxes(
        tickmode = 'linear', 
        showgrid=True, 
        gridcolor='lightgray', 
        #griddash='dot'
        # 邊框線設定
        showline=True, 
        linewidth=1.5, 
        linecolor='black', 
    )
    fig.update_layout(  
        #font=dict(family="Arial, Noto Sans CJK TC, sans-serif", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor='rgba(245,245,245,1)', 
        hovermode="x unified",
        width=800,
        height=600,
    )
    # 顯示圖表, fig.show()
    st.plotly_chart(fig, width='stretch', key="tab3_show_elbow_plot")  
        
    #    
    # format_mapping = {
    #     '分群數量 K': "{:d}",      # 或 "{:.0f}"，用於整數
    #     '群內平方和 (WCSS/SSE)': "{:.4f}"  # 或您的第二個欄位名稱
    # }
    # styled_df = elbow_df.style.format(format_mapping)
    # st.dataframe(styled_df, hide_index=True, width='stretch')
    
    #--- AI  ---      
    input = f"Please explain Elbow method result: {elbow_df} and answer in Traditional Chinese."            
    if st.button("AI 解釋 Elbow method", key="key_ai_explain_elbow_method"):       
    #llm_chat(input) 
        with st.status("📂 AI 解釋 Elbow method 中...", expanded=True) as status:  
            from agent.langGraph_gemini import stream_llm_updates     
            response = stream_llm_updates(input)     
            if response:                                  
                st.markdown(response)             
                status.update(label="📂 AI 解釋 Elbow method", state="complete", expanded=True)  
                
            status.update(label="📂 AI 解釋 Elbow method", state="complete", expanded=True)      