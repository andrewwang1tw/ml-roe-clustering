import streamlit as st

from ml.clustering import run_analysis

import ui.ui_main as ui
from ui.tab0 import tab0_show_data
from ui.tab1_corr import tab1_show_correlation_heatmap
from ui.tab_pca import tab2_show_pca_before
from ui.tab3_elbow import tab3_show_elbow_plot
from ui.tab_2d import tab4_show_pca_2d
from ui.tab_3d import tab4_show_pca_3d 
from ui.tab5_summary import tab5_show_cluster_summary
from ui.tab6_cluster import tab6_show_cluster_details

def main():
    #ui.set_font()
    ui.set_layout()

    # --- 取得 sidebar 參數, 當 user 按 run 的時候執行 ---
    params = ui.sidebar_inputs()   
    if params['run']:
        with st.spinner('連線、擷取資料中...'):                           
            run_analysis(params)     
            print("\n\n======> [RUN], params ===>\n", params)

    
    # --- 若 session_state 已經有分析結果，顯示結果 ---
    if 'df_results' in st.session_state:        
        df_raw = st.session_state['df_raw']
        #
        corr_before = st.session_state['corr_before']
        corr_after = st.session_state['corr_after']
        features_to_remove = st.session_state['features_to_remove']        
        df_filtered = st.session_state['df_filtered']
        #
        X_scaled = st.session_state['X_scaled']        
        df_pca_2d = st.session_state['df_pca_2d']
        df_pca_3d = st.session_state['df_pca_3d']
        pca_2d = st.session_state['pca_2d']
        pca_3d = st.session_state['pca_3d']
        #
        wcss = st.session_state['wcss']
        # 
        cluster_summary = st.session_state['cluster_summary'] 
        df_results = st.session_state['df_results']    
       

        # --- cluster_summary ---   
        with st.container(border=True):          
            tab5_show_cluster_summary(cluster_summary, df_results)  
            #tab5_show_cluster_feature_boxplots(df_clustered, params['last_n_months'])   
                        
        # --- 分群結果 ---   
        with st.container(border=True):
        #with st.expander(label="🔵 分群結果 "):   
            tab6_show_cluster_details(df_results, page_size=10)    
                         

        # --- tab1: #st.expander("""📈 **PCA 主成分分析** """) ---
        with st.container(border=True): 
        #with st.expander(label="📈 PCA 主成分分析 "): 
             tab2_show_pca_before(X_scaled, df_filtered.columns)
             #tab2_show_pca_2d_before(df_pca_2d, pca_2d, df_filtered.columns)


        # --- st.markdown("""📊 **Elbow method/K值** """) ---
        with st.container(border=True):
        #with st.expander(label="📊 Elbow method/K值 "):
            tab3_show_elbow_plot(None, 15, wcss)            
                          
         
        # 建立 tabs "📈 Elbow method/K值",
        tab1, tab2, tab3, tab4 = st.tabs([ 
            "📈 PCA 2D圖(w/Cluster)",
            "📈 PCA 3D圖(w/Cluster)",
            "📊 相關分析/矩陣",
            "📅 原始特徵資料",       
        ])   
        
        with tab1: 
            tab4_show_pca_2d(df_pca_2d, pca_2d, df_filtered.columns, params['manual_k']) 
        
        with tab2:            
            tab4_show_pca_3d(df_pca_3d, pca_3d, df_filtered.columns, params['manual_k'])     
 
        #st.expander("""📊 **相關分析/矩陣** """):           
        with tab3: 
            tab1_show_correlation_heatmap(corr_before, corr_after, dropped=features_to_remove) 
        
        #st.expander("""✏️ **原始特徵資料**"""):    
        with tab4: 
            tab0_show_data(df_raw)

    else:  
        with st.container(border=True) as main_container:    
            st.markdown(
            """
            <div style='text-align: ;'>
                <h2>使用機器學習(Machine Learning) 對台股 1,800 多家公司 分群(Clustering)的分析工具</h2>
                <ul>
                    <li><h5>Web UI: Python, Streamlit</h5></li>
                    <li><h5>AI Agent, LLM: LangGraph(AI Agent), Gemini(LLM)</h5></li>
                    <li><h5>Unsupervised Learning: Clustering(K-Means), PCA(Principal Component Analysis)</h5></li>
                    <li><h5>傳統上，我們需要花很多時間手動篩選股票，而財務報表通常是靜態的，難以找出趨勢或相似的公司。 
                        我們將透過<b>機器學習</b>中的<b>分群(Clustering)</b>將台股 1,800 多家公司依最近3年財報與近一年股價資料分成不同屬性的<b>群集(Cluster)</b>，
                        提供一個簡潔、高層次的概觀。幫助你快速理解每個<b>群集</b>的核心特徵，進一步理解台股中的好公司。
                        </h5></li>
                </ul>
                
            </div>
            """, 
            unsafe_allow_html=True  # 必須允許 HTML 才能讓樣式生效
            )
            
            
            st.info('🚀 請在左側設定參數，然後按 🚀「執行分析」🚀 !!!')
            st.info('✏️ 產業別選擇: 可選擇「1-全部」或「2-電子」(電子包含 半導體, 電腦與週邊, 電子零組件, 光電, 通信網路, 其他電子) 或特定產業類別.')
            st.info('✏️ PCA 主成分分析的 P(主成分數) 建議設 6~10 之間.')
            st.info('✏️ Clustering 分群的 k(分群數) 建議設 3~7 之間.')
            
            
            # st.header("""一個機器學習(Machine Learning) 分群(Clustering)的股票分析工具""") 
            # st.header("""使用 Python, LangGraph, Gemini(LLM), Streamlit, Machine Learning(Clustering).""") 
            # st.markdown( """🔑傳統上，我們需要花很多時間手動篩選股票，而財務報表通常是靜態的，難以找出趨勢或相似的公司。 
            #             我們將透過**機器學習**中的**分群(Clustering)**將台股 1,800 多家公司依最近3年財報與近一年股價資料分成不同屬性的**群集(Cluster)**，
            #             提供一個簡潔、高層次的概觀。幫助你快速理解每個**群集**的核心**特徵**，進一步理解台股中的好公司。 """)                
                   
        # with st.expander('分群'):
        #     st.markdown("""🔑 表列出各**群集**的每一檔股票""")      
        #     st.markdown("""🔑 AI Agent 功能  
        #         任意點選一檔股票, 程式使用 **LangGraph** 透過 **TavilySearch** 搜尋工具搜尋該個股在網路上的相關資料, 
        #         再由的 **Gemini** LLM 針對搜尋結果做總結。
        #         """) 
            
        

if __name__ == "__main__":
    main()
