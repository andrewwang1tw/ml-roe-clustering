import base64
import platform
import streamlit as st
import matplotlib
import seaborn as sns
from agent.langGraph_gemini import stream_llm_updates

#-----------------------------------------------------------------------------------
# 顯示 sidebar 輸入元件，回傳所有參數值
#-----------------------------------------------------------------------------------
def sidebar_inputs():  
    # 在 sidebar 建立一個容器，讓按鈕顯示在最上面
    top_container = st.sidebar.container()
    run = top_container.button('🚀 執行分析 🚀', key='run_button')
     
    st.sidebar.header('產業別選擇')
    recommended_values = ['1-全部', '2-電子', '3-傳產', '4-半導體','5-電腦與週邊','6-電子零組件','7-光電','8-通信網路','9-其他電子'] 
    selected_data = st.sidebar.selectbox(
        label='選擇產業:',
        options=recommended_values,
        index=0, 
        key='key_select' # 連結到 session_state
    )
    selected_data_value = int(selected_data.split('-')[0])


    st.sidebar.header('PCA 選項')
    manual_pca = st.sidebar.number_input('手動設定 P(主成分數)', min_value=2, max_value=20, value=8, key="manual_pca")


    st.sidebar.header('Clustering 選項')
    manual_k = st.sidebar.number_input('手動設定 k(分群數)', min_value=1, max_value=15, value=4, key="manual_k")
    #k_min = st.sidebar.number_input('k 最低', min_value=1, max_value=10, value=3, key="k_min")
    #k_max = st.sidebar.number_input('k 最高', min_value=2, max_value=15, value=15, key="k_max")
        
    # st.sidebar.header('資料庫連線')
    # server = st.sidebar.text_input('SQL Server', value='localhost', key="db_server")
    # database = st.sidebar.text_input('Database', value='stock_tw', key="db_name")
    # trusted = st.sidebar.checkbox('Use Trusted Connection (Windows Auth)', value=True, key="trusted_connection")
    
    return {
        # 'server': server,
        # 'database': database,
        # 'trusted': trusted,
        #'k_min': k_min,
        #'k_max': k_max,
        'selected_data_value': selected_data_value,
        'manual_k': manual_k,
        'manual_pca': manual_pca,
        'run': run
    }


#-----------------------------------------------------------------------------------
# 輔助函式：取得群集標籤列表
#-----------------------------------------------------------------------------------
def get_cluster_labels(df_clustered):
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    cluster_labels = [f"群 {i} ({cnt} 筆)" for i, cnt in cluster_counts.items()]
    return cluster_counts, cluster_labels

   
#-----------------------------------------------------------------------------------
# llm_chat
#-----------------------------------------------------------------------------------
def llm_chat(input: str):  
    with st.status("📂 AI 解釋中...", expanded=True) as status:       
        response = stream_llm_updates(input)     
        if response:                                  
            st.markdown(response) 
        
        status.update(label="📂 AI 解釋", state="complete", expanded=True)    
       

#-----------------------------------------------------------------------------------
# set_font 中文
#-----------------------------------------------------------------------------------       
# def set_font():       
#     sns.set_theme(style="whitegrid")  # 你也可以選 whitegrid, darkgrid, ticks 等    
#     matplotlib.rcParams['axes.unicode_minus'] = False
    
#     if platform.system() == "Windows":
#         matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
#     elif platform.system() == "Darwin":
#         matplotlib.rcParams['font.family'] = 'Heiti TC'
#     else:
#         matplotlib.rcParams['font.family'] = 'Noto Sans CJK TC'  
    

    
#-----------------------------------------------------------------------------------
# set_layout
#-----------------------------------------------------------------------------------        
def set_layout():
    st.set_page_config(
        layout='wide', 
        page_title='📈 台股分群(Clustering)'
    )  
    
    # 浮水印的 CSS 樣式
    watermark_css = """
    <style>
    .watermark {
        position: fixed;
        top: 5%;
        left: 70%;
        #transform: translate(-50%, -50%) rotate(-40deg);
        font-size: 3rem;
        font-weight: bold;
        color: rgba(128, 128, 128, 0.1); /* 灰色的透明度 */
        user-select: none;
        pointer-events: none;
        z-index: 1000;
    }
    .watermark-img {{
        width: 100px; /* 圖片大小 */
        height: 50px;
    }}
    </style>
    """  
    
    #寫入 CSS, 加入浮水印文字
    st.markdown(watermark_css, unsafe_allow_html=True)   
    st.markdown('<div class="watermark">台股分群(Clustering)</div>', unsafe_allow_html=True)
    
    # # 加入 Image
    # image_path = "_img/1.png"
    # def get_base64_image(path):
    #     with open(path, "rb") as image_file:
    #         return base64.b64encode(image_file.read()).decode()
    
    # image_b64 = get_base64_image(image_path)    
    # st.markdown(f'<div><img src="data:image/png;base64,{image_b64}" style="width: 500px; display: block; margin-left: auto; margin-right: auto;"></div>', 
    #             unsafe_allow_html=True)

