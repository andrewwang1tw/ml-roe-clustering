import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
import os

#--------------------------------------------------------------------------------
# make_engine
#--------------------------------------------------------------------------------
@st.cache_resource
def make_engine(server, database, driver='ODBC Driver 17 for SQL Server', trusted=True, user=None, password=None):
    if trusted:
        conn_str = f"mssql+pyodbc://{server}/{database}?trusted_connection=yes&driver={driver}"
    else:
        conn_str = f"mssql+pyodbc://{user}:{password}@{server}/{database}?driver={driver}"
    return create_engine(conn_str)

#--------------------------------------------------------------------------------
# load_sales_data
#--------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_sales_data(_engine, year=114, month_from=1):
    query = f"""
with avg_roe as (select 公司代號
	,AVG(ROE) as ROE
	,AVG(每股盈餘) as EPS
	,AVG(每股自由現金流量)	 as FCF
	,AVG(毛利率)	as 毛利率
	,AVG(營業利益率)	as 營業利益率
	,AVG(純益率)	as 純益率	
	,AVG(現金殖利率) as 現金殖利率
	,AVG(高登) as 高登
	,AVG(總資產周轉率) as 總資產周轉率
	,AVG(盈餘成長率) as 盈餘成長率
	,AVG(累計營收年成長率) as 營收年成長率	
FROM V2_ROE_每年資料_N where 年 >= 111 group by 公司代號
)
,most_recent_year as(
	select 公司代號,每股淨值,本益比 as PER,淨值比 as PBR,權益乘數, 股數 
	FROM V2_ROE_每年資料_N where 年 = 113
)
,price as(
	select 公司代號, avg(收盤價) as 平均價,stdev(收盤價) as 股價標準差, var(收盤價) as 股價變異數 
	from V_Price  where 開盤日期 >= '2024-10-01' and len(公司代號) = 4 group by 公司代號
)
select replace(g.產業別, ' ', '') as 產業別, g.名稱, a.*, 
	m.PBR, m.PER, m.每股淨值, m.權益乘數, (m.股數*10)/100000000 as 股本億元,
	p.平均價, p.股價標準差, p.股價變異數
from avg_roe a join most_recent_year m on a.公司代號 = m.公司代號
	join price p on a.公司代號 = p.公司代號
	join T_GROUP_N g on convert(varchar, a.公司代號) = g.代號
--where g.產業別 in ( '電腦及週邊設備', '電子零組件', '其他電子', '通信網路', '光電', '半導體')
--where g.產業別 not in ('電腦及週邊設備', '電子零組件', '其他電子', '通信網路', '光電', '半導體')
--where g.產業別 in ('半導體')
--where g.產業別 in ('電腦及週邊設備') 
--where g.產業別 in ('電子零組件')
--where g.產業別 in ('通信網路') 
--where g.產業別 in ('光電') 
where g.產業別 in ('其他電子') 
order by a.公司代號           
    """    
    df = pd.read_sql(query, _engine)
    return df

#--------------------------------------------------------------------------------
# load_sales_data_from_csv
#--------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data_from_csv(selected_data_value):    
    script_dir = os.path.dirname(os.path.abspath(__file__))   
     
    # '1-全部', '2-電子', '3-傳產', 
    # '4-半導體','5-電腦與週邊','6-電子零組件','7-光電','8-通信網路','9-其他電子'
    if selected_data_value == 1:
        file_path = os.path.join(script_dir, "stock-roe-avg-3y.csv")
    elif selected_data_value == 2:
        file_path = os.path.join(script_dir, "stock-roe-avg-3y-電子.csv")
    elif selected_data_value == 3:
        file_path = os.path.join(script_dir, "stock-roe-avg-3y-傳產.csv")
    elif selected_data_value == 4:
        file_path = os.path.join(script_dir, "stock-roe-avg-3y-半導體.csv")    
    elif selected_data_value == 5:
        file_path = os.path.join(script_dir, "stock-roe-avg-3y-電腦與週邊.csv")
    elif selected_data_value == 6:
        file_path = os.path.join(script_dir, "stock-roe-avg-3y-電子零組件.csv")
    elif selected_data_value == 7:
        file_path = os.path.join(script_dir, "stock-roe-avg-3y-光電.csv")  
    elif selected_data_value == 8:
        file_path = os.path.join(script_dir, "stock-roe-avg-3y-通信網路.csv")
    elif selected_data_value == 9:
        file_path = os.path.join(script_dir, "stock-roe-avg-3y-其他電子.csv")   
    else:
        file_path = os.path.join(script_dir, "stock-roe-avg-3y-all.csv")
         
    try:
        df = pd.read_csv(file_path)     
        print("\n====> load_data_from_csv, 成功讀取 Excel 檔案：", file_path,               
              "\n--- df.shape ---\n", df.shape, 
              "\n--- df.columns ---\n", df.columns.tolist(), 
              "\n--- df ---\n", df.head()
            )
        return df    
    except FileNotFoundError:
        print(f"\nload_data_from_csv ===> 錯誤:找不到檔案 '{file_path}'。請確認檔案已上傳至 GitHub 儲存庫。")
    except Exception as e:
        print(f"\nload_data_from_csv ===> 發生錯誤:{e}")    
    return None

#================================================================================
# 新增的主執行區塊
#================================================================================
if __name__ == "__main__":
    load_data_from_csv()
