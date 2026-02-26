import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# ==============================
# 1️⃣ 設定頁面
# ==============================
st.set_page_config(page_title="Gold-Silver AI 實盤輔助", layout="wide")
st.title("🏆 Gold & Silver AI 實盤輔助系統")

# ==============================
# 2️⃣ 側邊欄：功能說明與設定
# ==============================
st.sidebar.header("⚙️ 功能設定與說明")
st.sidebar.markdown("""
**單筆風險百分比**：建議單次操作投入資金占總資金比例  
**方向勝率**：AI 對未來價格漲跌判斷的準確率  
**回測天數**：使用歷史資料模擬策略效果的天數  
**風控與倉位建議**：依據風險評級建議持倉大小  
**歷史價格與技術指標**：價格、均線、RSI、波動率等技術分析  
**買入/賣出建議**：AI 預測何時買入或賣出，以及價格區間  
**預測時間框**：未來一天、兩天、一週、一個月、三個月、半年漲跌預測
""")

asset = st.sidebar.selectbox("選擇資產", ["黃金 XAU/USD", "白銀 XAG/USD"])
risk_pct = st.sidebar.slider("單筆風險百分比", 1, 10, 2)
backtest_days = st.sidebar.slider("回測天數", 30, 365, 90)

symbol = "XAU" if "黃金" in asset else "XAG"

# ==============================
# 3️⃣ Gold‑API 即時價格抓取
# ==============================
API_KEY = "goldapi-quickstart-XXXX"  # Quickstart Key
url = f"https://www.goldapi.io/api/{symbol}/USD"
headers = {"x-access-token": API_KEY, "Content-Type": "application/json"}

try:
    response = requests.get(url, headers=headers, timeout=10)
    data = response.json()
    curr_price = data.get('price', None)
    timestamp = data.get('timestamp', datetime.now().isoformat())
except Exception as e:
    st.error("即時資料抓取失敗，請稍後再試")
    st.stop()

st.subheader(f"📈 {asset} 即時價格")
st.metric("即時價格 (USD)", f"${curr_price:,.2f}", delta=None)

# ==============================
# 4️⃣ 取得歷史資料 (yfinance)
# ==============================
import yfinance as yf

ticker = "GC=F" if symbol=="XAU" else "SI=F"
hist = yf.download(ticker, period="5y", interval="1d")['Close'].ffill()

df = pd.DataFrame()
df['price'] = hist
df['ma20'] = df['price'].rolling(20).mean()
df['ma50'] = df['price'].rolling(50).mean()
delta = df['price'].diff()
gain = (delta.where(delta>0,0)).rolling(14).mean()
loss = (-delta.where(delta<0,0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df['rsi'] = 100 - (100 / (1 + rs))
df['target'] = df['price'].shift(-1)
df = df.dropna()

# ==============================
# 5️⃣ AI 模型預測
# ==============================
features = ['price','ma20','ma50','rsi']
train_size = int(len(df)*0.8)
train = df[:train_size]
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(train[features], train['target'])

latest = df[features].tail(1)
pred_next = model.predict(latest)[0]
diff_pct = (pred_next - curr_price)/curr_price*100

st.subheader("🤖 AI 預測")
st.metric("明日價格預測", f"${pred_next:,.2f}", f"{diff_pct:+.2f}%")

# ==============================
# 6️⃣ 回測與方向勝率
# ==============================
df['pred'] = model.predict(df[features])
df['correct'] = np.sign(df['pred'].diff()) == np.sign(df['target'].diff())
win_rate = df['correct'].tail(backtest_days).mean() * 100

st.subheader("📊 回測與方向勝率")
st.metric(f"{backtest_days} 日方向勝率", f"{win_rate:.2f}%")

# ==============================
# 7️⃣ 歷史價格與技術指標圖
# ==============================
st.subheader("📈 歷史價格與技術指標")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['price'], name='價格', line=dict(color='#FFD700')))
fig.add_trace(go.Scatter(x=df.index, y=df['ma20'], name='20日均線', line=dict(color='#00BFFF')))
fig.add_trace(go.Scatter(x=df.index, y=df['ma50'], name='50日均線', line=dict(color='#FF4500')))
fig.update_layout(template="plotly_dark", height=450, margin=dict(l=20,r=20,t=30,b=20))
st.plotly_chart(fig, use_container_width=True)

# ==============================
# 8️⃣ 買入/賣出建議
# ==============================
st.subheader("💡 買入/賣出建議")
future_days = [1,2,7,30,90,180]
pred_prices = []

for d in future_days:
    # 假設用單步預測作為簡單模擬
    last_feat = df[features].iloc[-1:].copy()
    pred_list = []
    price_sim = last_feat['price'].values[0]
    for i in range(d):
        last_feat['price'] = price_sim
        price_sim = model.predict(last_feat)[0]
    pred_prices.append(price_sim)

suggestion = []
for i, d in enumerate(future_days):
    buy_sell = "買入" if pred_prices[i] > curr_price else "賣出"
    suggestion.append(f"未來 {d} 天 → 預測 {buy_sell}，價格: ${pred_prices[i]:,.2f}")

st.write("\n".join(suggestion))
