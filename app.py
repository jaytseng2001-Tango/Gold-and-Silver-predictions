# --------------------------
# 2026 金銀 AI 實盤輔助版 (Streamlit + Moomoo snapshot)
# --------------------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from moomoo import quote
import datetime

# ==========================
# 1. Streamlit 設定
# ==========================
st.set_page_config(page_title="金銀 AI 實盤輔助", layout="wide")
st.title("🏆 2026 金銀 AI 實盤輔助版")
st.sidebar.header("⚙️ 系統設定")

# ==========================
# 2. Moomoo 快照行情抓取
# ==========================
def get_snapshot(symbol):
    quote_ctx = quote.OpenQuoteContext(host='127.0.0.1', port=11111)
    ret, data = quote_ctx.get_market_snapshot([symbol])
    quote_ctx.close()
    if ret == 0:
        return data['last_price'][0]
    else:
        return None

# 側邊欄選擇商品
target_label = st.sidebar.selectbox("監測資產", ["黃金 (GC)", "白銀 (SI)"])
symbol_map = {"黃金 (GC)": "US.GC", "白銀 (SI)": "US.SI"}
symbol = symbol_map[target_label]

# 抓即時價格
current_price = get_snapshot(symbol)
st.metric("💰 即時價格", f"${current_price:.2f}")

# ==========================
# 3. 歷史資料（Yahoo）作 AI 訓練
# ==========================
import yfinance as yf
hist = yf.download({"GC=F":"GC=F","SI=F":"SI=F"}[target_label.split()[0]+"=F"],
                   period="5y", interval="1d")['Close'].ffill().dropna()
df = pd.DataFrame(hist)
df.rename(columns={df.columns[0]:'price'}, inplace=True)

# 技術指標
df['ma20'] = df['price'].rolling(20).mean()
df['ma50'] = df['price'].rolling(50).mean()
delta = df['price'].diff()
gain = (delta.where(delta>0,0)).rolling(14).mean()
loss = (-delta.where(delta<0,0)).rolling(14).mean()
df['rsi'] = 100 - (100/(1+gain/loss))
df['target'] = df['price'].shift(-1)
df.dropna(inplace=True)

# ==========================
# 4. AI 模型預測
# ==========================
features = ['price','ma20','ma50','rsi']
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(df[features][:-1], df['target'][:-1])

latest_feat = df[features].tail(1)
pred_1d = model.predict(latest_feat)[0]
pred_2d = pred_1d * 1.002  # 簡單加權模擬多日預測
pred_3d = pred_1d * 1.004
pred_1w = pred_1d * 1.01
pred_1m = pred_1d * 1.03
pred_3m = pred_1d * 1.08
pred_6m = pred_1d * 1.15

st.subheader("📈 AI 漲跌預測")
st.write(f"明日 1D: ${pred_1d:.2f}")
st.write(f"2 日 2D: ${pred_2d:.2f}")
st.write(f"3 日 3D: ${pred_3d:.2f}")
st.write(f"1 週 1W: ${pred_1w:.2f}")
st.write(f"1 月 1M: ${pred_1m:.2f}")
st.write(f"3 月 3M: ${pred_3m:.2f}")
st.write(f"6 月 6M: ${pred_6m:.2f}")

# ==========================
# 5. 回測 & 方向勝率
# ==========================
df['pred'] = model.predict(df[features])
df['direction_correct'] = (df['pred'].shift(1) - df['price'].shift(1)) * (df['target'] - df['price']) > 0
win_rate = df['direction_correct'].mean() * 100
st.metric("🎯 方向勝率", f"{win_rate:.2f}%")

# 累積回測收益
df['returns'] = (df['pred'].shift(1) / df['price'].shift(1) - 1)
df['cum_returns'] = (1 + df['returns']).cumprod()
st.line_chart(df[['price','cum_returns']].tail(200))

# ==========================
# 6. 單筆風險百分比
# ==========================
risk_pct = st.sidebar.slider("單筆風險 (%)", 0.1, 5.0, 1.0)
st.info(f"建議單筆風險控制在 {risk_pct:.1f}% 之內")

# ==========================
# 7. 建議買入/賣出時點
# ==========================
st.subheader("💡 買賣建議")
advice = "觀望"
if df['rsi'].iloc[-1] < 30 and (pred_1d - current_price)/current_price > 0.5/100:
    advice = "建議買入"
elif df['rsi'].iloc[-1] > 70:
    advice = "建議賣出"
st.write(advice)
st.write(f"RSI: {df['rsi'].iloc[-1]:.1f}")

# ==========================
# 8. 歷史價格與技術指標
# ==========================
st.subheader("📊 歷史價格與技術指標")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[-120:], y=df['price'].tail(120), name="歷史價格", line=dict(color='#FFD700')))
fig.add_trace(go.Scatter(x=df.index[-120:], y=df['ma20'].tail(120), name="20日均線", line=dict(color='#00BFFF')))
fig.add_trace(go.Scatter(x=df.index[-120:], y=df['ma50'].tail(120), name="50日均線", line=dict(color='#FF4500')))
fig.update_layout(template="plotly_dark", height=450)
st.plotly_chart(fig, use_container_width=True)
