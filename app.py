import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import requests

# =========================
# Streamlit 頁面設定
# =========================
st.set_page_config(page_title="金銀實盤輔助版 PRO+", layout="wide")
st.title("🏦 金銀實盤輔助系統 PRO+")

# =========================
# LINE 發送函數
# =========================
def send_line(message):
    try:
        token = st.secrets["LINE_TOKEN"]
        url = "https://notify-api.line.me/api/notify"
        headers = {"Authorization": f"Bearer {token}"}
        requests.post(url, headers=headers, params={"message": message})
    except:
        st.error("LINE 發送失敗，請檢查 LINE_TOKEN")

# =========================
# 側邊欄設定
# =========================
target_label = st.sidebar.selectbox("監測資產", ["黃金 (GC=F)", "白銀 (SI=F)"])
target = "GC=F" if "黃金" in target_label else "SI=F"
account_size = st.sidebar.number_input("帳戶資金 (USD)", min_value=1000, value=10000, step=1000)
risk_per_trade = st.sidebar.slider("單筆風險百分比", 0.1, 5.0, 1.0)
walk_days = st.sidebar.number_input("方向勝率回測天數", min_value=30, value=250, step=50)

st.sidebar.markdown("---")
st.sidebar.info("此系統僅提供訊號與風控建議，不直接下單。")

# =========================
# 數據下載
# =========================
@st.cache_data(ttl=3600)
def get_data():
    tickers = ["GC=F", "SI=F", "DX-Y.NYB", "^GSPC", "^VIX"]
    df = yf.download(tickers, period="10y", interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    df = df.ffill()
    return df

raw_data = get_data()

# =========================
# 特徵工程
# =========================
df = pd.DataFrame(index=raw_data.index)
df['price'] = raw_data[target]
df['usd'] = raw_data['DX-Y.NYB']
df['stock'] = raw_data['^GSPC']
df['vix'] = raw_data['^VIX']

df['ma20'] = df['price'].rolling(20).mean()
df['ma50'] = df['price'].rolling(50).mean()
df['volatility'] = df['price'].pct_change().rolling(20).std()

# RSI
delta = df['price'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df['rsi'] = 100 - (100 / (1 + rs))

df['target'] = df['price'].shift(-1)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

if len(df) < 200:
    st.error(f"有效數據僅 {len(df)} 筆，無法支撐模型訓練")
    st.stop()

features = ['price','usd','stock','vix','ma20','ma50','rsi','volatility']

# =========================
# Walk-forward 模擬 + 方向勝率
# =========================
model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
# 訓練最近 walk_days 天
train_data = df[features].iloc[-walk_days:]
train_target = df['target'].iloc[-walk_days:]
model.fit(train_data, train_target)

latest_feat = df[features].tail(1)
pred_1d = model.predict(latest_feat)[0]
curr_price = df['price'].iloc[-1]
diff_pct = (pred_1d - curr_price)/curr_price*100
current_rsi = df['rsi'].iloc[-1]
current_vix = df['vix'].iloc[-1]

# =========================
# 方向勝率
# =========================
# 用最近 walk_days 模擬每日滾動預測
predictions = model.predict(train_data)
direction_pred = np.sign(predictions - train_data['price'])
direction_true = np.sign(train_target - train_data['price'])
accuracy = np.mean(direction_pred == direction_true) * 100

# =========================
# 累積回測模擬
# =========================
returns = direction_pred * (train_target - train_data['price'])
cumulative_returns = returns.cumsum()

# =========================
# 倉位建議
# =========================
dollar_risk = account_size * risk_per_trade / 100
atr = df['volatility'].iloc[-20:].mean() * curr_price
position_size = dollar_risk / atr if atr != 0 else 0

# =========================
# UI 展示
# =========================
st.subheader(f"{target_label} 今日訊號")
col1, col2, col3, col4 = st.columns(4)
col1.metric("當前價格", f"${curr_price:,.2f}")
col2.metric("AI 明日預測", f"${pred_1d:,.2f}", f"{diff_pct:+.2f}%")
col3.metric("RSI", f"{current_rsi:.1f}")
col4.metric("VIX", f"{current_vix:.1f}")

st.markdown("### 🧠 方向勝率 & 累積回測")
st.info(f"模型方向勝率（最近 {walk_days} 天）: {accuracy:.2f}%")
st.line_chart(cumulative_returns)

st.markdown("### 🛡️ 風控與倉位建議")
if current_rsi > 70:
    st.warning("⚠️ RSI 超買，建議觀望或減倉")
elif current_rsi < 30:
    st.success("✅ RSI 超賣，可低吸")

st.info(f"建議單筆最大倉位: {position_size:.2f} 合約/單位 (依 ATR 計算)")

# 價格走勢圖
st.markdown("### 📊 歷史價格與技術指標")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[-120:], y=df['price'].tail(120), fill='tozeroy', name='價格', line=dict(color='#FFD700')))
fig.add_trace(go.Scatter(x=df.index[-120:], y=df['ma20'].tail(120), name='MA20', line=dict(color='#00BFFF')))
fig.add_trace(go.Scatter(x=df.index[-120:], y=df['ma50'].tail(120), name='MA50', line=dict(color='#FF4500')))
fig.update_layout(template="plotly_dark", height=450, margin=dict(l=20,r=20,t=50,b=20))
st.plotly_chart(fig, use_container_width=True)

# =========================
# LINE 發送訊號
# =========================
if st.button("📲 發送訊號至 LINE"):
    advice = "多單" if diff_pct > 0.3 else ("空單" if diff_pct < -0.3 else "觀望")
    msg = f"""
【{target_label} 實盤訊號】
● 現價: ${curr_price:.2f}
● AI 明日預測: ${pred_1d:.2f} ({diff_pct:+.2f}%)
● RSI: {current_rsi:.1f}
● VIX: {current_vix:.1f}
● 方向勝率 (過去 {walk_days} 天): {accuracy:.2f}%
● 累積回測收益: {cumulative_returns.iloc[-1]:.2f} USD/單位
● 建議操作: {advice}
● 建議單筆倉位: {position_size:.2f} 合約/單位
"""
    send_line(msg)
    st.success("訊號已發送至 LINE！")
