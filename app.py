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
account_size = st.sidebar.number_input("帳戶資金 (USD)", min_value=1000, value=10000, step=1000,
                                       help="你目前可用投資資金")
risk_per_trade = st.sidebar.slider("單筆風險百分比", 0.1, 5.0, 1.0,
                                   help="每次交易最多承擔多少資金風險，例如 1%")
walk_days = st.sidebar.number_input("方向勝率回測天數", min_value=30, value=250, step=50,
                                    help="用最近多少天計算模型預測漲跌的正確率")

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
    df = df.ffill().bfill()
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

# 生成多時間目標
timeframes = {
    '1天':1, '2天':2, '3天':3, '1週':5, '1個月':20, '3個月':60, '6個月':120
}
for name, shift in timeframes.items():
    df[f'target_{name}'] = df['price'].shift(-shift)

# 清理數據
df = df.replace([np.inf, -np.inf], np.nan).dropna()
features = ['price','usd','stock','vix','ma20','ma50','rsi','volatility']

if len(df) < 200:
    st.error(f"有效數據僅 {len(df)} 筆，無法支撐模型訓練")
    st.stop()

# =========================
# 訓練模型
# =========================
models = {}
predictions = {}
for name in timeframes.keys():
    models[name] = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    models[name].fit(df[features], df[f'target_{name}'])
    predictions[name] = models[name].predict(df[features].tail(1))[0]

curr_price = df['price'].iloc[-1]
diff_pct = {k:(predictions[k]-curr_price)/curr_price*100 for k in predictions}
current_rsi = df['rsi'].iloc[-1]
current_vix = df['vix'].iloc[-1]

# =========================
# 方向勝率 & 累積回測
# =========================
train_data = df[features].iloc[-walk_days:]
accuracy = {}
cumulative_returns = {}
for name in timeframes.keys():
    train_target = df[f'target_{name}'].iloc[-walk_days:]
    pred = models[name].predict(train_data)
    dir_pred = np.sign(pred - train_data['price'])
    dir_true = np.sign(train_target - train_data['price'])
    accuracy[name] = np.mean(dir_pred==dir_true)*100
    returns = dir_pred * (train_target - train_data['price'])
    cumulative_returns[name] = returns.cumsum()

# =========================
# 單筆倉位建議
# =========================
dollar_risk = account_size * risk_per_trade / 100
atr = df['volatility'].iloc[-20:].mean() * curr_price
position_size = dollar_risk / atr if atr!=0 else 0

# =========================
# UI 展示
# =========================
st.subheader(f"{target_label} 今日訊號")
col1, col2, col3, col4 = st.columns(4)
col1.metric("當前價格", f"${curr_price:,.2f}")
col2.metric("RSI", f"{current_rsi:.1f}")
col3.metric("VIX", f"{current_vix:.1f}")
col4.metric("單筆建議倉位", f"{position_size:.2f} 單位")

# 多時間AI預測與建議
st.markdown("### 🧠 AI 多時間漲跌預測")
pred_table = pd.DataFrame({
    "時間": list(timeframes.keys()),
    "AI 預測價格": [f"${predictions[k]:.2f}" for k in timeframes.keys()],
    "漲跌幅 (%)": [f"{diff_pct[k]:+.2f}%" for k in timeframes.keys()],
    "方向勝率 (%)": [f"{accuracy[k]:.1f}%" for k in timeframes.keys()]
})
st.table(pred_table)

# 建議買入/賣出價格區間
st.markdown("### 💡 建議買入 / 賣出價格區間")
buy_price = {k: curr_price*(1-0.005) for k in timeframes.keys()}
sell_price = {k: curr_price*(1+0.005) for k in timeframes.keys()}
price_table = pd.DataFrame({
    "時間": list(timeframes.keys()),
    "建議買入價格": [f"${buy_price[k]:.2f}" for k in timeframes.keys()],
    "建議賣出價格": [f"${sell_price[k]:.2f}" for k in timeframes.keys()]
})
st.table(price_table)

# 累積回測圖
st.markdown("### 📈 累積回測模擬")
cumu_fig = go.Figure()
for name in cumulative_returns.keys():
    cumu_fig.add_trace(go.Scatter(y=cumulative_returns[name], name=name))
cumu_fig.update_layout(template="plotly_dark", height=400, margin=dict(l=20,r=20,t=20,b=20))
st.plotly_chart(cumu_fig, use_container_width=True)

# 歷史價格圖
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
    msg = f"【{target_label} 實盤訊號】\n現價: ${curr_price:.2f}\nRSI: {current_rsi:.1f}\nVIX: {current_vix:.1f}\n單筆建議倉位: {position_size:.2f}"
    for k in timeframes.keys():
        msg += f"\n{k}: AI預測 ${predictions[k]:.2f} ({diff_pct[k]:+.2f}%), 方向勝率 {accuracy[k]:.1f}%"
        msg += f"\n建議買入: ${buy_price[k]:.2f} / 賣出: ${sell_price[k]:.2f}"
    send_line(msg)
    st.success("訊號已發送至 LINE！")
