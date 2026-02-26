import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import yfinance as yf
import pytz

# ==============================
# 1️⃣ 頁面 UI 美化與設定
# ==============================
st.set_page_config(page_title="Gold & Silver AI Pro+", layout="wide", initial_sidebar_state="expanded")

# 自定義 CSS 讓介面更有科技感
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #FFD700; }
    .stMetric { background-color: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .status-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏆 Gold & Silver AI 實盤輔助系統 (Pro+ Elite)")
st.markdown("---")

# ==============================
# 2️⃣ 側邊欄：功能設定
# ==============================
st.sidebar.header("🛡️ 實盤風控參數")
asset = st.sidebar.selectbox("選擇資產", ["黃金 XAU/USD", "白銀 XAG/USD"])
risk_pct = st.sidebar.slider("單筆風險金 (%)", 0.5, 5.0, 2.0, 0.5)
total_capital = st.sidebar.number_input("總投資本金 (USD)", value=10000)

st.sidebar.markdown("---")
st.sidebar.header("🧠 AI 模型配置")
backtest_days = st.sidebar.slider("勝率回測窗口 (天)", 30, 180, 90)
use_market_context = st.sidebar.checkbox("引入市場關聯 (美元/標普)", value=True)

symbol = "XAU" if "黃金" in asset else "XAG"
ticker = "GC=F" if symbol=="XAU" else "SI=F"

# ==============================
# 3️⃣ 多維度數據抓取 (關鍵：提高準確率)
# ==============================
@st.cache_data(ttl=600)
def fetch_enhanced_data(ticker):
    # 同時抓取目標、美元指數(DXY)、標普500(SPY)、恐慌指數(VIX)
    tickers = [ticker, "DX-Y.NYB", "SPY", "^VIX"]
    data = yf.download(tickers, period="5y", interval="1d")['Close'].ffill()
    
    df = pd.DataFrame(index=data.index)
    df['price'] = data[ticker]
    df['dxy'] = data['DX-Y.NYB']
    df['spy'] = data['SPY']
    df['vix'] = data['^VIX']
    
    # --- 特徵工程升級 ---
    df['ma20'] = df['price'].rolling(20).mean()
    df['ma50'] = df['price'].rolling(50).mean()
    # ATR 波動率概念
    df['volatility'] = df['price'].pct_change().rolling(20).std()
    # RSI
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    
    # 預測目標：隔日漲跌幅 (pct_change 比絕對價格更好預測)
    df['target_return'] = df['price'].pct_change().shift(-1)
    return df.dropna()

df = fetch_enhanced_data(ticker)

# ==============================
# 4️⃣ 價格儀表板
# ==============================
curr_price = df['price'].iloc[-1]
price_diff = df['price'].iloc[-1] - df['price'].iloc[-2]
price_pct = (price_diff / df['price'].iloc[-2]) * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("即時報價 (USD)", f"${curr_price:,.2f}", f"{price_pct:+.2f}%")
with col2:
    st.metric("RSI 指標 (14D)", f"{df['rsi'].iloc[-1]:.1f}")
with col3:
    vol_status = "高" if df['volatility'].iloc[-1] > df['volatility'].mean() else "低"
    st.metric("市場波動率", vol_status)
with col4:
    st.metric("DXY 美元權重", f"{df['dxy'].iloc[-1]:.2f}")

# ==============================
# 5️⃣ AI 核心預測核心 (Random Forest + Pct Change)
# ==============================
features = ['price', 'dxy', 'spy', 'vix', 'ma20', 'rsi', 'volatility']
X = df[features]
y = df['target_return']

# 訓練模型
train_idx = int(len(df) * 0.8)
model = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X[:train_idx], y[:train_idx])

# 預測未來
latest_feat = X.tail(1)
pred_return = model.predict(latest_feat)[0]
pred_next_price = curr_price * (1 + pred_return)
conf_score = model.score(X[train_idx:], y[train_idx:]) # 使用 R^2 作為信心參考

# 計算勝率
df['pred_ret'] = model.predict(X)
df['correct_dir'] = np.sign(df['pred_ret']) == np.sign(df['target_return'])
win_rate = df['correct_dir'].tail(backtest_days).mean() * 100

st.markdown("---")
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("🤖 AI 未來走勢預測")
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = pred_next_price,
        delta = {'reference': curr_price, 'relative': True, 'position': "top"},
        title = {'text': f"明日 {asset} 預測價"},
        gauge = {
            'axis': {'range': [curr_price*0.97, curr_price*1.03]},
            'bar': {'color': "#FFD700"},
            'steps': [{'range': [0, curr_price], 'color': "#1e212b"}]
        }
    ))
    fig_pred.update_layout(height=300, margin=dict(t=50, b=0), paper_bgcolor="#0e1117", font={'color': "white"})
    st.plotly_chart(fig_pred, use_container_width=True)

with c2:
    st.subheader("📊 系統信心與勝率")
    st.metric("方向預測勝率", f"{win_rate:.1f}%")
    st.progress(win_rate / 100)
    st.write(f"模型信心 (R²): {conf_score:.2f}")
    st.caption("※ 信心高於 0.1 代表模型具有參考價值")

# ==============================
# 6️⃣ 買賣建議與風控 (實盤核心)
# ==============================
st.subheader("💡 實盤交易策略建議")

# 倉位計算
stop_loss_dist = curr_price * 0.015  # 假設停損設在 1.5% 處
position_size = (total_capital * (risk_pct/100)) / stop_loss_dist
position_size = round(position_size, 2)

advice_col1, advice_col2 = st.columns(2)

with advice_col1:
    if pred_return > 0.003 and win_rate > 52:
        st.success("✅ **建議方向：看多 (LONG)**")
        st.write(f"👉 建議入場：當前價格或回測 ${curr_price*0.998:.2f}")
        st.write(f"🛑 建議停損：${curr_price - stop_loss_dist:.2f}")
    elif pred_return < -0.003 and win_rate > 52:
        st.error("🔻 **建議方向：看空 (SHORT)**")
        st.write(f"👉 建議入場：當前價格或反彈 ${curr_price*1.002:.2f}")
        st.write(f"🛑 建議停損：${curr_price + stop_loss_dist:.2f}")
    else:
        st.warning("⚖️ **建議方向：觀望 (NEUTRAL)**")
        st.write("目前趨勢不明或勝率不足，建議等待訊號。")

with advice_col2:
    st.info(f"📏 **風控倉位建議**")
    st.write(f"建議持倉量：**{position_size}** 盎司 / 口")
    st.write(f"風險本金消耗：${total_capital * (risk_pct/100):.2f}")
    st.caption("依據您的單筆風險百分比計算，請嚴格執行停損。")

# ==============================
# 7️⃣ 視覺化 K 線與均線
# ==============================
st.subheader("📈 歷史走勢與技術矩陣")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df.index[-120:], y=df['price'].tail(120), name='Price', line=dict(color='#FFD700', width=2)))
fig_hist.add_trace(go.Scatter(x=df.index[-120:], y=df['ma20'].tail(120), name='MA20', line=dict(color='#00BFFF', dash='dot')))
fig_hist.update_layout(template="plotly_dark", height=450, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
st.plotly_chart(fig_hist, use_container_width=True)
