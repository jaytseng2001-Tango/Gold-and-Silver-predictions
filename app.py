import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

# 頁面配置
st.set_page_config(page_title="2026 金銀 AI 領航員 PRO+", layout="wide")

# ==============================
# 強化版數據下載（修復資料不足問題）
# ==============================
@st.cache_data(ttl=3600)
def get_data():
    # 使用 10 年數據確保樣本數足夠
    tickers = ["GC=F", "SI=F", "DX-Y.NYB", "^GSPC", "^VIX"]
    df = yf.download(
        tickers, 
        period="10y", 
        interval="1d", 
        auto_adjust=True, 
        progress=False
    )
    
    # 處理 yfinance 可能返回的 MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    
    # 【關鍵修正】先進行前後填補，避免因單一欄位缺失導致整行被刪除
    # 金融數據中，假日或開盤時間差異常導致空值，ffill (往前填補) 是標準做法
    df = df.ffill().bfill()
    
    return df

raw_data = get_data()

# ==============================
# 側邊欄
# ==============================
target_label = st.sidebar.selectbox("監測資產", ["黃金 (GC=F)", "白銀 (SI=F)"])
target = "GC=F" if "黃金" in target_label else "SI=F"

# ==============================
# 特徵工程
# ==============================
df = pd.DataFrame(index=raw_data.index)
df['price'] = raw_data[target]
df['usd'] = raw_data['DX-Y.NYB']
df['stock'] = raw_data['^GSPC']
df['vix'] = raw_data['^VIX']

# 技術指標
df['ma20'] = df['price'].rolling(20).mean()
df['ma50'] = df['price'].rolling(50).mean()
df['volatility'] = df['price'].pct_change().rolling(20).std()

# RSI 計算
delta = df['price'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss.replace(0, np.nan)
df['rsi'] = 100 - (100 / (1 + rs))

# 預測目標：隔日價格
df['target'] = df['price'].shift(-1)

# 清理最終數據集
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# ==============================
# 檢查點：動態調整門檻
# ==============================
if len(df) < 20: # 降到最低門檻以確保能跑出結果
    st.error(f"有效數據僅 {len(df)} 筆，不足以支撐 AI 模型。")
    st.info("請確認網路連線，或稍後再試（Yahoo Finance 資料更新中可能導致短暫缺失）。")
    st.stop()

# ==============================
# 模型訓練
# ==============================
features = ['price', 'usd', 'stock', 'vix', 'ma20', 'ma50', 'rsi', 'volatility']
# 確保特徵完整
df = df.dropna(subset=features + ['target'])

train_size = int(len(df) * 0.8)
train = df[:train_size]

model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=8, 
    random_state=42, 
    n_jobs=-1
)

model.fit(train[features], train['target'])

# ==============================
# 預測與 UI
# ==============================
latest = df[features].tail(1)
pred_1d = model.predict(latest)[0]
curr_price = df['price'].iloc[-1]
diff_pct = (pred_1d - curr_price) / curr_price * 100

st.title("🏆 2026 金銀 AI 領航員 PRO+")
st.success(f"數據載入成功！已使用 {len(df)} 天歷史樣本進行訓練。")

col1, col2, col3, col4 = st.columns(4)
col1.metric("當前價格", f"${curr_price:,.2f}")
col2.metric("AI 明日預測", f"${pred_1d:,.2f}", f"{diff_pct:+.2f}%")
col3.metric("RSI", f"{df['rsi'].iloc[-1]:.1f}")
col4.metric("VIX (恐慌指數)", f"{df['vix'].iloc[-1]:.1f}")

# 價格走勢圖
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df.index[-120:], 
    y=df['price'].tail(120), 
    fill='tozeroy', 
    name='歷史價格', 
    line=dict(color='#FFD700')
))
fig.add_trace(go.Scatter(
    x=df.index[-120:], 
    y=df['ma20'].tail(120), 
    name='20日均線', 
    line=dict(color='#00BFFF')
))

fig.update_layout(
    template="plotly_dark", 
    height=450,
    margin=dict(l=20, r=20, t=50, b=20),
    xaxis_title="日期",
    yaxis_title="價格 (USD)"
)
st.plotly_chart(fig, use_container_width=True)
