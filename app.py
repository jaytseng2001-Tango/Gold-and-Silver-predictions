import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

st.set_page_config(page_title="2026 金銀 AI 領航員 PRO+", layout="wide")

# ==============================
# 強化版數據下載（防止炸裂）
# ==============================
@st.cache_data(ttl=3600)
def get_data():
    tickers = ["GC=F", "SI=F", "DX-Y.NYB", "^GSPC", "^VIX"]
    df = yf.download(
        tickers,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    # 安全取得 Close
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    else:
        df = df

    df = df.ffill()

    # 確保關鍵欄位存在
    required = ["GC=F", "SI=F", "DX-Y.NYB", "^GSPC", "^VIX"]
    for col in required:
        if col not in df.columns:
            st.error(f"缺少資料欄位: {col}")
            st.stop()

    return df.dropna()

raw_data = get_data()

# ==============================
# 側邊欄
# ==============================
target_label = st.sidebar.selectbox("監測資產", ["黃金 (GC=F)", "白銀 (SI=F)"])
target = "GC=F" if "黃金" in target_label else "SI=F"

# ==============================
# 特徵工程（防止 RSI 無限值）
# ==============================
df = pd.DataFrame()
df['price'] = raw_data[target]
df['usd'] = raw_data['DX-Y.NYB']
df['stock'] = raw_data['^GSPC']
df['vix'] = raw_data['^VIX']

df['ma20'] = df['price'].rolling(20).mean()
df['ma50'] = df['price'].rolling(50).mean()
df['volatility'] = df['price'].pct_change().rolling(20).std()

delta = df['price'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()

rs = gain / (loss.replace(0, np.nan))
df['rsi'] = 100 - (100 / (1 + rs))

df['target'] = df['price'].shift(-1)

df = df.replace([np.inf, -np.inf], np.nan).dropna()

if len(df) < 100:
    st.error("資料不足，無法建立模型")
    st.stop()

# ==============================
# 模型（減少記憶體消耗）
# ==============================
features = ['price','usd','stock','vix','ma20','ma50','rsi','volatility']

train_size = int(len(df) * 0.8)
train = df[:train_size]

model = RandomForestRegressor(
    n_estimators=200,  # 降低避免爆記憶體
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(train[features], train['target'])

# ==============================
# 預測
# ==============================
latest = df[features].tail(1)
pred_1d = model.predict(latest)[0]

curr_price = df['price'].iloc[-1]
diff_pct = (pred_1d - curr_price) / curr_price * 100

current_rsi = df['rsi'].iloc[-1]
current_vix = df['vix'].iloc[-1]

# ==============================
# UI
# ==============================
st.title("🏆 2026 金銀 AI 領航員 PRO+")

col1, col2, col3, col4 = st.columns(4)
col1.metric("當前價格", f"${curr_price:,.2f}")
col2.metric("AI 明日預測", f"${pred_1d:,.2f}", f"{diff_pct:+.2f}%")
col3.metric("RSI", f"{current_rsi:.1f}")
col4.metric("VIX", f"{current_vix:.1f}")

# 圖表
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df.index[-100:],
    y=df['price'].tail(100),
    fill='tozeroy',
    name='價格'
))
fig.add_trace(go.Scatter(
    x=df.index[-100:],
    y=df['ma20'].tail(100),
    name='MA20'
))

fig.update_layout(template="plotly_dark", height=350)
st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

st.set_page_config(page_title="2026 金銀 AI 領航員 PRO+", layout="wide")

# ==============================
# 強化版數據下載（修復資料不足問題）
# ==============================
@st.cache_data(ttl=3600)
def get_data():
    # 增加資料年限到 10 年，增加樣本數
    tickers = ["GC=F", "SI=F", "DX-Y.NYB", "^GSPC", "^VIX"]
    df = yf.download(
        tickers, 
        period="10y",  # 從 5y 改為 10y
        interval="1d", 
        auto_adjust=True, 
        progress=False
    )
    
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    
    # 核心修復：先填補空值，再刪除完全沒資料的行
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
df = pd.DataFrame()
df['price'] = raw_data[target]
df['usd'] = raw_data['DX-Y.NYB']
df['stock'] = raw_data['^GSPC']
df['vix'] = raw_data['^VIX']

# 計算技術指標
df['ma20'] = df['price'].rolling(20).mean()
df['ma50'] = df['price'].rolling(50).mean()
df['volatility'] = df['price'].pct_change().rolling(20).std()

# RSI 修復
delta = df['price'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / (loss.replace(0, np.nan))
df['rsi'] = 100 - (100 / (1 + rs))

df['target'] = df['price'].shift(-1)

# 清理數據
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# ==============================
# 檢查點：如果還是不足，顯示目前的資料量
# ==============================
if len(df) < 50:
    st.error(f"目前有效資料僅有 {len(df)} 筆，請嘗試更換瀏覽器或稍後再試。")
    st.info("這通常是 Yahoo Finance 暫時限制存取，建議等待 10 分鐘自動重試。")
    st.stop()

# ==============================
# 模型訓練
# ==============================
features = ['price','usd','stock','vix','ma20','ma50','rsi','volatility']
train_size = int(len(df) * 0.8)
train = df[:train_size]

model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
model.fit(train[features], train['target'])

# ==============================
# 預測與 UI
# ==============================
latest = df[features].tail(1)
pred_1d = model.predict(latest)[0]
curr_price = df['price'].iloc[-1]
diff_pct = (pred_1d - curr_price) / curr_price * 100

st.title("🏆 2026 金銀 AI 領航員 PRO+")
st.write(f"系統已成功載入 {len(df)} 天的歷史數據進行分析")

col1, col2, col3, col4 = st.columns(4)
col1.metric("當前價格", f"${curr_price:,.2f}")
col2.metric("AI 明日預測", f"${pred_1d:,.2f}", f"{diff_pct:+.2f}%")
col3.metric("RSI", f"{df['rsi'].iloc[-1]:.1f}")
col4.metric("VIX", f"{df['vix'].iloc[-1]:.1f}")

# 畫圖
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[-120:], y=df['price'].tail(120), fill='tozeroy', name='價格', line=dict(color='#FFD700')))
fig.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig, use_container_width=True)
