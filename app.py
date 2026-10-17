import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="Pocket AI Agent — Demo", page_icon="📊", layout="wide")
st.title("📊 Pocket AI Agent")
st.caption("Demo analiz prototipi · Pocket Option hesabına bağlanmaz · Gerçek işlem açmaz")

st.warning("Bu sürüm yalnızca analiz ve sanal işlem simülasyonu içindir. Gerçek hesap bağlantısı veya kâr garantisi yoktur.")

with st.sidebar:
    st.header("Risk ayarları")
    trade_amount = st.number_input("Sanal işlem tutarı (USD)", min_value=1.0, max_value=100.0, value=1.0, step=1.0)
    payout = st.slider("Varsayılan ödeme oranı (%)", min_value=50, max_value=95, value=85) / 100
    daily_loss_limit = st.number_input("Günlük sanal zarar durdurma (USD)", min_value=1.0, max_value=500.0, value=10.0, step=1.0)
    threshold = st.slider("Model güven eşiği (%)", min_value=50, max_value=95, value=65) / 100

def indicators(df):
    d = df.copy()
    d["ema_fast"] = d["close"].ewm(span=12, adjust=False).mean()
    d["ema_slow"] = d["close"].ewm(span=26, adjust=False).mean()
    delta = d["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))
    d["returns"] = d["close"].pct_change()
    d["volatility"] = d["returns"].rolling(14).std()
    d["future_up"] = (d["close"].shift(-1) > d["close"]).astype(int)
    return d

def sample_data(n=600, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0001, 0.004, n)
    close = 100 * np.exp(np.cumsum(rets))
    times = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="min")
    return pd.DataFrame({"timestamp": times, "close": close})

uploaded = st.file_uploader("Fiyat verisi CSV yükle (timestamp, close)", type=["csv"])
if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        if "close" not in raw.columns:
            st.error("CSV dosyasında 'close' sütunu bulunamadı.")
            st.stop()
        if "timestamp" not in raw.columns:
            raw["timestamp"] = np.arange(len(raw))
        raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
        raw = raw.dropna(subset=["close"]).reset_index(drop=True)
        if len(raw) < 80:
            st.error("Model denemesi için en az 80 fiyat satırı gerekiyor.")
            st.stop()
        df = raw[["timestamp", "close"]].copy()
        source_label = "Yüklenen CSV"
    except Exception as e:
        st.error(f"CSV okunamadı: {e}")
        st.stop()
else:
    df = sample_data()
    source_label = "Sentetik örnek veri (gerçek piyasa verisi değil)"

d = indicators(df).dropna().reset_index(drop=True)
feature_cols = ["ema_fast", "ema_slow", "rsi", "returns", "volatility"]
# Avoid using final row where the next-period target is unknown.
trainable = d.iloc[:-1].copy()
if len(trainable) < 50:
    st.error("Yeterli veri yok.")
    st.stop()

split = int(len(trainable) * 0.8)
X_train = trainable.loc[:split-1, feature_cols]
y_train = trainable.loc[:split-1, "future_up"]
X_test = trainable.loc[split:, feature_cols]
y_test = trainable.loc[split:, "future_up"]

model = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=5, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred) if len(y_test) else float("nan")
latest = d.iloc[-1]
latest_features = d.iloc[[-1]][feature_cols]
proba_up = float(model.predict_proba(latest_features)[0][list(model.classes_).index(1)]) if 1 in model.classes_ else 0.5
confidence = max(proba_up, 1-proba_up)
direction = "CALL / Yukarı" if proba_up >= 0.5 else "PUT / Aşağı"
signal = direction if confidence >= threshold else "İŞLEM YOK — güven eşiği altında"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Veri kaynağı", source_label)
c2.metric("Zaman sıralı test doğruluğu", f"{acc*100:.1f}%")
c3.metric("Son tahmin", direction)
c4.metric("Model güven skoru", f"{confidence*100:.1f}%")

st.subheader("Son durum")
st.write(f"**Karar:** {signal}")
st.write(f"**Son kapanış:** {latest['close']:.5f}")
st.write(f"**RSI:** {latest['rsi']:.1f} · **EMA12:** {latest['ema_fast']:.5f} · **EMA26:** {latest['ema_slow']:.5f}")
st.line_chart(d.set_index("timestamp")["close"])

st.subheader("Demo işlem simülasyonu")
st.write("Bu simülasyon yalnızca son test dilimindeki model tahminlerini kullanır. İkili opsiyon ödemesini basitleştirilmiş biçimde varsayar; gerçek işlem maliyetleri ve platform koşulları farklı olabilir.")
test_proba = model.predict_proba(X_test)
classes = list(model.classes_)
up_idx = classes.index(1) if 1 in classes else None
if up_idx is not None:
    test_up = test_proba[:, up_idx]
    test_conf = np.maximum(test_up, 1-test_up)
    test_dir = (test_up >= 0.5).astype(int)
    take = test_conf >= threshold
    outcomes = (test_dir[take] == y_test.to_numpy()[take])
    ntrades = int(take.sum())
    wins = int(outcomes.sum()) if ntrades else 0
    losses = ntrades - wins
    # binary payout approximation
    pnl = wins * trade_amount * payout - losses * trade_amount
    st.metric("Sanal işlem sinyali sayısı", ntrades)
    st.metric("Kazanan / Kaybeden", f"{wins} / {losses}")
    st.metric("Basitleştirilmiş sanal net sonuç", f"{pnl:+.2f} USD")
    if pnl <= -daily_loss_limit:
        st.error("Sanal sonuç, seçilen zarar sınırına ulaştı/aştı. Simülasyonda durma kuralı tetiklenirdi.")
    st.caption("Geçmiş test sonucu gelecekteki performansı garanti etmez. Çok az sinyal oluşması da modelin avantajlı olduğunu kanıtlamaz.")

with st.expander("Test ayrıntıları"):
    st.write("Özellikler:", feature_cols)
    st.write("Eğitim satırı:", len(X_train), "· Test satırı:", len(X_test))
    st.write("Eşik:", f"{threshold*100:.0f}%")
    st.write("Ödeme oranı:", f"{payout*100:.0f}%")
    st.dataframe(d.tail(20), use_container_width=True)
