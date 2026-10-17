import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Pocket AI Agent — Live M1/M5", page_icon="📊", layout="wide")
st.title("📊 Pocket AI Agent")
st.caption("Twelve Data fiyat verisiyle M1/M5 demo analiz · Pocket Option'a bağlanmaz · İşlem açmaz")
st.warning("Bu uygulama Twelve Data üzerinden normal piyasa fiyatlarını okur. Pocket Option/OTC fiyatlarıyla birebir aynı olmayabilir. İşlem açmaz.")

SYMBOLS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY"]

with st.sidebar:
    st.header("Piyasa ve sinyal ayarları")
    symbol = st.selectbox("Forex paritesi", SYMBOLS)
    outputsize = st.select_slider("İndirilecek mum sayısı", options=[100, 150, 200, 300], value=200)
    threshold = st.slider("Model güven eşiği (%)", 50, 95, 65) / 100
    st.caption("Yenileme iki API isteği yapabilir. Ücretsiz API limitlerini gözetin.")

try:
    API_KEY = st.secrets["TWELVE_DATA_API_KEY"]
except Exception:
    API_KEY = ""

@st.cache_data(ttl=60, show_spinner=False)
def fetch_time_series(symbol, interval, outputsize, api_key):
    response = requests.get(
        "https://api.twelvedata.com/time_series",
        params={"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": api_key, "format": "JSON"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") == "error" or "values" not in payload:
        raise RuntimeError(f"Twelve Data yanıtı ({payload.get('code', 'bilinmiyor')}): {payload.get('message', 'Mum verisi bulunamadı.')}")
    frame = pd.DataFrame(payload["values"])
    if not {"datetime", "close"}.issubset(frame.columns):
        raise RuntimeError("API yanıtında datetime/close alanları bulunamadı.")
    frame["timestamp"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
    if len(frame) < 80:
        raise RuntimeError(f"Yeterli mum gelmedi: {len(frame)}. En az 80 kayıt gerekiyor.")
    return frame[["timestamp", "close"]]

def analyze(frame, threshold):
    d = frame.copy().sort_values("timestamp").reset_index(drop=True)
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
    d = d.dropna().reset_index(drop=True)
    if len(d) < 80:
        raise RuntimeError("Yeterli kullanılabilir mum yok.")
    trainable = d.iloc[:-1].copy()
    split = int(len(trainable) * 0.8)
    if split < 40 or len(trainable) - split < 10:
        raise RuntimeError("Eğitim/test verisi yetersiz.")
    features = ["ema_fast", "ema_slow", "rsi", "returns", "volatility"]
    X_train, y_train = trainable.iloc[:split][features], trainable.iloc[:split]["future_up"].astype(int)
    X_test, y_test = trainable.iloc[split:][features], trainable.iloc[split:]["future_up"].astype(int)
    model = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=5, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test)) if len(y_test) else float("nan")
    classes = list(model.classes_)
    probs = model.predict_proba(d.iloc[[-1]][features])[0]
    prob_up = float(probs[classes.index(1)]) if 1 in classes else 0.5
    confidence = max(prob_up, 1 - prob_up)
    direction = "CALL / Yukarı" if prob_up >= 0.5 else "PUT / Aşağı"
    return {
        "data": d, "accuracy": accuracy, "direction": direction, "confidence": confidence,
        "signal": direction if confidence >= threshold else "İŞLEM YOK — güven eşiği altında",
        "close": float(d.iloc[-1]["close"]), "rsi": float(d.iloc[-1]["rsi"]),
        "ema_fast": float(d.iloc[-1]["ema_fast"]), "ema_slow": float(d.iloc[-1]["ema_slow"]),
    }

if not API_KEY:
    st.error("Twelve Data API anahtarı bulunamadı. Streamlit → Manage app → Settings → Secrets bölümünde TWELVE_DATA_API_KEY değişkenini kontrol edin.")
    st.code('TWELVE_DATA_API_KEY = "kendi_api_anahtariniz"', language="toml")
    st.stop()

if st.button("Fiyat verisini yenile"):
    st.cache_data.clear()

try:
    with st.spinner(f"{symbol} için M1 ve M5 fiyatları alınıyor..."):
        m1_data = fetch_time_series(symbol, "1min", outputsize, API_KEY)
        m5_data = fetch_time_series(symbol, "5min", outputsize, API_KEY)
except requests.exceptions.Timeout:
    st.error("Twelve Data isteği zaman aşımına uğradı. Daha sonra yeniden deneyin.")
    st.stop()
except requests.exceptions.RequestException as exc:
    st.error(f"Ağ bağlantısı hatası: {exc}")
    st.stop()
except RuntimeError as exc:
    st.error(str(exc))
    st.info("API anahtarı, sembol erişimi, istek limiti ve ücretsiz plan kısıtlarını kontrol edin.")
    st.stop()
except Exception as exc:
    st.error(f"Beklenmeyen hata: {exc}")
    st.stop()

st.success(f"Veri alındı: {symbol} · Son M1 mumu: {m1_data['timestamp'].iloc[-1]} · Son M5 mumu: {m5_data['timestamp'].iloc[-1]}")
results = {}
cols = st.columns(2)
for col, timeframe, frame in [(cols[0], "M1", m1_data), (cols[1], "M5", m5_data)]:
    with col:
        st.subheader(f"⏱️ {timeframe} Analizi")
        try:
            result = analyze(frame, threshold)
            results[timeframe] = result
        except Exception as exc:
            st.error(f"{timeframe} analizi yapılamadı: {exc}")
            continue
        st.metric("Geçmiş test doğruluğu", f"{result['accuracy'] * 100:.1f}%")
        st.metric("Son yön tahmini", result["direction"])
        st.metric("Model güven skoru", f"{result['confidence'] * 100:.1f}%")
        st.write(f"**Karar:** {result['signal']}")
        st.write(f"**Son kapanış:** {result['close']:.5f}")
        st.write(f"**RSI:** {result['rsi']:.1f} · **EMA12:** {result['ema_fast']:.5f} · **EMA26:** {result['ema_slow']:.5f}")
        st.line_chart(result["data"].set_index("timestamp")["close"])

st.subheader("M1 / M5 karşılaştırması")
if len(results) == 2:
    m1, m5 = results["M1"], results["M5"]
    if m1["direction"] == m5["direction"]:
        st.info(f"M1 ve M5 aynı yönü gösteriyor: {m1['direction']}. Bu, tahminin doğru olacağını garanti etmez.")
    else:
        st.warning("M1 ve M5 yönleri farklı. Sinyaller çelişiyor; temkinli değerlendirin.")
    st.dataframe(pd.DataFrame([
        {"Zaman dilimi": tf, "Yön": r["direction"], "Güven (%)": round(r["confidence"] * 100, 1), "Karar": r["signal"], "Test doğruluğu (%)": round(r["accuracy"] * 100, 1)}
        for tf, r in results.items()
    ]), hide_index=True, use_container_width=True)

with st.expander("Veri ve model ayrıntıları"):
    st.write("Kaynak: Twelve Data API")
    st.write("Sembol:", symbol)
    st.write("M1 kayıt sayısı:", len(m1_data), "· M5 kayıt sayısı:", len(m5_data))
    st.write("Güven eşiği:", f"{threshold * 100:.0f}%")
    st.caption("Test doğruluğu indirilen geçmiş veri aralığındaki basit zaman sıralı testtir; gelecekteki performansı garanti etmez.")

st.caption("Normal piyasa verileri Pocket Option/OTC fiyatlarından farklılık gösterebilir. Prototip yatırım tavsiyesi değildir ve otomatik işlem yapmaz.")
