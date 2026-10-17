
    
  
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Pocket AI Agent", page_icon="📈", layout="wide")
st.markdown('''<style>
.stApp{background:linear-gradient(145deg,#090b0f,#11151c 60%,#0b0d11);color:#f1f3f5}
[data-testid="stSidebar"]{background:#11151b;border-right:1px solid #39332a}
h1,h2,h3{color:#e8c56a!important}
[data-testid="stMetric"]{background:#151a21;border:1px solid #39332a;padding:14px;border-radius:12px}
.stButton>button{background:#d6ad4f;color:#111318;border:0;border-radius:9px;font-weight:700}
</style>''', unsafe_allow_html=True)
st.title("POCKET AI AGENT")
st.caption("Twelve Data · M1 / M5 sinyal analizi · Premium Market Lab")
st.warning("Bu uygulama analiz prototipidir; Pocket Option'a bağlanmaz ve işlem açmaz. Normal forex fiyatları Pocket Option/OTC fiyatlarından farklı olabilir. Sinyaller kâr garantisi değildir; önce demo ortamında test edin.")

SYMBOLS = ["EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF","NZD/USD","EUR/GBP","EUR/JPY","GBP/JPY"]
TZ = ZoneInfo("Europe/Istanbul")
with st.sidebar:
    st.header("Piyasa ayarları")
    symbol = st.selectbox("Forex paritesi", SYMBOLS)
    outputsize = st.select_slider("Geçmiş mum sayısı", options=[150,200,300,500], value=300)
    threshold = st.slider("Minimum model yön skoru (%)",50,80,58)/100
    min_votes = st.select_slider("Gerekli teknik uyum", options=[1,2,3], value=2)
    refresh = st.button("↻ Veriyi yenile", use_container_width=True)
    st.caption("Her yenileme M1 ve M5 için API isteği yapar.")
try:
    API_KEY = st.secrets["TWELVE_DATA_API_KEY"]
except Exception:
    API_KEY = ""
if not API_KEY:
    st.error("API anahtarı bulunamadı. Streamlit → Manage app → Settings → Secrets içinde TWELVE_DATA_API_KEY tanımlı olmalı.")
    st.code('TWELVE_DATA_API_KEY = "kendi_anahtariniz"', language="toml")
    st.stop()
if refresh:
    st.cache_data.clear()

@st.cache_data(ttl=45, show_spinner=False)
def fetch_time_series(pair, interval, size, api_key):
    response = requests.get("https://api.twelvedata.com/time_series", params={"symbol":pair,"interval":interval,"outputsize":size,"timezone":"Europe/Istanbul","apikey":api_key,"format":"JSON"}, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status")=="error" or "values" not in payload:
        raise RuntimeError(f"Twelve Data ({payload.get('code','hata')}): {payload.get('message','Mum verisi alınamadı.')}")
    frame = pd.DataFrame(payload["values"])
    required = {"datetime", "open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        raise RuntimeError(
            "Twelve Data yanıtında OHLC mum alanları eksik. "
            "Grafik için datetime, open, high, low ve close alanları gerekiyor."
        )
    frame["timestamp"] = pd.to_datetime(frame["datetime"], errors="coerce")
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    if len(frame) < 100:
        raise RuntimeError(f"Yeterli mum gelmedi: {len(frame)}. En az 100 kayıt gerekli.")
    # A flat series can indicate a feed/symbol issue; surface it rather than disguising it.
    if frame["close"].tail(30).nunique() < 3:
        st.warning("Son 30 mumda fiyat neredeyse hiç değişmiyor. Sembol/veri akışını kontrol edin; bu veriyle sinyal güvenilir kabul edilmemeli.")
    return frame[["timestamp", "open", "high", "low", "close"]]

FEATURES=["ema12","ema26","ema50","rsi","macd","macd_signal","return1","volatility"]
def indicators(frame):
    d=frame.copy().sort_values("timestamp").reset_index(drop=True); c=d.close
    d["ema12"]=c.ewm(span=12,adjust=False).mean(); d["ema26"]=c.ewm(span=26,adjust=False).mean(); d["ema50"]=c.ewm(span=50,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"]=100-(100/(1+gain/loss.replace(0,np.nan)))
    d["macd"]=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); d["macd_signal"]=d.macd.ewm(span=9,adjust=False).mean()
    d["return1"]=c.pct_change(); d["volatility"]=d.return1.rolling(14).std()
    d["future_up"]=(c.shift(-1)>c).astype(float); d.loc[d.index[-1],"future_up"]=np.nan
    return d

def analyze(frame, threshold, min_votes):
    usable=indicators(frame).dropna(subset=FEATURES).reset_index(drop=True)
    labeled=usable.dropna(subset=["future_up"]).copy().reset_index(drop=True)
    if len(labeled)<100: raise RuntimeError("Model için yeterli kullanılabilir mum yok.")
    split=int(len(labeled)*.8)
    if split<50 or len(labeled)-split<15: raise RuntimeError("Zaman sıralı eğitim/test örneği yetersiz.")
    model=RandomForestClassifier(n_estimators=250,max_depth=5,min_samples_leaf=8,max_features="sqrt",random_state=42,class_weight="balanced")
    model.fit(labeled.loc[:split-1,FEATURES],labeled.loc[:split-1,"future_up"].astype(int))
    test=labeled.loc[split:]; accuracy=accuracy_score(test.future_up.astype(int),model.predict(test[FEATURES]))
    latest=usable.iloc[[-1]]; classes=list(model.classes_); probs=model.predict_proba(latest[FEATURES])[0]
    p_up=float(probs[classes.index(1)]) if 1 in classes else .5; confidence=max(p_up,1-p_up); model_dir="Yükseliş" if p_up>=.5 else "Düşüş"
    r=latest.iloc[0]
    votes=[1 if r.ema12>r.ema26 else -1,1 if r.ema26>r.ema50 else -1,1 if r.macd>r.macd_signal else -1]
    votes.append(1 if r.rsi>=55 else -1 if r.rsi<=45 else 0)
    up=sum(v==1 for v in votes); down=sum(v==-1 for v in votes); tech_dir="Yükseliş" if up>down else "Düşüş" if down>up else "BEKLE"
    support=up if model_dir=="Yükseliş" else down
    if confidence<threshold: signal,reason="BEKLE","Model yön skoru eşik altında."
    elif tech_dir=="BEKLE": signal,reason="BEKLE","Teknik göstergeler net bir yönde uzlaşmıyor."
    elif model_dir!=tech_dir: signal,reason="BEKLE","Model yönü ile teknik göstergeler çelişiyor."
    elif support<min_votes: signal,reason="BEKLE","Yeterli teknik gösterge aynı yönü desteklemiyor."
    else: signal,reason=model_dir,"Model yönü ve teknik göstergeler aynı yönde; bu garanti değildir."
    return {"data":usable,"accuracy":float(accuracy),"model_direction":model_dir,"confidence":confidence,"technical_direction":tech_dir,"signal":signal,"reason":reason,"close":float(r.close),"rsi":float(r.rsi),"ema12":float(r.ema12),"ema26":float(r.ema26),"ema50":float(r.ema50),"macd":float(r.macd),"timestamp":r.timestamp}

try:
    with st.spinner(f"{symbol} için M1 ve M5 verileri alınıyor..."):
        m1_data=fetch_time_series(symbol,"1min",outputsize,API_KEY); m5_data=fetch_time_series(symbol,"5min",outputsize,API_KEY)
except requests.exceptions.Timeout:
    st.error("Twelve Data isteği zaman aşımına uğradı. Biraz sonra deneyin."); st.stop()
except requests.exceptions.RequestException as exc:
    st.error(f"Ağ bağlantısı hatası: {exc}"); st.stop()
except RuntimeError as exc:
    st.error(str(exc)); st.info("API anahtarı, sembol erişimi ve kullanım limitini kontrol edin."); st.stop()

now=datetime.now(TZ); now_naive=now.replace(tzinfo=None)
def age(ts): return (now_naive-ts.to_pydatetime().replace(tzinfo=None)).total_seconds()/60
age1,age5=age(m1_data.timestamp.iloc[-1]),age(m5_data.timestamp.iloc[-1])
st.caption(f"İstanbul saati: {now:%d.%m.%Y %H:%M:%S} · Kaynak: Twelve Data REST · Sürekli WebSocket akışı değildir")
if age1>5 or age5>10: st.warning(f"Veri eski görünüyor: M1 son mum {m1_data.timestamp.iloc[-1]:%H:%M}, M5 son mum {m5_data.timestamp.iloc[-1]:%H:%M}. Piyasa kapalı olabilir veya veri gecikiyor olabilir; sinyale güvenmeden önce kontrol edin.")
elif age1 < -1 or age5 < -1: st.warning("API zaman damgası İstanbul saatinden ileride görünüyor. Saat dilimini kontrol etmek gerekiyor.")
else: st.success(f"Veri alındı · M1: {m1_data.timestamp.iloc[-1]:%d.%m %H:%M} · M5: {m5_data.timestamp.iloc[-1]:%d.%m %H:%M} (İstanbul)")

results={}; cols=st.columns(2)
for col,tf,frame in [(cols[0],"M1",m1_data),(cols[1],"M5",m5_data)]:
    with col:

