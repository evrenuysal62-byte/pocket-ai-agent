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

FEATURES=["ema12_gap","ema26_gap","ema50_gap","rsi","macd_gap","macd_signal_gap","return1","momentum3","volatility","range_pct","body_pct"]
def indicators(frame):
    d=frame.copy().sort_values("timestamp").reset_index(drop=True); c=d.close
    d["ema12"]=c.ewm(span=12,adjust=False).mean(); d["ema26"]=c.ewm(span=26,adjust=False).mean(); d["ema50"]=c.ewm(span=50,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan)
    d["rsi"]=100-(100/(1+rs))
    d.loc[(loss==0)&(gain>0),"rsi"]=100
    d["macd"]=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); d["macd_signal"]=d.macd.ewm(span=9,adjust=False).mean()
    # Fiyat seviyesinden bağımsız, farklı paritelerde daha karşılaştırılabilir özellikler.
    safe_close=c.replace(0,np.nan)
    d["ema12_gap"]=(c-d.ema12)/safe_close*10000
    d["ema26_gap"]=(c-d.ema26)/safe_close*10000
    d["ema50_gap"]=(c-d.ema50)/safe_close*10000
    d["macd_gap"]=(d.macd-d.macd_signal)/safe_close*10000
    d["macd_signal_gap"]=d.macd_signal.diff()
    d["return1"]=c.pct_change()*100
    d["momentum3"]=c.pct_change(3)*100
    d["volatility"]=d.return1.rolling(14).std()
    d["range_pct"]=(d.high-d.low)/safe_close*100
    d["body_pct"]=(d.close-d.open)/safe_close*100
    d["future_up"]=(c.shift(-1)>c).astype(float); d.loc[d.index[-1],"future_up"]=np.nan
    return d

def analyze(frame, threshold, min_votes):
    all_data=indicators(frame)
    usable=all_data.dropna(subset=FEATURES+['ema12','ema26','ema50','macd','macd_signal','rsi']).reset_index(drop=True)
    labeled=usable.dropna(subset=["future_up"]).copy().reset_index(drop=True)
    if len(labeled)<100: raise RuntimeError("Model için yeterli kullanılabilir mum yok.")
    split=int(len(labeled)*.8)
    if split<50 or len(labeled)-split<15: raise RuntimeError("Zaman sıralı eğitim/test örneği yetersiz.")
    model=RandomForestClassifier(n_estimators=350,max_depth=6,min_samples_leaf=10,max_features="sqrt",random_state=42,class_weight="balanced_subsample")
    model.fit(labeled.loc[:split-1,FEATURES],labeled.loc[:split-1,"future_up"].astype(int))
    test=labeled.loc[split:]
    accuracy=accuracy_score(test.future_up.astype(int),model.predict(test[FEATURES]))
    latest=usable.iloc[[-1]]; classes=list(model.classes_); probs=model.predict_proba(latest[FEATURES])[0]
    p_up=float(probs[classes.index(1)]) if 1 in classes else .5
    confidence=max(p_up,1-p_up); model_dir="Yükseliş" if p_up>=.5 else "Düşüş"
    r=latest.iloc[0]
    # Altı ayrı teknik kontrol; nötr momentum/RSI zorla yön oyu üretmez.
    votes=[
        1 if r.ema12>r.ema26 else -1,
        1 if r.ema26>r.ema50 else -1,
        1 if r.macd>r.macd_signal else -1,
        1 if r.close>r.ema50 else -1,
        1 if r.momentum3>0.015 else -1 if r.momentum3 < -0.015 else 0,
        1 if r.rsi>=52 else -1 if r.rsi<=48 else 0,
    ]
    up=sum(v==1 for v in votes); down=sum(v==-1 for v in votes)
    tech_dir="Yükseliş" if up>down else "Düşüş" if down>up else "BEKLE"
    support=up if model_dir=="Yükseliş" else down
    if confidence<threshold:
        signal,reason="BEKLE","Model yön skoru seçilen eşik altında."
    elif tech_dir=="BEKLE":
        signal,reason="BEKLE","Teknik kontroller aynı yönde yeterince birleşmiyor."
    elif model_dir!=tech_dir:
        signal,reason="BEKLE","Model yönü ve teknik kontroller birbiriyle çelişiyor."
    elif support<min_votes:
        signal,reason="BEKLE",f"Yönü destekleyen teknik kontrol sayısı yetersiz ({support})."
    else:
        signal,reason=model_dir,f"Model ve teknik kontroller aynı yönde ({support}/6). Bu, işlem başarısı garantisi değildir."
    return {"data":usable,"accuracy":float(accuracy),"model_direction":model_dir,"confidence":confidence,"p_up":p_up,"technical_direction":tech_dir,"signal":signal,"reason":reason,"support":support,"up_votes":up,"down_votes":down,"close":float(r.close),"rsi":float(r.rsi),"ema12":float(r.ema12),"ema26":float(r.ema26),"ema50":float(r.ema50),"macd":float(r.macd),"momentum3":float(r.momentum3),"timestamp":r.timestamp}

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

results = {}
cols = st.columns(2)

for col, tf, frame in [(cols[0], "M1", m1_data), (cols[1], "M5", m5_data)]:
    with col:
        st.subheader(f"{tf} ANALİZİ")
        try:
            result = analyze(frame, threshold, min_votes)
            results[tf] = result
        except Exception as exc:
            st.error(f"{tf} analizi yapılamadı: {exc}")
            continue

        st.metric("Karar", result["signal"])
        st.metric("Model yön skoru (olasılık değildir)", f'{result["confidence"] * 100:.1f}%')
        st.metric("Kronolojik test doğruluğu", f'{result["accuracy"] * 100:.1f}%')
        st.write(f'**Son fiyat:** {result["close"]:.5f}')
        st.write(f'**Model:** {result["model_direction"]} · **Teknik:** {result["technical_direction"]}')
        st.write(f'**RSI:** {result["rsi"]:.1f} · **3 mum momentumu:** {result["momentum3"]:.3f}%')
        st.write(f'**Teknik destek:** {result["support"]}/6 kontrol · **Yükseliş/Düşüş oyları:** {result["up_votes"]}/{result["down_votes"]}')
        st.caption(result["reason"])

        chart_data = result["data"].tail(100).copy()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=chart_data["timestamp"], open=chart_data["open"], high=chart_data["high"],
            low=chart_data["low"], close=chart_data["close"], name="Fiyat",
            increasing_line_color="#d6ad4f", decreasing_line_color="#e05b63",
        ))
        fig.add_trace(go.Scatter(x=chart_data["timestamp"], y=chart_data["ema12"], mode="lines",
                                 name="EMA12", line=dict(color="#f3d27a", width=1.5)))
        fig.add_trace(go.Scatter(x=chart_data["timestamp"], y=chart_data["ema26"], mode="lines",
                                 name="EMA26", line=dict(color="#68a7ff", width=1.3)))
        fig.add_trace(go.Scatter(x=chart_data["timestamp"], y=chart_data["ema50"], mode="lines",
                                 name="EMA50", line=dict(color="#b08cff", width=1.2)))
        fig.update_layout(
            template="plotly_dark", height=420, margin=dict(l=8, r=8, t=28, b=8),
            paper_bgcolor="#11151c", plot_bgcolor="#11151c",
            xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.04, x=0),
            xaxis_title="İstanbul saati", yaxis_title="Fiyat",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True})

        with st.expander(f"{tf} son 10 mum"):
            recent = result["data"][["timestamp", "open", "high", "low", "close"]].tail(10).copy()
            recent["değişim (%)"] = recent["close"].pct_change() * 100
            st.dataframe(recent, hide_index=True, use_container_width=True)

st.subheader("M1 / M5 UYUM KONTROLÜ")
if len(results) == 2:
    m1, m5 = results["M1"], results["M5"]
    if m1["signal"] == m5["signal"] and m1["signal"] in ("Yükseliş", "Düşüş"):
        st.success(f'GENEL DURUM: {m1["signal"]} ADAYI · M1 ve M5 aynı yönde. Bu otomatik işlem emri değildir; başarı garantisi yoktur.')
    elif m1["model_direction"] == m5["model_direction"] and m1["model_direction"] in ("Yükseliş", "Düşüş"):
        st.info(f'GENEL DURUM: {m1["model_direction"]} YÖNÜ İZLENİYOR · Model yönleri aynı ancak teknik şartlar tam onay vermiyor. Şimdilik BEKLE.')
    elif "BEKLE" in (m1["signal"], m5["signal"]):
        st.info("GENEL DURUM: BEKLE · En az bir zaman diliminde onay eksik. Çelişkiyi zorla sinyale çevirmiyoruz.")
    else:
        st.warning("GENEL DURUM: BEKLE · M1 ve M5 yönleri farklı.")
    st.dataframe(pd.DataFrame([
        {"Zaman": tf, "Karar": r["signal"], "Model yönü": r["model_direction"],
         "Model yön skoru (%)": round(r["confidence"] * 100, 1),
         "Test doğruluğu (%)": round(r["accuracy"] * 100, 1), "RSI": round(r["rsi"], 1),
         "Teknik destek (/6)": r["support"], "3 mum momentumu (%)": round(r["momentum3"], 3),
         "Son mum (İstanbul)": str(r["timestamp"])}
        for tf, r in results.items()
    ]), hide_index=True, use_container_width=True)

with st.expander("Veri ve model notları"):
    st.write(f"Sembol: {symbol} · M1 kayıt: {len(m1_data)} · M5 kayıt: {len(m5_data)}")
    st.write(f"Son M1 yaşı: {age1:.1f} dakika · Son M5 yaşı: {age5:.1f} dakika")
    st.caption("Model yön skoru kazanma olasılığı değildir. Teknik destek 6 kontrolden hesaplanır. Test geçmiş verinin kronolojik bölümünde yapılır; geleceği garanti etmez. Uygulama otomatik işlem açmaz.")

st.caption("Pocket AI Agent · Demo analiz aracı · Normal forex verisi OTC fiyatıyla farklı olabilir.")
