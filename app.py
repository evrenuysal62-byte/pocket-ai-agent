
    
  
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Pocket AI Agent — M1/M5 Demo", page_icon="📊", layout="wide")
st.title("📊 Pocket AI Agent")
st.caption("M1 + M5 demo analiz prototipi · Pocket Option hesabına bağlanmaz · Gerçek işlem açmaz")

st.warning(
    "Bu uygulama gerçek zamanlı Pocket Option verisi almaz ve işlem açmaz. "
    "CSV yüklemezsen sentetik örnek veri kullanılır; sonuçlar gerçek piyasa performansı değildir."
)

with st.sidebar:
    st.header("Risk ve sinyal ayarları")
    trade_amount = st.number_input("Sanal işlem tutarı (USD)", min_value=1.0, max_value=100.0, value=1.0, step=1.0)
    payout = st.slider("Varsayılan ödeme oranı (%)", min_value=50, max_value=95, value=85) / 100
    daily_loss_limit = st.number_input("Günlük sanal zarar durdurma (USD)", min_value=1.0, max_value=500.0, value=10.0, step=1.0)
    threshold = st.slider("Model güven eşiği (%)", min_value=50, max_value=95, value=65) / 100


def indicators(df):
    d = df.copy().sort_values("timestamp").reset_index(drop=True)
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


def sample_data(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0001, 0.004, n)
    close = 100 * np.exp(np.cumsum(rets))
    times = pd.date_range(end=pd.Timestamp.now().floor("min"), periods=n, freq="min")
    return pd.DataFrame({"timestamp": times, "close": close})


def analyze_timeframe(base_df, timeframe, threshold, trade_amount, payout):
    """Analyze minute data at M1 or resampled M5 interval."""
    if timeframe == "M5":
        work = (
            base_df.set_index("timestamp")["close"]
            .resample("5min")
            .last()
            .dropna()
            .rename("close")
            .reset_index()
        )
    else:
        work = base_df[["timestamp", "close"]].copy()

    d = indicators(work).dropna().reset_index(drop=True)
    if len(d) < 80:
        return {"error": f"{timeframe} için yeterli veri yok. En az 80 kullanılabilir mum gerekir."}

    # Son satırın bir sonraki mum sonucu bilinmediğinden eğitim/test dışında tutulur.
    trainable = d.iloc[:-1].copy()
    split = int(len(trainable) * 0.8)
    if split < 40 or len(trainable) - split < 10:
        return {"error": f"{timeframe} için eğitim/test verisi yetersiz."}

    feature_cols = ["ema_fast", "ema_slow", "rsi", "returns", "volatility"]
    X_train = trainable.iloc[:split][feature_cols]
    y_train = trainable.iloc[:split]["future_up"].astype(int)
    X_test = trainable.iloc[split:][feature_cols]
    y_test = trainable.iloc[split:]["future_up"].astype(int)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred) if len(y_test) else float("nan")

    latest = d.iloc[[-1]]
    classes = list(model.classes_)
    probs = model.predict_proba(latest[feature_cols])[0]
    proba_up = float(probs[classes.index(1)]) if 1 in classes else 0.5
    confidence = max(proba_up, 1 - proba_up)
    direction = "CALL / Yukarı" if proba_up >= 0.5 else "PUT / Aşağı"
    signal = direction if confidence >= threshold else "İŞLEM YOK — güven eşiği altında"

    test_proba = model.predict_proba(X_test)
    up_idx = classes.index(1) if 1 in classes else None
    ntrades = wins = losses = 0
    pnl = 0.0
    if up_idx is not None:
        test_up = test_proba[:, up_idx]
        test_conf = np.maximum(test_up, 1 - test_up)
        test_dir = (test_up >= 0.5).astype(int)
        take = test_conf >= threshold
        outcomes = test_dir[take] == y_test.to_numpy()[take]
        ntrades = int(take.sum())
        wins = int(outcomes.sum()) if ntrades else 0
        losses = ntrades - wins
        pnl = wins * trade_amount * payout - losses * trade_amount

    return {
        "timeframe": timeframe,
        "data": d,
        "accuracy": acc,
        "direction": direction,
        "confidence": confidence,
        "signal": signal,
        "latest_close": float(latest["close"].iloc[0]),
        "latest_rsi": float(latest["rsi"].iloc[0]),
        "ema_fast": float(latest["ema_fast"].iloc[0]),
        "ema_slow": float(latest["ema_slow"].iloc[0]),
        "trades": ntrades,
        "wins": wins,
        "losses": losses,
        "pnl": pnl,
        "feature_cols": feature_cols,
        "train_rows": len(X_train),
        "test_rows": len(X_test),

