}


uploaded = st.file_uploader("Fiyat verisi CSV yükle (timestamp, close)", type=["csv"])
if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        if "close" not in raw.columns:
            st.error("CSV dosyasında 'close' sütunu bulunamadı.")
            st.stop()
        if "timestamp" not in raw.columns:
            st.error("CSV dosyasında 'timestamp' sütunu bulunmalı. Örnek: 2026-10-17 14:30:00")
            st.stop()

        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
        raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
        raw = raw.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
        raw = raw.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        if len(raw) < 100:
            st.error("M1/M5 analizi için en az 100 fiyat satırı yükle.")
            st.stop()
        df = raw[["timestamp", "close"]].copy()
        source_label = "Yüklenen CSV (verinin doğruluğu kullanıcıya bağlı)"
    except Exception as e:
        st.error(f"CSV okunamadı: {e}")
        st.stop()
else:
    df = sample_data()
    source_label = "Sentetik örnek veri — gerçek piyasa verisi değil"

st.info(f"Veri kaynağı: **{source_label}** · M1 bir dakikalık, M5 beş dakikalık kapanışlara göre analiz edilir.")

results = {}
left, right = st.columns(2)
for col, timeframe in [(left, "M1"), (right, "M5")]:
    with col:
        st.subheader(f"⏱️ {timeframe} Analizi")
        result = analyze_timeframe(df, timeframe, threshold, trade_amount, payout)
        results[timeframe] = result
        if "error" in result:
            st.error(result["error"])
            continue
        st.metric("Zaman sıralı test doğruluğu", f"{result['accuracy'] * 100:.1f}%")
        st.metric("Son tahmin", result["direction"])
        st.metric("Model güven skoru", f"{result['confidence'] * 100:.1f}%")
        st.write(f"**Karar:** {result['signal']}")
        st.write(f"**Son kapanış:** {result['latest_close']:.5f}")
        st.write(
            f"**RSI:** {result['latest_rsi']:.1f} · "
            f"**EMA12:** {result['ema_fast']:.5f} · **EMA26:** {result['ema_slow']:.5f}"
        )
        st.line_chart(result["data"].set_index("timestamp")["close"])

st.subheader("M1 / M5 sinyal karşılaştırması")
valid = [results[t] for t in ["M1", "M5"] if t in results and "error" not in results[t]]
if len(valid) == 2:
    if valid[0]["direction"] == valid[1]["direction"]:
        st.success(
            f"İki zaman diliminin yön tahmini aynı: {valid[0]['direction']}. "
            "Bu, tahminin doğru olacağını garanti etmez."
        )
    else:
        st.warning(
            "M1 ve M5 yön tahminleri farklı. Bu prototipte sinyaller birbiriyle çelişiyor; "
            "gerçek işlem kararı olarak değerlendirme."
        )
    comparison = pd.DataFrame([
        {
            "Zaman dilimi": r["timeframe"],
            "Yön tahmini": r["direction"],
            "Güven (%)": round(r["confidence"] * 100, 1),
            "Karar": r["signal"],
            "Test doğruluğu (%)": round(r["accuracy"] * 100, 1),
        }
        for r in valid
    ])
    st.dataframe(comparison, hide_index=True, use_container_width=True)

st.subheader("Demo işlem simülasyonu")
st.write(
    "Sonuçlar her zaman diliminin kendi geçmiş test bölümündeki tahminlerine dayanır. "
    "Ödeme hesabı basitleştirilmiştir; gerçek işlem maliyetlerini veya platform koşullarını içermez."
)
sim_cols = st.columns(2)
for col, timeframe in zip(sim_cols, ["M1", "M5"]):
    with col:
        st.markdown(f"**{timeframe} sanal sonuçları**")
        r = results[timeframe]
        if "error" in r:
            st.caption("Simülasyon için yeterli veri yok.")
            continue
        st.metric("Sanal sinyal sayısı", r["trades"])
        st.metric("Kazanan / Kaybeden", f"{r['wins']} / {r['losses']}")
        st.metric("Basitleştirilmiş sanal net sonuç", f"{r['pnl']:+.2f} USD")
        if r["pnl"] <= -daily_loss_limit:
            st.error("Bu test bölümünde sanal sonuç seçilen zarar sınırına ulaştı/aştı.")
        st.caption("Geçmiş test sonucu gelecekteki performansı garanti etmez.")

with st.expander("Test ayrıntıları"):
    for timeframe in ["M1", "M5"]:
        r = results[timeframe]
        st.markdown(f"**{timeframe}**")
        if "error" in r:
            st.write(r["error"])
        else:
            st.write("Özellikler:", r["feature_cols"])
            st.write("Eğitim satırı:", r["train_rows"], "· Test satırı:", r["test_rows"])
            st.write("Güven eşiği:", f"{threshold * 100:.0f}%")
            st.write("Ödeme oranı:", f"{payout * 100:.0f}%")
            st.dataframe(r["data"].tail(10), use_container_width=True)

st.caption(
    "Uyarı: Bu prototip yatırım tavsiyesi değildir. Canlı fiyat bağlantısı ve Pocket Option entegrasyonu yoktur. "
    "Gerçek para ile işlem açmaz."
)
