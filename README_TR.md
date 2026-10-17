# Pocket AI Agent — Demo MVP

Bu proje, Pocket Option'a bağlanmadan çalışan bir **demo analiz ajanı** başlangıç sürümüdür.
Gerçek hesapta işlem açmaz, hesap şifresi/çerez istemez ve kâr garantisi vermez.

## İçindekiler
- Streamlit web paneli
- CSV fiyat verisi yükleme
- EMA / RSI / volatilite tabanlı örnek sinyal üretimi
- Basit makine öğrenmesi modeli (scikit-learn) ile deneysel yukarı/aşağı tahmini
- Demo işlem simülasyonu ve işlem geçmişi
- İşlem tutarı ve günlük zarar limiti ayarları

## Çalıştırma
Python 3.11 önerilir.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Panel açıldığında tarayıcıda gösterilen yerel adresi kullanın.

## CSV biçimi
En az şu sütunlar olmalı:
- `timestamp` (tarih/saat)
- `close` (kapanış fiyatı)

İsteğe bağlı: `open`, `high`, `low`, `volume`.

## Önemli sınırlar
- Model eğitim amaçlıdır; tahminleri yatırım tavsiyesi değildir.
- Örnek sinyal stratejisi kârlı olduğu iddia edilen bir strateji değildir.
- Canlı Pocket Option otomasyonu bu projede yoktur. Önce platformun izin verdiği resmî entegrasyon doğrulanmalıdır.
- Gerçek hesaba geçiş yapılmadan önce zaman sıralı out-of-sample test, ödeme oranı ve kayıp serileri değerlendirilmelidir.
