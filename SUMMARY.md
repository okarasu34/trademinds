# TradeMinds - Capital.com Integration Summary

## 👋 Merhaba!

Capital.com entegrasyonu tamamlandı! Bot artık Capital.com'daki "TradeMinds" watchlist'inizdeki sembolleri otomatik olarak trade edebilir.

---

## 📦 Yapılan İşler

### 1. Capital.com Adapter (Broker Entegrasyonu)
✅ **Dosya**: `backend/brokers/capital_adapter.py`

**Özellikler**:
- Login/logout (CST ve X-SECURITY-TOKEN ile)
- Hesap bilgileri (bakiye, equity, margin)
- Gerçek zamanlı fiyatlar (bid/ask/spread)
- Tarihsel mum verileri (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
- Emir açma (market order + SL/TP)
- Pozisyon yönetimi (açma/kapama)
- **Watchlist entegrasyonu** (otomatik sembol yükleme)

### 2. Trading Bot Güncelleme
✅ **Dosya**: `backend/bot/trading_bot.py`

**Değişiklik**: `_get_symbols()` metodu güncellendi
- Önce strategy'deki symbols'e bakar
- Yoksa adapter'dan watchlist symbols'ü alır
- Yoksa default symbols'leri kullanır

### 3. Base Adapter Güncelleme
✅ **Dosya**: `backend/brokers/base_adapter.py`

**Değişiklik**: Factory'ye Capital.com eklendi
```python
if broker_type in ["capital", "capital.com", "capitalcom"]:
    from brokers.capital_adapter import CapitalAdapter
    return CapitalAdapter(account)
```

### 4. Watchlist Sync Script
✅ **Dosya**: `scripts/update_capital_watchlist.py`

**Kullanım**:
```bash
cd /root/trademinds
venv/bin/python3 scripts/update_capital_watchlist.py
```

**Ne yapar**:
- Capital.com'a bağlanır
- "TradeMinds" watchlist'ini bulur
- Tüm sembolleri çeker
- Aktif strategy'leri bu sembollerle günceller

### 5. Deployment Scripts
✅ **Dosyalar**: 
- `scripts/deploy_capital_integration.sh` (Linux/Mac)
- `scripts/deploy_capital_integration.ps1` (Windows)

**Kullanım**: Dosyaları sunucuya otomatik kopyalar ve backend'i restart eder

### 6. Dokümantasyon
✅ **Dosyalar**:
- `docs/CAPITAL_COM_INTEGRATION.md` - Detaylı entegrasyon rehberi
- `CHANGELOG_CAPITAL_INTEGRATION.md` - Tüm değişikliklerin listesi
- `QUICK_START_CAPITAL.md` - Hızlı başlangıç rehberi
- `SUMMARY.md` - Bu dosya (özet)

---

## 🚀 Nasıl Çalışır?

### Otomatik Mod (Varsayılan)

1. **Bot başlatılır**
   - Capital.com'a bağlanır
   - "TradeMinds" watchlist'ini arka planda yükler
   - Sembolleri memory'de cache'ler

2. **Her dakika**
   - Watchlist'teki sembolleri tarar
   - Her sembol için:
     - 1h mum verileri (entry timeframe)
     - 4h mum verileri (trend filtresi)
     - İndikatörler hesaplanır (ADX, EMA, RSI, vb.)
     - Strateji filtreleri uygulanır
     - Sinyal üretilir (rule-based)
     - Güçlü sinyal varsa emir açılır

3. **Her saat**
   - Watchlist cache'i otomatik yenilenir
   - Yeni eklenen semboller otomatik dahil edilir

### Manuel Mod (Opsiyonel)

Watchlist'i database'e kaydetmek isterseniz:
```bash
python3 scripts/update_capital_watchlist.py
```

Bu script `Strategy.symbols` alanını günceller.

---

## 📋 Yapılacaklar (Deployment)

### 1. Dosyaları Sunucuya Kopyala

**Otomatik (Önerilen)**:
```powershell
# Windows PowerShell
# Önce server IP'sini düzenle
notepad scripts\deploy_capital_integration.ps1

# Çalıştır
.\scripts\deploy_capital_integration.ps1
```

**Manuel**:
```bash
# Her dosyayı tek tek kopyala
scp backend/brokers/capital_adapter.py root@your-server:/root/trademinds/backend/brokers/
scp backend/brokers/base_adapter.py root@your-server:/root/trademinds/backend/brokers/
scp backend/bot/trading_bot.py root@your-server:/root/trademinds/backend/bot/
scp scripts/update_capital_watchlist.py root@your-server:/root/trademinds/scripts/
```

### 2. Backend'i Restart Et

```bash
ssh root@your-server
supervisorctl restart trademinds

# Backend'in başladığını kontrol et
sleep 10
curl http://localhost:8001/health
```

### 3. Capital.com'da Watchlist Oluştur

1. [Capital.com](https://capital.com) web veya mobil uygulamasına giriş yap
2. **"TradeMinds"** adında bir watchlist oluştur (tam olarak bu isim)
3. Trade etmek istediğin sembolleri ekle:
   - Forex: EURUSD, GBPUSD, USDJPY, vb.
   - Crypto: BTCUSD, ETHUSD, XRPUSD, vb.
   - Commodity: GOLD, SILVER, OIL_BRENT, vb.
   - Index: US500, US100, US30, vb.
   - Stock: AAPL, MSFT, GOOGL, vb.

### 4. Bot'u Başlat

```bash
# Token al
TOKEN=$(curl -s -X POST http://your-server:8001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"ozckrs34@gmail.com","password":"Admin1234"}' \
  | jq -r '.access_token')

# Bot'u başlat
curl -X POST http://your-server:8001/api/bot/start \
  -H "Authorization: Bearer $TOKEN"

# Durumu kontrol et
curl http://your-server:8001/api/bot/status \
  -H "Authorization: Bearer $TOKEN"
```

### 5. Logları İzle

```bash
ssh root@your-server
tail -f /tmp/trademinds.log | grep -i capital
```

Göreceğin loglar:
```
Connected to Capital.com | account=<account_id>
Loaded 50 symbols from TradeMinds watchlist
Found 50 markets in TradeMinds watchlist
Order placed on Capital.com: EURUSD buy 0.1 | ref=<deal_ref>
```

---

## 🎯 Önemli Notlar

### Watchlist Senkronizasyonu
- ✅ **Otomatik**: Bot başladığında yüklenir, her saat yenilenir
- ✅ **Manuel**: `update_capital_watchlist.py` script'i ile
- ✅ **Cache**: Memory'de tutulur, database'e yazmak opsiyonel

### Sembol Formatı
Capital.com "epic" kodları kullanır:
- Forex: `EURUSD`, `GBPUSD`, `USDJPY`
- Crypto: `BTCUSD`, `ETHUSD`, `XRPUSD`
- Commodity: `GOLD`, `SILVER`, `OIL_BRENT`
- Index: `US500`, `US100`, `US30`
- Stock: `AAPL`, `MSFT`, `GOOGL`

### Market Limitleri
Bot config'te tanımlı:
```json
{
  "market_limits": {
    "forex": 10,      // Aynı anda max 10 forex pozisyonu
    "crypto": 5,      // Aynı anda max 5 crypto pozisyonu
    "commodity": 4,   // Aynı anda max 4 commodity pozisyonu
    "stock": 4,       // Aynı anda max 4 stock pozisyonu
    "index": 2        // Aynı anda max 2 index pozisyonu
  }
}
```

### Risk Yönetimi
- **Max daily loss**: %5 (config'te ayarlanabilir)
- **Max risk per trade**: %1 (config'te ayarlanabilir)
- **Stop loss**: ATR bazlı otomatik hesaplanır
- **Take profit**: ATR bazlı otomatik hesaplanır
- **Lot size**: Risk yüzdesine göre otomatik hesaplanır

### Strateji
- **Entry timeframe**: 1h (200 bar)
- **Trend filter**: 4h (100 bar)
- **Indicators**: ADX, EMA 50/200, RSI, ATR
- **Signal type**: Rule-based (AI kullanmıyor)
- **HTF trend required**: Evet (4h trend ile aynı yönde trade)

---

## 🔧 Sorun Giderme

### "TradeMinds watchlist not found"
**Çözüm**: Capital.com'da "TradeMinds" adında watchlist oluştur

### "No markets found in watchlist"
**Çözüm**: Watchlist'e sembol ekle

### "Bot not trading Capital.com symbols"
**Kontrol et**:
1. Broker account aktif mi? → `GET /api/brokers`
2. Strategy aktif mi? → `GET /api/strategies`
3. Strategy.symbols = null mi? → (null olmalı watchlist kullanmak için)
4. Logları kontrol et → `tail -f /tmp/trademinds.log`

### "Capital.com login failed: 401"
**Çözüm**: API credentials'ları kontrol et (broker account settings)

### "Connection refused"
**Çözüm**: Backend çalışıyor mu? → `curl http://localhost:8001/health`

---

## 📊 Örnek Watchlist

Capital.com'da şu sembolleri ekleyebilirsin:

**Forex** (10 max):
```
EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD, EURJPY, GBPJPY, EURCHF
```

**Crypto** (5 max):
```
BTCUSD, ETHUSD, XRPUSD, SOLUSD, ADAUSD
```

**Commodities** (4 max):
```
GOLD, SILVER, OIL_BRENT, NATURALGAS
```

**Indices** (2 max):
```
US500, US100
```

**Stocks** (4 max):
```
AAPL, MSFT, GOOGL, TSLA
```

---

## 📚 Dokümantasyon

Detaylı bilgi için:
- **Hızlı Başlangıç**: `QUICK_START_CAPITAL.md`
- **Tam Rehber**: `docs/CAPITAL_COM_INTEGRATION.md`
- **Değişiklikler**: `CHANGELOG_CAPITAL_INTEGRATION.md`

---

## ✅ Checklist

- [ ] Dosyaları sunucuya kopyaladım
- [ ] Backend'i restart ettim
- [ ] Capital.com'da "TradeMinds" watchlist oluşturdum
- [ ] Watchlist'e semboller ekledim
- [ ] Bot'u başlattım
- [ ] Logları izliyorum
- [ ] İlk trade'i gördüm! 🎉

---

## 🎉 Hazırsın!

Bot artık Capital.com watchlist'indeki sembolleri otomatik olarak trade edecek.

**Başarılar! 📈**

---

## 📞 Yardım

Sorun olursa:
1. Logları kontrol et: `tail -f /tmp/trademinds.log`
2. Dokümantasyonu oku: `docs/CAPITAL_COM_INTEGRATION.md`
3. Test script'ini çalıştır: `scripts/update_capital_watchlist.py`

---

**Son Güncelleme**: 18 Nisan 2026
**Durum**: ✅ Tamamlandı ve test edilmeye hazır
