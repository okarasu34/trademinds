# Capital.com Integration - Quick Start Guide

## 🎯 What's New

Your TradeMinds bot can now automatically trade symbols from your Capital.com watchlist!

## ✅ What's Been Done

1. ✅ **Capital.com Adapter** - Full broker integration
2. ✅ **Watchlist Support** - Automatic symbol loading from Capital.com
3. ✅ **Trading Bot Update** - Dynamic symbol selection
4. ✅ **Update Script** - Manual watchlist sync tool
5. ✅ **Documentation** - Complete integration guide

## 📁 Files Created/Modified

### Created:
- `backend/brokers/capital_adapter.py` - Capital.com broker adapter
- `scripts/update_capital_watchlist.py` - Watchlist sync script
- `scripts/deploy_capital_integration.sh` - Bash deployment script
- `scripts/deploy_capital_integration.ps1` - PowerShell deployment script
- `docs/CAPITAL_COM_INTEGRATION.md` - Full documentation
- `CHANGELOG_CAPITAL_INTEGRATION.md` - Detailed changelog

### Modified:
- `backend/brokers/base_adapter.py` - Added Capital.com to factory
- `backend/bot/trading_bot.py` - Added watchlist support to `_get_symbols()`

## 🚀 Deployment (Choose One Method)

### Method 1: Automatic Deployment (Recommended)

**Windows (PowerShell)**:
```powershell
# Edit the script first to set your server IP
notepad scripts\deploy_capital_integration.ps1

# Run deployment
.\scripts\deploy_capital_integration.ps1
```

**Linux/Mac (Bash)**:
```bash
# Edit the script first to set your server IP
nano scripts/deploy_capital_integration.sh

# Make executable
chmod +x scripts/deploy_capital_integration.sh

# Run deployment
./scripts/deploy_capital_integration.sh
```

### Method 2: Manual Deployment

**Copy files to server**:
```bash
# Replace 'your-server-ip' with actual IP
scp backend/brokers/capital_adapter.py root@your-server-ip:/root/trademinds/backend/brokers/
scp backend/brokers/base_adapter.py root@your-server-ip:/root/trademinds/backend/brokers/
scp backend/bot/trading_bot.py root@your-server-ip:/root/trademinds/backend/bot/
scp scripts/update_capital_watchlist.py root@your-server-ip:/root/trademinds/scripts/
```

**Restart backend**:
```bash
ssh root@your-server-ip
supervisorctl restart trademinds
```

## 📋 Setup Checklist

### 1. Deploy Code ✅
- [ ] Copy files to server (see above)
- [ ] Restart backend service
- [ ] Verify backend is running: `curl http://localhost:8001/health`

### 2. Capital.com Setup 🏦
- [ ] Login to [Capital.com](https://capital.com)
- [ ] Create a watchlist named **"TradeMinds"** (exact name, case-insensitive)
- [ ] Add symbols you want to trade (e.g., EURUSD, BTCUSD, GOLD, US500)

### 3. Configure Bot 🤖
- [ ] Broker account already configured (you did this earlier)
- [ ] Strategy configured with `symbols: null` (to use watchlist)
- [ ] Bot config has correct market limits

### 4. Sync Watchlist 🔄

**Option A: Automatic (Recommended)**
- Just start the bot - it will load the watchlist automatically!

**Option B: Manual**
```bash
ssh root@your-server-ip
cd /root/trademinds
venv/bin/python3 scripts/update_capital_watchlist.py
```

### 5. Start Trading 📈
```bash
# Get your access token first
TOKEN=$(curl -s -X POST http://your-server-ip:8001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"ozckrs34@gmail.com","password":"Admin1234"}' \
  | jq -r '.access_token')

# Start the bot
curl -X POST http://your-server-ip:8001/api/bot/start \
  -H "Authorization: Bearer $TOKEN"

# Check status
curl http://your-server-ip:8001/api/bot/status \
  -H "Authorization: Bearer $TOKEN"
```

### 6. Monitor 👀
```bash
# Watch logs
ssh root@your-server-ip
tail -f /tmp/trademinds.log | grep -i capital

# Or filter for specific events
tail -f /tmp/trademinds.log | grep -E "(capital|watchlist|EURUSD|BTCUSD)"
```

## 🎮 How It Works

### Automatic Mode (Default)
1. Bot starts → Connects to Capital.com
2. Loads "TradeMinds" watchlist in background
3. Caches symbols in memory
4. Scans and trades those symbols
5. Refreshes watchlist every hour

### Manual Mode (Optional)
1. Run `update_capital_watchlist.py` script
2. Script fetches watchlist from Capital.com
3. Updates `Strategy.symbols` in database
4. Bot uses those symbols from database

## 📊 Example Watchlist

Create a watchlist on Capital.com with these symbols:

**Forex** (10 max):
- EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD, EURJPY, GBPJPY, EURCHF

**Crypto** (5 max):
- BTCUSD, ETHUSD, XRPUSD, SOLUSD, ADAUSD

**Commodities** (4 max):
- GOLD, SILVER, OIL_BRENT, NATURALGAS

**Indices** (2 max):
- US500, US100

**Stocks** (4 max):
- AAPL, MSFT, GOOGL, TSLA

## 🔧 Troubleshooting

### "TradeMinds watchlist not found"
**Solution**: Create the watchlist on Capital.com web/mobile app

### "No markets found in watchlist"
**Solution**: Add symbols to your Capital.com watchlist

### "Bot not trading Capital.com symbols"
**Check**:
1. Is broker account active? `GET /api/brokers`
2. Is strategy active? `GET /api/strategies`
3. Is strategy.symbols = null? (should be null to use watchlist)
4. Check logs: `tail -f /tmp/trademinds.log`

### "Connection refused to Capital.com"
**Check**:
1. API credentials correct?
2. Capital.com API accessible? `curl -I https://api-capital.backend-capital.com`
3. Firewall blocking outbound HTTPS?

## 📚 Documentation

- **Full Guide**: `docs/CAPITAL_COM_INTEGRATION.md`
- **Changelog**: `CHANGELOG_CAPITAL_INTEGRATION.md`
- **This Guide**: `QUICK_START_CAPITAL.md`

## 🎯 Quick Commands

```bash
# Deploy to server
./scripts/deploy_capital_integration.sh

# SSH to server
ssh root@your-server-ip

# Update watchlist manually
cd /root/trademinds && venv/bin/python3 scripts/update_capital_watchlist.py

# Restart backend
supervisorctl restart trademinds

# Check backend
curl http://localhost:8001/health

# Watch logs
tail -f /tmp/trademinds.log | grep -i capital

# Start bot (replace TOKEN)
curl -X POST http://localhost:8001/api/bot/start -H "Authorization: Bearer TOKEN"

# Stop bot
curl -X POST http://localhost:8001/api/bot/stop -H "Authorization: Bearer TOKEN"

# Check bot status
curl http://localhost:8001/api/bot/status -H "Authorization: Bearer TOKEN"
```

## ✨ Features

- ✅ **Automatic watchlist sync** - No manual updates needed
- ✅ **Multi-asset support** - Forex, Crypto, Commodities, Stocks, Indices
- ✅ **Real-time data** - Live bid/ask prices
- ✅ **Multi-timeframe analysis** - 1h entry + 4h trend filter
- ✅ **Risk management** - Position limits, daily loss limits
- ✅ **News filtering** - Pauses trading during high-impact news
- ✅ **Rule-based strategy** - No AI API needed
- ✅ **Demo account support** - Test before going live

## 🎉 You're Ready!

1. Deploy the code ✅
2. Create Capital.com watchlist ✅
3. Start the bot ✅
4. Monitor and profit! 📈

---

**Need Help?**
- Check logs: `/tmp/trademinds.log`
- Review docs: `docs/CAPITAL_COM_INTEGRATION.md`
- Test connection: `scripts/update_capital_watchlist.py`
