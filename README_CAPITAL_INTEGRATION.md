# 🚀 Capital.com Integration - Complete Package

## 📦 What's Included

This package contains everything you need to integrate Capital.com with your TradeMinds trading bot, including automatic watchlist synchronization.

### Files Created

#### Core Integration
- ✅ `backend/brokers/capital_adapter.py` - Capital.com broker adapter (full implementation)
- ✅ `backend/brokers/base_adapter.py` - Updated factory (Capital.com support added)
- ✅ `backend/bot/trading_bot.py` - Updated bot (watchlist support added)

#### Scripts
- ✅ `scripts/update_capital_watchlist.py` - Manual watchlist sync
- ✅ `scripts/test_capital_connection.py` - Connection test script
- ✅ `scripts/deploy_capital_integration.sh` - Bash deployment script
- ✅ `scripts/deploy_capital_integration.ps1` - PowerShell deployment script

#### Documentation
- ✅ `docs/CAPITAL_COM_INTEGRATION.md` - Full integration guide
- ✅ `docs/CAPITAL_FLOW_DIAGRAM.md` - Visual flow diagrams
- ✅ `CHANGELOG_CAPITAL_INTEGRATION.md` - Detailed changelog
- ✅ `QUICK_START_CAPITAL.md` - Quick start guide
- ✅ `SUMMARY.md` - Turkish summary
- ✅ `README_CAPITAL_INTEGRATION.md` - This file

---

## 🎯 Quick Start (5 Minutes)

### 1. Deploy to Server

**Windows**:
```powershell
# Edit server IP first
notepad scripts\deploy_capital_integration.ps1

# Deploy
.\scripts\deploy_capital_integration.ps1
```

**Linux/Mac**:
```bash
# Edit server IP first
nano scripts/deploy_capital_integration.sh

# Deploy
chmod +x scripts/deploy_capital_integration.sh
./scripts/deploy_capital_integration.sh
```

### 2. Create Watchlist on Capital.com

1. Login to [Capital.com](https://capital.com)
2. Create watchlist named **"TradeMinds"**
3. Add symbols (EURUSD, BTCUSD, GOLD, etc.)

### 3. Test Connection

```bash
ssh root@your-server
cd /root/trademinds
venv/bin/python3 scripts/test_capital_connection.py
```

### 4. Start Bot

```bash
# Get token
TOKEN=$(curl -s -X POST http://your-server:8001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"your@email.com","password":"yourpassword"}' \
  | jq -r '.access_token')

# Start bot
curl -X POST http://your-server:8001/api/bot/start \
  -H "Authorization: Bearer $TOKEN"
```

### 5. Monitor

```bash
tail -f /tmp/trademinds.log | grep -i capital
```

---

## 📚 Documentation Guide

### For Quick Setup
→ Read: `QUICK_START_CAPITAL.md`

### For Complete Understanding
→ Read: `docs/CAPITAL_COM_INTEGRATION.md`

### For Visual Learners
→ Read: `docs/CAPITAL_FLOW_DIAGRAM.md`

### For Turkish Speakers
→ Read: `SUMMARY.md`

### For Developers
→ Read: `CHANGELOG_CAPITAL_INTEGRATION.md`

---

## 🔧 Features

### Automatic Watchlist Integration
- ✅ Loads symbols from Capital.com "TradeMinds" watchlist
- ✅ Refreshes automatically every hour
- ✅ No manual configuration needed
- ✅ Add/remove symbols on Capital.com, bot adapts automatically

### Multi-Asset Support
- ✅ Forex (EURUSD, GBPUSD, etc.)
- ✅ Crypto (BTCUSD, ETHUSD, etc.)
- ✅ Commodities (GOLD, SILVER, OIL)
- ✅ Indices (US500, US100, etc.)
- ✅ Stocks (AAPL, MSFT, etc.)

### Advanced Trading
- ✅ Multi-timeframe analysis (1h entry + 4h trend filter)
- ✅ Technical indicators (ADX, EMA, RSI, ATR)
- ✅ Rule-based strategy (no AI API needed)
- ✅ Risk management (position sizing, SL/TP)
- ✅ News filtering (pauses during high-impact events)

### Real-Time Data
- ✅ Live bid/ask prices
- ✅ Historical OHLC candles
- ✅ Position monitoring
- ✅ Account balance tracking

---

## 🏗️ Architecture

```
Capital.com Watchlist
         ↓
   (Auto-sync every hour)
         ↓
  Trading Bot Cache
         ↓
   Symbol Selection
         ↓
   Market Data Fetch
         ↓
  Technical Analysis
         ↓
   Signal Generation
         ↓
   Risk Management
         ↓
    Order Placement
         ↓
  Position Monitoring
```

---

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Review `QUICK_START_CAPITAL.md`
- [ ] Understand `docs/CAPITAL_COM_INTEGRATION.md`
- [ ] Check `docs/CAPITAL_FLOW_DIAGRAM.md`

### Deployment
- [ ] Copy files to server (use deployment script)
- [ ] Restart backend service
- [ ] Verify backend is running

### Capital.com Setup
- [ ] Create "TradeMinds" watchlist
- [ ] Add symbols to watchlist
- [ ] Verify broker account is configured

### Testing
- [ ] Run `test_capital_connection.py`
- [ ] Verify all tests pass
- [ ] Check watchlist symbols loaded

### Go Live
- [ ] Start the bot
- [ ] Monitor logs
- [ ] Verify trades are being placed

---

## 🧪 Testing

### Test Connection
```bash
cd /root/trademinds
venv/bin/python3 scripts/test_capital_connection.py
```

**Expected Output**:
```
✅ Found Capital.com account
✅ Connected successfully
✅ Account info retrieved
✅ Found 1 watchlist(s)
✅ Found TradeMinds watchlist
✅ Found 50 market(s) in TradeMinds watchlist
✅ Tick data retrieved
✅ Candle data retrieved
✅ Watchlist cached
✅ All tests passed!
```

### Test Watchlist Sync
```bash
cd /root/trademinds
venv/bin/python3 scripts/update_capital_watchlist.py
```

**Expected Output**:
```
Found Capital.com account: <id>
Found 1 watchlists
Found TradeMinds watchlist: <id>
Found 50 markets in TradeMinds watchlist:
  - EURUSD
  - GBPUSD
  ...
Updated strategy 'My Strategy' with 50 symbols
✅ Successfully updated strategies
```

---

## 🔍 Troubleshooting

### Connection Issues

**Problem**: "Failed to connect to Capital.com"
**Solution**: 
1. Check API credentials in broker account
2. Verify Capital.com API is accessible
3. Check firewall settings

### Watchlist Issues

**Problem**: "TradeMinds watchlist not found"
**Solution**:
1. Create watchlist on Capital.com
2. Name it exactly "TradeMinds"
3. Add at least one symbol

**Problem**: "No markets found in watchlist"
**Solution**:
1. Open Capital.com app
2. Go to TradeMinds watchlist
3. Add symbols

### Trading Issues

**Problem**: "Bot not trading Capital.com symbols"
**Solution**:
1. Check broker account is active
2. Verify strategy is active
3. Ensure strategy.symbols is null (to use watchlist)
4. Check bot logs for errors

### Data Issues

**Problem**: "Failed to get tick data"
**Solution**:
1. Verify symbol epic code is correct
2. Check if market is open
3. Review Capital.com API status

---

## 📊 Configuration

### Environment Variables (.env)
```bash
CAPITAL_API_KEY=your_identifier
CAPITAL_API_SECRET=your_password
CAPITAL_BASE_URL=https://api-capital.backend-capital.com/api/v1
```

### Broker Account
```json
{
  "broker_type": "capital.com",
  "account_name": "Capital.com Demo",
  "market_type": "forex",
  "api_key": "your_identifier",
  "api_secret": "your_password",
  "is_demo": true,
  "is_active": true
}
```

### Strategy
```json
{
  "name": "Capital.com Strategy",
  "strategy_type": "trend_following",
  "markets": ["forex", "crypto", "commodity", "index"],
  "symbols": null,  // null = use watchlist
  "parameters": {
    "adx_threshold": 25,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "ema_fast": 50,
    "ema_slow": 200,
    "htf_trend_required": true
  },
  "is_active": true
}
```

### Bot Config
```json
{
  "trade_mode": "live",
  "max_positions": 25,
  "max_daily_loss_pct": 5.0,
  "max_risk_per_trade_pct": 1.0,
  "pause_on_high_impact_news": true,
  "news_pause_minutes": 30,
  "market_limits": {
    "forex": 10,
    "crypto": 5,
    "commodity": 4,
    "stock": 4,
    "index": 2
  }
}
```

---

## 🎓 Learning Resources

### Beginner
1. Start with `QUICK_START_CAPITAL.md`
2. Run `test_capital_connection.py`
3. Create watchlist on Capital.com
4. Start bot and monitor

### Intermediate
1. Read `docs/CAPITAL_COM_INTEGRATION.md`
2. Review `docs/CAPITAL_FLOW_DIAGRAM.md`
3. Understand risk management
4. Customize strategy parameters

### Advanced
1. Study `CHANGELOG_CAPITAL_INTEGRATION.md`
2. Review `backend/brokers/capital_adapter.py`
3. Modify `backend/bot/trading_bot.py`
4. Implement custom strategies

---

## 🚨 Important Notes

### Security
- API credentials are encrypted in database
- Session tokens stored in memory only
- HTTPS used for all API communication
- Never commit credentials to git

### Rate Limits
- Capital.com has API rate limits
- Adapter handles rate limiting automatically
- Bot scans every 60 seconds (safe rate)

### Market Hours
- Some markets only available during specific hours
- Bot will skip closed markets automatically
- Check Capital.com for market hours

### Demo vs Live
- Test with demo account first
- Demo may have different spreads
- Switch to live when confident

### Watchlist Sync
- Automatic refresh every hour
- Manual sync available via script
- Changes on Capital.com reflected in bot

---

## 📞 Support

### Check Logs
```bash
tail -f /tmp/trademinds.log
tail -f /tmp/trademinds.log | grep -i capital
tail -f /tmp/trademinds.log | grep -E "(error|warning)"
```

### Test Connection
```bash
python3 scripts/test_capital_connection.py
```

### Manual Watchlist Sync
```bash
python3 scripts/update_capital_watchlist.py
```

### Restart Services
```bash
supervisorctl restart trademinds
```

### Check Status
```bash
curl http://localhost:8001/health
curl http://localhost:8001/api/bot/status -H "Authorization: Bearer TOKEN"
```

---

## 🎉 Success Criteria

You'll know the integration is working when:

1. ✅ Test script passes all tests
2. ✅ Bot logs show "Connected to Capital.com"
3. ✅ Bot logs show "Loaded X symbols from TradeMinds watchlist"
4. ✅ Signals are being generated for Capital.com symbols
5. ✅ Orders are being placed on Capital.com
6. ✅ Positions are being monitored and closed

---

## 📈 Next Steps

After successful deployment:

1. **Monitor Performance**
   - Track win rate
   - Review PnL
   - Analyze signal quality

2. **Optimize Strategy**
   - Adjust parameters
   - Test different timeframes
   - Refine risk management

3. **Scale Up**
   - Add more symbols to watchlist
   - Increase position sizes
   - Expand to more markets

4. **Automate**
   - Set up alerts
   - Configure notifications
   - Schedule reports

---

## 🏆 Best Practices

### Watchlist Management
- Keep watchlist focused (20-50 symbols)
- Group by market type
- Remove low-volume symbols
- Update regularly

### Risk Management
- Start with small position sizes
- Use proper stop losses
- Don't risk more than 1% per trade
- Monitor daily loss limits

### Monitoring
- Check logs daily
- Review trades weekly
- Analyze performance monthly
- Adjust strategy as needed

### Testing
- Always test on demo first
- Verify all features work
- Run for at least 1 week
- Review results before going live

---

## 📝 Version History

**v1.0.0** (April 18, 2026)
- Initial Capital.com integration
- Automatic watchlist support
- Multi-asset trading
- Risk management
- Complete documentation

---

## 🙏 Credits

- **Capital.com API**: [https://open-api.capital.com/](https://open-api.capital.com/)
- **TradeMinds**: Trading bot platform
- **Developer**: Integration and documentation

---

## 📄 License

This integration is part of the TradeMinds project.

---

## 🔗 Quick Links

- [Capital.com Website](https://capital.com)
- [Capital.com API Docs](https://open-api.capital.com/)
- [Quick Start Guide](QUICK_START_CAPITAL.md)
- [Full Documentation](docs/CAPITAL_COM_INTEGRATION.md)
- [Flow Diagrams](docs/CAPITAL_FLOW_DIAGRAM.md)
- [Changelog](CHANGELOG_CAPITAL_INTEGRATION.md)

---

**Ready to trade? Let's go! 🚀**
