# Capital.com Integration Guide

## Overview

TradeMinds now supports Capital.com as a broker for trading Forex, Indices, Commodities, Stocks, and Crypto CFDs.

## Features

- ✅ Demo and Live account support
- ✅ Real-time market data (bid/ask prices)
- ✅ Historical OHLC candles (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
- ✅ Market orders with Stop Loss and Take Profit
- ✅ Position management (open/close)
- ✅ Account balance and equity tracking
- ✅ **Watchlist integration** - Trade symbols from your Capital.com watchlists

## Setup

### 1. Create Capital.com Account

1. Sign up at [Capital.com](https://capital.com)
2. Enable API access in your account settings
3. Get your API credentials (identifier and password)

### 2. Add Broker Account in TradeMinds

```bash
# Login to TradeMinds
curl -X POST http://your-server:8001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"your@email.com","password":"yourpassword"}'

# Add Capital.com broker account
curl -X POST http://your-server:8001/api/brokers \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "broker_type": "capital.com",
    "account_name": "Capital.com Demo",
    "market_type": "forex",
    "api_key": "your_capital_identifier",
    "api_secret": "your_capital_password",
    "is_demo": true
  }'
```

### 3. Configure Environment Variables

Add to your `.env` file:

```bash
# Capital.com API (optional - can be set per broker account)
CAPITAL_API_KEY=your_identifier
CAPITAL_API_SECRET=your_password
CAPITAL_BASE_URL=https://api-capital.backend-capital.com/api/v1
```

## Watchlist Integration

### How It Works

1. **Create a watchlist** on Capital.com web/mobile app named **"TradeMinds"**
2. **Add symbols** you want to trade to this watchlist
3. TradeMinds will automatically:
   - Fetch symbols from the watchlist on bot startup
   - Refresh the list every hour
   - Trade only the symbols in your watchlist

### Manual Watchlist Update

If you want to manually update the strategy with watchlist symbols:

```bash
# On the server
cd /root/trademinds
python3 scripts/update_capital_watchlist.py
```

This script will:
- Connect to Capital.com
- Find the "TradeMinds" watchlist
- Extract all market epics
- Update all active strategies with these symbols

### Automatic Watchlist Loading

The bot automatically loads the watchlist when it starts. No manual intervention needed!

## Symbol Format

Capital.com uses "epic" codes for symbols. Examples:

- **Forex**: `EURUSD`, `GBPUSD`, `USDJPY`
- **Indices**: `US500`, `US100`, `US30`, `GER40`
- **Commodities**: `GOLD`, `SILVER`, `OIL_BRENT`, `NATURALGAS`
- **Crypto**: `BTCUSD`, `ETHUSD`, `XRPUSD`
- **Stocks**: `AAPL`, `MSFT`, `GOOGL`, `TSLA`

## API Endpoints

### Get Watchlists

```bash
curl -X GET http://your-server:8001/api/brokers/capital/watchlists \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Get Watchlist Markets

```bash
curl -X GET http://your-server:8001/api/brokers/capital/watchlists/{watchlist_id}/markets \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## Trading Flow

1. **Bot starts** → Connects to Capital.com → Loads "TradeMinds" watchlist
2. **Every minute** → Scans symbols from watchlist
3. **For each symbol**:
   - Fetches 1h candles (entry timeframe)
   - Fetches 4h candles (higher timeframe trend filter)
   - Calculates indicators (ADX, EMA, RSI, etc.)
   - Applies strategy filters
   - Generates trading signal (rule-based or AI)
   - Places order if signal is strong enough
4. **Position management** → Monitors open positions, syncs with broker

## Configuration

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

### Strategy Config

```json
{
  "name": "Capital.com Multi-Market",
  "strategy_type": "trend_following",
  "markets": ["forex", "crypto", "commodity", "index"],
  "symbols": null,  // Leave null to use watchlist
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

## Troubleshooting

### Connection Issues

```bash
# Check if Capital.com API is reachable
curl -I https://api-capital.backend-capital.com/api/v1/session

# Check bot logs
tail -f /tmp/trademinds.log | grep -i capital
```

### Watchlist Not Found

1. Ensure watchlist is named exactly **"TradeMinds"** (case-insensitive)
2. Check if watchlist has markets added
3. Run the manual update script to verify

### No Trades Being Placed

1. Check if bot is running: `GET /api/bot/status`
2. Verify strategy is active and has correct markets
3. Check signal logs: `GET /api/signals`
4. Review risk limits (daily loss, max positions)

## Market Types

Capital.com supports multiple asset classes:

| Market Type | Examples | Max Positions |
|-------------|----------|---------------|
| Forex | EURUSD, GBPUSD, USDJPY | 10 |
| Crypto | BTCUSD, ETHUSD, XRPUSD | 5 |
| Commodity | GOLD, SILVER, OIL | 4 |
| Stock | AAPL, MSFT, GOOGL | 4 |
| Index | US500, US100, GER40 | 2 |

## Commission & Spreads

Capital.com charges:
- **Spread**: Variable, depends on market conditions
- **Commission**: None for most CFDs
- **Overnight fees**: Applied to positions held overnight

The backtest engine includes realistic commission estimates:
- Forex: 0.0001% of trade value
- Crypto: 0.1% of trade value
- Commodities: 0.01% of trade value
- Stocks: 0.05% of trade value
- Indices: 0.01% of trade value

## Security

- API credentials are encrypted in the database
- Session tokens (CST, X-SECURITY-TOKEN) are stored in memory only
- HTTPS is used for all API communication
- Tokens expire after inactivity

## Limitations

- **Rate limits**: Capital.com has API rate limits (handled automatically)
- **Market hours**: Some markets are only available during specific hours
- **Demo account**: May have different spreads/execution than live
- **Watchlist sync**: Updates every hour (not real-time)

## Next Steps

1. ✅ Create Capital.com account
2. ✅ Add broker in TradeMinds
3. ✅ Create "TradeMinds" watchlist
4. ✅ Add symbols to watchlist
5. ✅ Start the bot
6. ✅ Monitor trades and performance

## Support

For issues or questions:
- Check logs: `/tmp/trademinds.log`
- Review API docs: [Capital.com API Documentation](https://open-api.capital.com/)
- Contact support: support@trademinds.com
