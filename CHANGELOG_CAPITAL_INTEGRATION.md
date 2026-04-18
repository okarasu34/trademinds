# Capital.com Integration - Changelog

## Date: April 18, 2026

### Summary
Completed Capital.com broker integration with automatic watchlist support. The bot can now trade symbols from your Capital.com "TradeMinds" watchlist automatically.

---

## Files Created

### 1. `backend/brokers/capital_adapter.py`
**Purpose**: Capital.com broker adapter implementing the BrokerAdapter interface

**Features**:
- ✅ Authentication with CST and X-SECURITY-TOKEN
- ✅ Account info (balance, equity, margin)
- ✅ Real-time tick data (bid/ask/spread)
- ✅ Historical OHLC candles (1m to 1w)
- ✅ Market order placement with SL/TP
- ✅ Position management (open/close)
- ✅ Watchlist integration
- ✅ Automatic watchlist caching (refreshes hourly)

**Key Methods**:
- `connect()` - Login and get session tokens
- `get_account_info()` - Fetch balance and equity
- `get_tick()` - Get current bid/ask prices
- `get_candles()` - Fetch OHLC data
- `place_order()` - Place market order with SL/TP
- `close_order()` - Close open position
- `get_open_orders()` - List all open positions
- `get_watchlists()` - Fetch all watchlists
- `get_watchlist_markets()` - Get symbols from a watchlist
- `load_trademinds_watchlist()` - Load and cache TradeMinds watchlist
- `get_cached_watchlist_symbols()` - Get cached symbols (auto-refresh)

### 2. `scripts/update_capital_watchlist.py`
**Purpose**: Manual script to fetch Capital.com watchlist and update strategies

**Usage**:
```bash
cd /root/trademinds/backend
venv/bin/python3 ../scripts/update_capital_watchlist.py
```

**What it does**:
1. Connects to Capital.com
2. Finds "TradeMinds" watchlist
3. Extracts all market epics
4. Updates all active strategies with these symbols

### 3. `docs/CAPITAL_COM_INTEGRATION.md`
**Purpose**: Complete documentation for Capital.com integration

**Contents**:
- Setup instructions
- Watchlist integration guide
- Symbol format reference
- API endpoints
- Trading flow explanation
- Configuration examples
- Troubleshooting guide
- Market types and limits
- Commission structure
- Security notes

---

## Files Modified

### 1. `backend/brokers/base_adapter.py`
**Changes**: Added Capital.com to the adapter factory

```python
# Capital.com CFD broker
if broker_type in ["capital", "capital.com", "capitalcom"]:
    from brokers.capital_adapter import CapitalAdapter
    return CapitalAdapter(account)
```

### 2. `backend/bot/trading_bot.py`
**Changes**: Updated `_get_symbols()` method to support dynamic watchlist loading

**Before**:
```python
def _get_symbols(self, strategy, market_type):
    if strategy.symbols:
        return strategy.symbols
    return {
        "forex": ["EURUSD", "GBPUSD", ...],
        ...
    }.get(market_type, [])
```

**After**:
```python
def _get_symbols(self, strategy, market_type):
    """Get symbols to trade - from strategy config or adapter watchlist"""
    if strategy.symbols:
        return strategy.symbols
    
    # For Capital.com, try to get symbols from watchlist
    adapter = self.adapters.get(market_type)
    if adapter and hasattr(adapter, 'get_watchlist_symbols'):
        try:
            symbols = adapter.get_cached_watchlist_symbols()
            if symbols:
                return symbols
        except Exception as e:
            logger.warning(f"Failed to get watchlist symbols: {e}")
    
    # Fallback to default symbols
    return {
        "forex": ["EURUSD", "GBPUSD", ...],
        ...
    }.get(market_type, [])
```

---

## How It Works

### Automatic Watchlist Integration

1. **Bot Startup**:
   - Bot connects to Capital.com
   - Adapter automatically loads "TradeMinds" watchlist in background
   - Symbols are cached in memory

2. **Symbol Selection**:
   - When bot scans for trading opportunities
   - First checks if strategy has explicit symbols configured
   - If not, checks if adapter has cached watchlist symbols
   - Falls back to default symbols if watchlist unavailable

3. **Cache Refresh**:
   - Watchlist cache expires after 1 hour
   - Automatically refreshes in background
   - No interruption to trading

### Manual Update (Optional)

If you prefer to store watchlist symbols in the database:

```bash
python3 scripts/update_capital_watchlist.py
```

This updates the `Strategy.symbols` field directly.

---

## Configuration

### Environment Variables (.env)

```bash
# Capital.com API
CAPITAL_API_KEY=your_identifier
CAPITAL_API_SECRET=your_password
CAPITAL_BASE_URL=https://api-capital.backend-capital.com/api/v1
```

### Broker Account Setup

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

### Strategy Setup

```json
{
  "name": "Capital.com Strategy",
  "strategy_type": "trend_following",
  "markets": ["forex", "crypto", "commodity", "index"],
  "symbols": null,  // null = use watchlist
  "is_active": true
}
```

---

## Deployment Steps

### On Local Machine (Desktop)

1. ✅ Files already created in `C:\Users\ozckr\Desktop\trademinds\`
2. ✅ Capital.com adapter implemented
3. ✅ Trading bot updated with watchlist support
4. ✅ Documentation created

### On Server (Vultr)

**Copy files to server**:

```bash
# From local machine
scp backend/brokers/capital_adapter.py root@your-server:/root/trademinds/backend/brokers/
scp backend/brokers/base_adapter.py root@your-server:/root/trademinds/backend/brokers/
scp backend/bot/trading_bot.py root@your-server:/root/trademinds/backend/bot/
scp scripts/update_capital_watchlist.py root@your-server:/root/trademinds/scripts/
```

**Or manually update on server**:

```bash
# SSH to server
ssh root@your-server

# Navigate to project
cd /root/trademinds

# Create/update files using nano or vim
nano backend/brokers/capital_adapter.py
# (paste content)

nano backend/brokers/base_adapter.py
# (update factory function)

nano backend/bot/trading_bot.py
# (update _get_symbols method)

# Create scripts directory if needed
mkdir -p scripts
nano scripts/update_capital_watchlist.py
# (paste content)
```

**Restart services**:

```bash
# Stop bot
curl -X POST http://localhost:8001/api/bot/stop \
  -H "Authorization: Bearer YOUR_TOKEN"

# Restart backend
supervisorctl restart trademinds

# Wait for backend to start
sleep 10

# Start bot
curl -X POST http://localhost:8001/api/bot/start \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Verify**:

```bash
# Check bot status
curl http://localhost:8001/api/bot/status \
  -H "Authorization: Bearer YOUR_TOKEN"

# Check logs
tail -f /tmp/trademinds.log | grep -i capital
```

---

## Testing

### 1. Test Capital.com Connection

```bash
cd /root/trademinds/backend
venv/bin/python3 -c "
import asyncio
from brokers.capital_adapter import CapitalAdapter
from db.models import BrokerAccount

async def test():
    # Create mock account (replace with real credentials)
    account = BrokerAccount(
        broker_type='capital.com',
        encrypted_api_key='your_encrypted_key',
        encrypted_api_secret='your_encrypted_secret'
    )
    
    adapter = CapitalAdapter(account)
    connected = await adapter.connect()
    print(f'Connected: {connected}')
    
    if connected:
        info = await adapter.get_account_info()
        print(f'Balance: {info.balance} {info.currency}')
        
        watchlists = await adapter.get_watchlists()
        print(f'Watchlists: {[w[\"name\"] for w in watchlists]}')
        
        await adapter.disconnect()

asyncio.run(test())
"
```

### 2. Test Watchlist Loading

```bash
python3 scripts/update_capital_watchlist.py
```

Expected output:
```
Found Capital.com account: <account_id>
Found 1 watchlists
Found TradeMinds watchlist: <watchlist_id>
Found 50 markets in TradeMinds watchlist:
  - EURUSD
  - GBPUSD
  - USDJPY
  ...
Updated strategy 'Capital.com Strategy' with 50 symbols
✅ Successfully updated strategies with Capital.com watchlist symbols
```

### 3. Test Bot Trading

```bash
# Start bot
curl -X POST http://localhost:8001/api/bot/start \
  -H "Authorization: Bearer YOUR_TOKEN"

# Monitor logs
tail -f /tmp/trademinds.log

# Check signals
curl http://localhost:8001/api/signals \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## Troubleshooting

### Issue: "Capital.com login failed: 401"
**Solution**: Check API credentials in broker account settings

### Issue: "TradeMinds watchlist not found"
**Solution**: 
1. Login to Capital.com web/mobile app
2. Create a watchlist named "TradeMinds" (exact name)
3. Add symbols to the watchlist
4. Run update script again

### Issue: "No markets found in TradeMinds watchlist"
**Solution**: Add symbols to your Capital.com watchlist

### Issue: "Bot not trading Capital.com symbols"
**Solution**:
1. Check if broker account is active and connected
2. Verify strategy has correct market types
3. Ensure strategy.symbols is null (to use watchlist)
4. Check bot logs for errors

---

## Next Steps

1. ✅ Copy files to server
2. ✅ Restart backend service
3. ✅ Create "TradeMinds" watchlist on Capital.com
4. ✅ Add symbols to watchlist
5. ✅ Run update script (optional)
6. ✅ Start bot
7. ✅ Monitor trades

---

## Notes

- **Watchlist sync**: Automatic, refreshes every hour
- **Symbol format**: Use Capital.com epic codes (e.g., "EURUSD", "BTCUSD")
- **Market types**: Forex, Crypto, Commodity, Stock, Index
- **Demo vs Live**: Set `is_demo` flag in broker account
- **Security**: API credentials are encrypted in database

---

## Support

For questions or issues:
- Check logs: `/tmp/trademinds.log`
- Review documentation: `docs/CAPITAL_COM_INTEGRATION.md`
- Test connection: `scripts/update_capital_watchlist.py`
