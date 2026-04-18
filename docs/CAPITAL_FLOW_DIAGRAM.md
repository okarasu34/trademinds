# Capital.com Integration - Flow Diagram

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         TradeMinds Bot                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 1. Start Bot
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Connect to Capital.com                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  POST /api/v1/session                                     │  │
│  │  → Get CST + X-SECURITY-TOKEN                             │  │
│  │  → Store in memory                                        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 2. Load Watchlist (Background)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Load TradeMinds Watchlist                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  GET /api/v1/watchlists                                   │  │
│  │  → Find "TradeMinds" watchlist                            │  │
│  │  → Get watchlist ID                                       │  │
│  │                                                            │  │
│  │  GET /api/v1/watchlists/{id}                              │  │
│  │  → Get all market epics                                   │  │
│  │  → Cache in memory (expires in 1 hour)                    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 3. Main Trading Loop (Every 60s)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Get Symbols to Trade                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. Check Strategy.symbols                                │  │
│  │     ├─ If set → Use those symbols                         │  │
│  │     └─ If null → Get from adapter cache                   │  │
│  │                                                            │  │
│  │  2. Get cached watchlist symbols                          │  │
│  │     ├─ If cache valid → Return symbols                    │  │
│  │     └─ If expired → Refresh in background                 │  │
│  │                                                            │  │
│  │  3. Fallback to default symbols                           │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 4. For Each Symbol
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Fetch Market Data                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  GET /api/v1/markets/{symbol}                             │  │
│  │  → Get current bid/ask/spread                             │  │
│  │                                                            │  │
│  │  GET /api/v1/prices/{symbol}?resolution=HOUR&max=200      │  │
│  │  → Get 1h candles (entry timeframe)                       │  │
│  │                                                            │  │
│  │  GET /api/v1/prices/{symbol}?resolution=HOUR_4&max=100    │  │
│  │  → Get 4h candles (trend filter)                          │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 5. Calculate Indicators
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Technical Analysis                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Entry Timeframe (1h):                                    │  │
│  │  ├─ ADX (Wilder's smoothing)                              │  │
│  │  ├─ EMA 50 / EMA 200                                      │  │
│  │  ├─ RSI 14                                                │  │
│  │  ├─ ATR 14                                                │  │
│  │  └─ Patterns (engulfing, doji, etc.)                      │  │
│  │                                                            │  │
│  │  Higher Timeframe (4h):                                   │  │
│  │  ├─ ADX                                                    │  │
│  │  ├─ EMA 50 / EMA 200                                      │  │
│  │  └─ Trend direction (bullish/bearish/neutral)             │  │
│  │                                                            │  │
│  │  Apply Strategy Filters:                                  │  │
│  │  ├─ ADX threshold (default: 25)                           │  │
│  │  ├─ RSI levels (30/70)                                    │  │
│  │  └─ HTF trend required (yes/no)                           │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 6. Generate Signal
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Rule-Based Strategy                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Check Conditions:                                        │  │
│  │  ├─ ADX > threshold (trend strength)                      │  │
│  │  ├─ EMA 50 > EMA 200 (bullish) or < (bearish)            │  │
│  │  ├─ RSI not overbought/oversold                           │  │
│  │  ├─ HTF trend matches entry direction                     │  │
│  │  └─ No high-impact news in next 30 min                    │  │
│  │                                                            │  │
│  │  Generate Signal:                                         │  │
│  │  ├─ "buy" (confidence 0.7-0.9)                            │  │
│  │  ├─ "sell" (confidence 0.7-0.9)                           │  │
│  │  └─ "hold" (confidence < 0.65)                            │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 7. Risk Management
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Risk Checks                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. Calculate Position Size:                              │  │
│  │     ├─ Risk = Account Balance × Risk %                    │  │
│  │     ├─ SL Distance = |Entry - Stop Loss|                  │  │
│  │     └─ Lot Size = Risk / SL Distance                      │  │
│  │                                                            │  │
│  │  2. Validate SL/TP:                                       │  │
│  │     ├─ SL: Entry ± (1.5 × ATR)                            │  │
│  │     └─ TP: Entry ± (3.0 × ATR)                            │  │
│  │                                                            │  │
│  │  3. Check Limits:                                         │  │
│  │     ├─ Daily loss < 5%                                    │  │
│  │     ├─ Open positions < max_positions                     │  │
│  │     ├─ Market-specific limits (forex: 10, crypto: 5)      │  │
│  │     └─ No duplicate symbols                               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 8. Place Order (if approved)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Execute Trade                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  POST /api/v1/positions                                   │  │
│  │  {                                                         │  │
│  │    "epic": "EURUSD",                                      │  │
│  │    "direction": "BUY",                                    │  │
│  │    "size": 0.1,                                           │  │
│  │    "stopLevel": 1.0950,                                   │  │
│  │    "profitLevel": 1.1050                                  │  │
│  │  }                                                         │  │
│  │  → Get dealReference                                      │  │
│  │  → Save to database                                       │  │
│  │  → Send notification                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 9. Monitor Positions (Every 60s)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Position Sync                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  GET /api/v1/positions                                    │  │
│  │  → Get all open positions from broker                     │  │
│  │                                                            │  │
│  │  For each position in database:                           │  │
│  │  ├─ If not in broker → Mark as closed                     │  │
│  │  ├─ If in broker → Update PnL                             │  │
│  │  └─ If SL/TP hit → Close and record                       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 10. Repeat Loop
                              └──────────────┐
                                             │
                                             ▼
                                    Wait 60 seconds
                                             │
                                             └─────► Back to Step 3
```

## Watchlist Update Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Manual Watchlist Update                      │
│                (scripts/update_capital_watchlist.py)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 1. Connect to Database
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Find Capital.com Broker Account                                │
│  ├─ broker_type = "capital.com"                                 │
│  ├─ is_active = True                                            │
│  └─ Get encrypted credentials                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 2. Connect to Capital.com
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Login to Capital.com API                                       │
│  ├─ POST /api/v1/session                                        │
│  ├─ Get CST + X-SECURITY-TOKEN                                  │
│  └─ Store tokens                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 3. Get Watchlists
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Fetch All Watchlists                                           │
│  ├─ GET /api/v1/watchlists                                      │
│  ├─ Find "TradeMinds" (case-insensitive)                        │
│  └─ Get watchlist ID                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 4. Get Markets
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Fetch Watchlist Markets                                        │
│  ├─ GET /api/v1/watchlists/{id}                                 │
│  ├─ Extract all epic codes                                      │
│  └─ Log symbols found                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 5. Update Strategies
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Update Database                                                │
│  ├─ Find all active strategies for user                         │
│  ├─ Set Strategy.symbols = [epic1, epic2, ...]                  │
│  ├─ Commit to database                                          │
│  └─ Log success                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 6. Disconnect
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Cleanup                                                        │
│  ├─ DELETE /api/v1/session                                      │
│  ├─ Close HTTP session                                          │
│  └─ Exit                                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Capital.com Watchlist
         │
         │ (Automatic on bot start)
         │ (Refresh every 1 hour)
         ▼
┌─────────────────────┐
│  Adapter Cache      │
│  (In Memory)        │
│  ├─ Symbols: []     │
│  └─ Timestamp       │
└─────────────────────┘
         │
         │ (Bot queries)
         ▼
┌─────────────────────┐
│  Trading Bot        │
│  _get_symbols()     │
└─────────────────────┘
         │
         │ (Scans each symbol)
         ▼
┌─────────────────────┐
│  Market Data        │
│  ├─ Tick            │
│  ├─ 1h Candles      │
│  └─ 4h Candles      │
└─────────────────────┘
         │
         │ (Calculates)
         ▼
┌─────────────────────┐
│  Indicators         │
│  ├─ ADX             │
│  ├─ EMA             │
│  ├─ RSI             │
│  └─ ATR             │
└─────────────────────┘
         │
         │ (Generates)
         ▼
┌─────────────────────┐
│  Trading Signal     │
│  ├─ buy/sell/hold   │
│  ├─ Confidence      │
│  ├─ SL/TP           │
│  └─ Reasoning       │
└─────────────────────┘
         │
         │ (If approved)
         ▼
┌─────────────────────┐
│  Capital.com Order  │
│  ├─ Market Order    │
│  ├─ Stop Loss       │
│  └─ Take Profit     │
└─────────────────────┘
         │
         │ (Saved to)
         ▼
┌─────────────────────┐
│  Database           │
│  ├─ Trade Record    │
│  ├─ Signal Log      │
│  └─ Health Log      │
└─────────────────────┘
```

## Component Interaction

```
┌──────────────────┐
│   Frontend       │
│   (React)        │
└────────┬─────────┘
         │ HTTP/WebSocket
         ▼
┌──────────────────┐
│   Backend API    │
│   (FastAPI)      │
└────────┬─────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌──────────────────┐  ┌──────────────────┐
│  Trading Bot     │  │  Database        │
│  (Async Loop)    │  │  (PostgreSQL)    │
└────────┬─────────┘  └──────────────────┘
         │
         ├─────────────────┬─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Capital Adapter │  │  Risk Manager    │  │  AI Engine       │
│  (Broker API)    │  │  (Limits)        │  │  (Rule-Based)    │
└────────┬─────────┘  └──────────────────┘  └──────────────────┘
         │
         ▼
┌──────────────────┐
│  Capital.com API │
│  (REST)          │
└──────────────────┘
```

## File Structure

```
trademinds/
├── backend/
│   ├── brokers/
│   │   ├── base_adapter.py          ← Updated (factory)
│   │   ├── capital_adapter.py       ← NEW (Capital.com)
│   │   ├── metaapi_adapter.py
│   │   └── ibkr_adapter.py
│   ├── bot/
│   │   ├── trading_bot.py           ← Updated (_get_symbols)
│   │   ├── ai_engine.py
│   │   ├── indicators.py
│   │   └── backtest_engine.py
│   └── ...
├── scripts/
│   ├── update_capital_watchlist.py  ← NEW (manual sync)
│   ├── deploy_capital_integration.sh    ← NEW (bash deploy)
│   └── deploy_capital_integration.ps1   ← NEW (powershell deploy)
├── docs/
│   ├── CAPITAL_COM_INTEGRATION.md   ← NEW (full guide)
│   └── CAPITAL_FLOW_DIAGRAM.md      ← NEW (this file)
├── CHANGELOG_CAPITAL_INTEGRATION.md ← NEW (changelog)
├── QUICK_START_CAPITAL.md           ← NEW (quick start)
└── SUMMARY.md                        ← NEW (summary)
```

## Sequence Diagram

```
User          Frontend      Backend       TradingBot    CapitalAdapter    Capital.com
 │                │            │              │               │                │
 │ Start Bot      │            │              │               │                │
 ├───────────────>│            │              │               │                │
 │                │ POST /bot/start           │               │                │
 │                ├───────────>│              │               │                │
 │                │            │ start()      │               │                │
 │                │            ├─────────────>│               │                │
 │                │            │              │ connect()     │                │
 │                │            │              ├──────────────>│                │
 │                │            │              │               │ POST /session  │
 │                │            │              │               ├───────────────>│
 │                │            │              │               │ CST + Token    │
 │                │            │              │               │<───────────────┤
 │                │            │              │               │ load_watchlist()│
 │                │            │              │               ├───────────────>│
 │                │            │              │               │ GET /watchlists│
 │                │            │              │               ├───────────────>│
 │                │            │              │               │ watchlists     │
 │                │            │              │               │<───────────────┤
 │                │            │              │               │ GET /watchlists/{id}
 │                │            │              │               ├───────────────>│
 │                │            │              │               │ markets        │
 │                │            │              │               │<───────────────┤
 │                │            │              │ connected     │                │
 │                │            │              │<──────────────┤                │
 │                │            │ started      │               │                │
 │                │            │<─────────────┤               │                │
 │                │ 200 OK     │              │               │                │
 │                │<───────────┤              │               │                │
 │ Bot Running    │            │              │               │                │
 │<───────────────┤            │              │               │                │
 │                │            │              │               │                │
 │                │            │ [Every 60s]  │               │                │
 │                │            │              │ _scan_and_execute()            │
 │                │            │              ├──────────────>│                │
 │                │            │              │ get_tick()    │                │
 │                │            │              ├──────────────>│                │
 │                │            │              │               │ GET /markets/{symbol}
 │                │            │              │               ├───────────────>│
 │                │            │              │               │ bid/ask        │
 │                │            │              │               │<───────────────┤
 │                │            │              │ tick          │                │
 │                │            │              │<──────────────┤                │
 │                │            │              │ get_candles() │                │
 │                │            │              ├──────────────>│                │
 │                │            │              │               │ GET /prices/{symbol}
 │                │            │              │               ├───────────────>│
 │                │            │              │               │ OHLCV          │
 │                │            │              │               │<───────────────┤
 │                │            │              │ candles       │                │
 │                │            │              │<──────────────┤                │
 │                │            │              │ [Calculate indicators]         │
 │                │            │              │ [Generate signal]              │
 │                │            │              │ [Check risk]  │                │
 │                │            │              │ place_order() │                │
 │                │            │              ├──────────────>│                │
 │                │            │              │               │ POST /positions│
 │                │            │              │               ├───────────────>│
 │                │            │              │               │ dealReference  │
 │                │            │              │               │<───────────────┤
 │                │            │              │ order_id      │                │
 │                │            │              │<──────────────┤                │
 │                │            │              │ [Save to DB]  │                │
 │                │            │              │ [Send notification]            │
 │                │            │              │               │                │
```

---

**Legend**:
- `→` : Synchronous call
- `├─>` : Async call
- `<─┤` : Response
- `[...]` : Internal operation
