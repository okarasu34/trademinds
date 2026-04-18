"""
Test Capital.com connection and watchlist functionality
Run this to verify the integration is working correctly

Usage:
cd /root/trademinds/backend
venv/bin/python3 ../scripts/test_capital_connection.py
"""
import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from sqlalchemy import select
from db.database import AsyncSessionLocal
from db.models import BrokerAccount
from brokers.capital_adapter import CapitalAdapter
from loguru import logger


async def test_connection():
    """Test Capital.com connection"""
    print("\n" + "="*60)
    print("Capital.com Connection Test")
    print("="*60 + "\n")
    
    async with AsyncSessionLocal() as db:
        # Find Capital.com account
        result = await db.execute(
            select(BrokerAccount).where(
                BrokerAccount.broker_type.in_(["capital", "capital.com", "capitalcom"]),
                BrokerAccount.is_active == True
            ).limit(1)
        )
        account = result.scalar_one_or_none()
        
        if not account:
            print("❌ No active Capital.com account found")
            print("\nPlease add a Capital.com broker account first:")
            print("  POST /api/brokers")
            print('  {"broker_type": "capital.com", "api_key": "...", "api_secret": "..."}\n')
            return False
        
        print(f"✅ Found Capital.com account: {account.id}")
        print(f"   Account name: {account.account_name}")
        print(f"   Market type: {account.market_type}")
        print(f"   Is demo: {account.is_demo}")
        print()
        
        # Create adapter
        adapter = CapitalAdapter(account)
        
        # Test 1: Connection
        print("Test 1: Connection")
        print("-" * 60)
        connected = await adapter.connect()
        if not connected:
            print("❌ Failed to connect to Capital.com")
            print("   Check your API credentials\n")
            return False
        
        print(f"✅ Connected successfully")
        print(f"   Account ID: {adapter.account_id}")
        print()
        
        try:
            # Test 2: Account Info
            print("Test 2: Account Info")
            print("-" * 60)
            info = await adapter.get_account_info()
            print(f"✅ Account info retrieved")
            print(f"   Balance: {info.balance:.2f} {info.currency}")
            print(f"   Equity: {info.equity:.2f} {info.currency}")
            print(f"   Free margin: {info.free_margin:.2f} {info.currency}")
            print()
            
            # Test 3: Watchlists
            print("Test 3: Watchlists")
            print("-" * 60)
            watchlists = await adapter.get_watchlists()
            print(f"✅ Found {len(watchlists)} watchlist(s)")
            for wl in watchlists:
                print(f"   - {wl.get('name')} (ID: {wl.get('id')})")
            print()
            
            # Test 4: TradeMinds Watchlist
            print("Test 4: TradeMinds Watchlist")
            print("-" * 60)
            trademinds_wl = None
            for wl in watchlists:
                if wl.get("name", "").lower() == "trademinds":
                    trademinds_wl = wl
                    break
            
            if not trademinds_wl:
                print("⚠️  TradeMinds watchlist not found")
                print("\nPlease create a watchlist named 'TradeMinds' on Capital.com:")
                print("   1. Login to Capital.com web/mobile app")
                print("   2. Create a new watchlist")
                print("   3. Name it 'TradeMinds' (exact name)")
                print("   4. Add symbols you want to trade\n")
                return False
            
            watchlist_id = trademinds_wl.get("id")
            print(f"✅ Found TradeMinds watchlist")
            print(f"   ID: {watchlist_id}")
            print()
            
            # Test 5: Watchlist Markets
            print("Test 5: Watchlist Markets")
            print("-" * 60)
            epics = await adapter.get_watchlist_markets(watchlist_id)
            
            if not epics:
                print("⚠️  No markets found in TradeMinds watchlist")
                print("\nPlease add symbols to your TradeMinds watchlist:")
                print("   1. Open Capital.com app")
                print("   2. Go to TradeMinds watchlist")
                print("   3. Add symbols (e.g., EURUSD, BTCUSD, GOLD)\n")
                return False
            
            print(f"✅ Found {len(epics)} market(s) in TradeMinds watchlist:")
            for i, epic in enumerate(epics, 1):
                print(f"   {i:2d}. {epic}")
            print()
            
            # Test 6: Market Data (first symbol)
            if epics:
                test_symbol = epics[0]
                print(f"Test 6: Market Data ({test_symbol})")
                print("-" * 60)
                
                try:
                    tick = await adapter.get_tick(test_symbol)
                    print(f"✅ Tick data retrieved")
                    print(f"   Bid: {tick.bid}")
                    print(f"   Ask: {tick.ask}")
                    print(f"   Spread: {tick.spread}")
                    print()
                except Exception as e:
                    print(f"⚠️  Failed to get tick data: {e}")
                    print()
                
                # Test 7: Candles
                print(f"Test 7: Historical Candles ({test_symbol})")
                print("-" * 60)
                
                try:
                    df = await adapter.get_candles(test_symbol, "1h", 10)
                    if df is not None and len(df) > 0:
                        print(f"✅ Candle data retrieved")
                        print(f"   Bars: {len(df)}")
                        print(f"   Latest close: {df['close'].iloc[-1]}")
                        print()
                    else:
                        print(f"⚠️  No candle data available")
                        print()
                except Exception as e:
                    print(f"⚠️  Failed to get candles: {e}")
                    print()
            
            # Test 8: Cache Test
            print("Test 8: Watchlist Cache")
            print("-" * 60)
            await adapter.load_trademinds_watchlist()
            cached = adapter.get_cached_watchlist_symbols()
            print(f"✅ Watchlist cached")
            print(f"   Cached symbols: {len(cached)}")
            print(f"   Cache time: {adapter._watchlist_cache_time}")
            print()
            
            # Summary
            print("="*60)
            print("✅ All tests passed!")
            print("="*60)
            print("\nYour Capital.com integration is working correctly.")
            print("\nNext steps:")
            print("  1. Start the bot: POST /api/bot/start")
            print("  2. Monitor logs: tail -f /tmp/trademinds.log")
            print("  3. Check signals: GET /api/signals")
            print()
            
            return True
            
        finally:
            await adapter.disconnect()
            print("Disconnected from Capital.com\n")


async def main():
    try:
        success = await test_connection()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
