"""
Data Fetcher
Pulls historical OHLCV candles from brokers and stores
them in the market_candles table for backtesting.
"""
import asyncio
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd

from db.database import AsyncSessionLocal
from db.models import MarketCandle, BrokerAccount
from brokers.base_adapter import get_broker_adapter
from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert


# Default symbols to keep updated for backtesting
WATCHLIST = {
    "forex": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURGBP"],
    "crypto": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
    "commodity": ["XAUUSD", "XAGUSD", "USOIL"],
    "index": ["US500", "NAS100", "US30"],
}

TIMEFRAMES = ["1h", "4h", "1d"]


async def fetch_and_store_candles(
    symbol: str,
    timeframe: str,
    adapter,
    limit: int = 500,
) -> int:
    """
    Fetch candles from broker and upsert into DB.
    Returns number of new candles stored.
    """
    try:
        df = await adapter.get_candles(symbol, timeframe, limit=limit)
        if df is None or df.empty:
            return 0

        records = []
        for ts, row in df.iterrows():
            records.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "open_time": ts if isinstance(ts, datetime) else pd.Timestamp(ts).to_pydatetime(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
            })

        async with AsyncSessionLocal() as db:
            stmt = pg_insert(MarketCandle).values(records)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["symbol", "timeframe", "open_time"]
            )
            result = await db.execute(stmt)
            await db.commit()
            return result.rowcount

    except Exception as e:
        logger.warning(f"Candle fetch failed {symbol}/{timeframe}: {e}")
        return 0


async def ingest_historical_data(user_id: str, days_back: int = 365):
    """
    Full historical data ingestion for all watchlist symbols.
    Run once on first setup, then incremental updates via scheduler.
    """
    logger.info(f"Starting historical data ingestion ({days_back} days back)")

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(BrokerAccount).where(
                BrokerAccount.user_id == user_id,
                BrokerAccount.is_active == True,
            )
        )
        brokers = result.scalars().all()

    if not brokers:
        logger.warning("No active brokers found for data ingestion")
        return

    total_new = 0

    for broker in brokers:
        market_type = broker.market_type.value
        symbols = WATCHLIST.get(market_type, [])

        try:
            adapter = get_broker_adapter(broker)
            connected = await adapter.connect()
            if not connected:
                continue

            for symbol in symbols:
                for timeframe in TIMEFRAMES:
                    count = await fetch_and_store_candles(symbol, timeframe, adapter, limit=days_back * 24)
                    total_new += count
                    logger.debug(f"  {symbol}/{timeframe}: +{count} candles")
                    await asyncio.sleep(0.5)  # rate limit

            await adapter.disconnect()

        except Exception as e:
            logger.error(f"Ingestion error for broker {broker.name}: {e}")

    logger.info(f"Historical data ingestion complete. {total_new} new candles stored.")


async def incremental_update(user_id: str):
    """
    Update candles with latest data (run every 15 min via scheduler).
    Only fetches the last 100 candles per symbol to stay fast.
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(BrokerAccount).where(
                BrokerAccount.user_id == user_id,
                BrokerAccount.is_active == True,
            )
        )
        brokers = result.scalars().all()

    for broker in brokers:
        market_type = broker.market_type.value
        symbols = WATCHLIST.get(market_type, [])

        try:
            adapter = get_broker_adapter(broker)
            await adapter.connect()
            for symbol in symbols:
                for timeframe in ["1h", "4h"]:
                    await fetch_and_store_candles(symbol, timeframe, adapter, limit=100)
                    await asyncio.sleep(0.2)
            await adapter.disconnect()
        except Exception as e:
            logger.warning(f"Incremental update error: {e}")


async def get_candle_coverage(symbol: str, timeframe: str) -> dict:
    """Check how much historical data we have for a symbol."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(
                MarketCandle.open_time
            ).where(
                MarketCandle.symbol == symbol,
                MarketCandle.timeframe == timeframe,
            ).order_by(MarketCandle.open_time)
        )
        rows = result.scalars().all()

    if not rows:
        return {"count": 0, "from": None, "to": None}

    return {
        "count": len(rows),
        "from": rows[0].isoformat() if rows else None,
        "to": rows[-1].isoformat() if rows else None,
    }
