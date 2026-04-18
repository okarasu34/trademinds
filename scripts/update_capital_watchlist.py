"""
Script to fetch symbols from Capital.com "TradeMinds" watchlist
and update the strategy to use those symbols.

Run this on the server:
cd /root/trademinds/backend && venv/bin/python3 ../scripts/update_capital_watchlist.py
"""
import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from sqlalchemy import select, update
from db.database import AsyncSessionLocal
from db.models import BrokerAccount, Strategy
from brokers.capital_adapter import CapitalAdapter
from loguru import logger


async def main():
    """Fetch Capital.com watchlist and update strategy symbols"""
    
    async with AsyncSessionLocal() as db:
        # Find Capital.com broker account
        result = await db.execute(
            select(BrokerAccount).where(
                BrokerAccount.broker_type.in_(["capital", "capital.com", "capitalcom"]),
                BrokerAccount.is_active == True
            ).limit(1)
        )
        account = result.scalar_one_or_none()
        
        if not account:
            logger.error("No active Capital.com account found")
            return
        
        logger.info(f"Found Capital.com account: {account.id}")
        
        # Connect to Capital.com
        adapter = CapitalAdapter(account)
        if not await adapter.connect():
            logger.error("Failed to connect to Capital.com")
            return
        
        try:
            # Get watchlists
            watchlists = await adapter.get_watchlists()
            logger.info(f"Found {len(watchlists)} watchlists")
            
            # Find "TradeMinds" watchlist
            trademinds_wl = None
            for wl in watchlists:
                if wl.get("name", "").lower() == "trademinds":
                    trademinds_wl = wl
                    break
            
            if not trademinds_wl:
                logger.error("TradeMinds watchlist not found")
                logger.info(f"Available watchlists: {[wl.get('name') for wl in watchlists]}")
                return
            
            watchlist_id = trademinds_wl.get("id")
            logger.info(f"Found TradeMinds watchlist: {watchlist_id}")
            
            # Get markets from watchlist
            epics = await adapter.get_watchlist_markets(watchlist_id)
            
            if not epics:
                logger.error("No markets found in TradeMinds watchlist")
                return
            
            logger.info(f"Found {len(epics)} markets in TradeMinds watchlist:")
            for epic in epics:
                logger.info(f"  - {epic}")
            
            # Update all active strategies to use these symbols
            result = await db.execute(
                select(Strategy).where(
                    Strategy.user_id == account.user_id,
                    Strategy.is_active == True
                )
            )
            strategies = result.scalars().all()
            
            if not strategies:
                logger.warning("No active strategies found")
                return
            
            for strategy in strategies:
                # Update symbols
                strategy.symbols = epics
                logger.info(f"Updated strategy '{strategy.name}' with {len(epics)} symbols")
            
            await db.commit()
            logger.info("✅ Successfully updated strategies with Capital.com watchlist symbols")
            
        finally:
            await adapter.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
