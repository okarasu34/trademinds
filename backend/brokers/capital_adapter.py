"""
Capital.com broker adapter for forex, indices, commodities, stocks, and crypto CFDs.
Uses Capital.com REST API v1.
"""
import aiohttp
import asyncio
from typing import Optional
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta

from brokers.base_adapter import BrokerAdapter, AccountInfo, TickData, OpenOrder
from core.security import decrypt_credential
from db.models import BrokerAccount


class CapitalAdapter(BrokerAdapter):
    """Capital.com CFD broker adapter"""

    BASE_URL = "https://api-capital.backend-capital.com/api/v1"
    
    TIMEFRAME_MAP = {
        "1m": "MINUTE",
        "5m": "MINUTE_5",
        "15m": "MINUTE_15",
        "30m": "MINUTE_30",
        "1h": "HOUR",
        "4h": "HOUR_4",
        "1d": "DAY",
        "1w": "WEEK",
    }

    def __init__(self, account: BrokerAccount):
        super().__init__(account)
        self.session: Optional[aiohttp.ClientSession] = None
        self.cst_token: Optional[str] = None
        self.x_security_token: Optional[str] = None
        self.account_id: Optional[str] = None
        self._cached_watchlist_symbols: list[str] = []
        self._watchlist_cache_time: Optional[datetime] = None

    async def connect(self) -> bool:
        """Login to Capital.com and get session tokens
        
        BrokerAccount fields:
          encrypted_api_key    → Capital.com API Key (vajc0aJ...)
          encrypted_api_secret → Account password
          encrypted_extra      → Account identifier (email)
        """
        try:
            cap_api_key  = decrypt_credential(self.account.encrypted_api_key)    # API key
            password     = decrypt_credential(self.account.encrypted_api_secret) # password
            identifier   = decrypt_credential(self.account.encrypted_extra) if self.account.encrypted_extra else cap_api_key
            
            self.session = aiohttp.ClientSession()
            
            # Login
            async with self.session.post(
                f"{self.BASE_URL}/session",
                json={"identifier": identifier, "password": password},
                headers={"X-CAP-API-KEY": cap_api_key}
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Capital.com login failed: {resp.status} - {text}")
                    return False
                
                self.cst_token = resp.headers.get("CST")
                self.x_security_token = resp.headers.get("X-SECURITY-TOKEN")
                
                if not self.cst_token or not self.x_security_token:
                    logger.error("Capital.com: Missing auth tokens in response")
                    return False
                
                data = await resp.json()
                self.account_id = data.get("accountId")
                
                logger.info(f"Connected to Capital.com | account={self.account_id}")
                
                # Load TradeMinds watchlist in background
                asyncio.create_task(self.load_trademinds_watchlist())
                
                return True
                
        except Exception as e:
            logger.error(f"Capital.com connect error: {e}")
            return False

    async def disconnect(self):
        """Logout and close session"""
        if self.session:
            try:
                if self.cst_token and self.x_security_token:
                    await self.session.delete(
                        f"{self.BASE_URL}/session",
                        headers=self._get_headers()
                    )
            except Exception as e:
                logger.error(f"Capital.com disconnect error: {e}")
            finally:
                await self.session.close()
                self.session = None

    def _get_headers(self) -> dict:
        """Get authenticated request headers"""
        return {
            "CST": self.cst_token,
            "X-SECURITY-TOKEN": self.x_security_token,
        }

    async def get_account_info(self) -> AccountInfo:
        """Get account balance and equity"""
        try:
            async with self.session.get(
                f"{self.BASE_URL}/accounts",
                headers=self._get_headers()
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Account info failed: {resp.status}")
                
                data = await resp.json()
                accounts = data.get("accounts", [])
                
                if not accounts:
                    raise Exception("No accounts found")
                
                # Use first account or find by account_id
                account = accounts[0]
                balance = account.get("balance", {})
                
                return AccountInfo(
                    balance=balance.get("balance", 0),
                    equity=balance.get("balance", 0) + balance.get("profitLoss", 0),
                    margin_used=balance.get("deposit", 0),
                    free_margin=balance.get("available", 0),
                    currency=balance.get("currency", "EUR"),
                )
                
        except Exception as e:
            logger.error(f"Capital.com get_account_info error: {e}")
            return AccountInfo(balance=0, equity=0, margin_used=0, free_margin=0, currency="EUR")

    async def get_tick(self, symbol: str) -> TickData:
        """Get current bid/ask prices"""
        try:
            # Capital.com uses epic codes, not standard symbols
            # We need to query market details
            async with self.session.get(
                f"{self.BASE_URL}/markets/{symbol}",
                headers=self._get_headers()
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Market data failed: {resp.status}")
                
                data = await resp.json()
                snapshot = data.get("snapshot", {})
                
                bid = float(snapshot.get("bid", 0))
                ask = float(snapshot.get("offer", 0))
                price = (bid + ask) / 2
                
                return TickData(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    price=price,
                    spread=ask - bid,
                    timestamp=datetime.utcnow().timestamp(),
                )
                
        except Exception as e:
            logger.error(f"Capital.com get_tick error for {symbol}: {e}")
            raise

    async def get_candles(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Get historical OHLC data"""
        try:
            resolution = self.TIMEFRAME_MAP.get(timeframe, "HOUR")
            
            # Calculate time range
            now = datetime.utcnow()
            if timeframe == "1m":
                start = now - timedelta(minutes=limit)
            elif timeframe == "5m":
                start = now - timedelta(minutes=limit * 5)
            elif timeframe == "15m":
                start = now - timedelta(minutes=limit * 15)
            elif timeframe == "30m":
                start = now - timedelta(minutes=limit * 30)
            elif timeframe == "1h":
                start = now - timedelta(hours=limit)
            elif timeframe == "4h":
                start = now - timedelta(hours=limit * 4)
            elif timeframe == "1d":
                start = now - timedelta(days=limit)
            elif timeframe == "1w":
                start = now - timedelta(weeks=limit)
            else:
                start = now - timedelta(hours=limit)
            
            # Format: YYYY-MM-DDTHH:MM:SS
            from_time = start.strftime("%Y-%m-%dT%H:%M:%S")
            to_time = now.strftime("%Y-%m-%dT%H:%M:%S")
            
            async with self.session.get(
                f"{self.BASE_URL}/prices/{symbol}",
                params={
                    "resolution": resolution,
                    "from": from_time,
                    "to": to_time,
                    "max": limit,
                },
                headers=self._get_headers()
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"Candles failed: {resp.status} - {text}")
                
                data = await resp.json()
                prices = data.get("prices", [])
                
                if not prices:
                    logger.warning(f"No candle data for {symbol} {timeframe}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame(prices)
                
                # Rename columns to standard format
                df = df.rename(columns={
                    "snapshotTime": "timestamp",
                    "openPrice": "open",
                    "highPrice": "high",
                    "lowPrice": "low",
                    "closePrice": "close",
                    "lastTradedVolume": "volume",
                })
                
                # Convert timestamp
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                
                # Select only OHLCV columns
                df = df[["open", "high", "low", "close", "volume"]]
                
                return df
                
        except Exception as e:
            logger.error(f"Capital.com get_candles error for {symbol}: {e}")
            return pd.DataFrame()

    async def place_order(
        self,
        symbol: str,
        side: str,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
        comment: str = "TradeMinds",
    ) -> Optional[str]:
        """Place a market order with SL/TP"""
        try:
            direction = "BUY" if side == "buy" else "SELL"
            
            order_payload = {
                "epic": symbol,
                "direction": direction,
                "size": lot_size,
                "guaranteedStop": False,
                "stopLevel": stop_loss,
                "profitLevel": take_profit,
            }
            
            async with self.session.post(
                f"{self.BASE_URL}/positions",
                json=order_payload,
                headers=self._get_headers()
            ) as resp:
                if resp.status not in [200, 201]:
                    text = await resp.text()
                    logger.error(f"Capital.com place_order failed: {resp.status} - {text}")
                    return None
                
                data = await resp.json()
                deal_reference = data.get("dealReference")
                
                logger.info(f"Order placed on Capital.com: {symbol} {side} {lot_size} | ref={deal_reference}")
                return deal_reference
                
        except Exception as e:
            logger.error(f"Capital.com place_order error: {e}")
            return None

    async def close_order(self, order_id: str, symbol: str) -> bool:
        """Close an open position"""
        try:
            # Get position details first
            positions = await self.get_open_orders()
            position = next((p for p in positions if p.order_id == order_id), None)
            
            if not position:
                logger.warning(f"Position {order_id} not found")
                return False
            
            # Close position (reverse direction)
            direction = "SELL" if position.side == "buy" else "BUY"
            
            async with self.session.delete(
                f"{self.BASE_URL}/positions",
                json={
                    "dealId": order_id,
                    "direction": direction,
                    "size": position.lot_size,
                },
                headers=self._get_headers()
            ) as resp:
                if resp.status not in [200, 201]:
                    text = await resp.text()
                    logger.error(f"Capital.com close_order failed: {resp.status} - {text}")
                    return False
                
                logger.info(f"Position closed on Capital.com: {order_id}")
                return True
                
        except Exception as e:
            logger.error(f"Capital.com close_order error: {e}")
            return False

    async def get_open_orders(self) -> list[OpenOrder]:
        """Get all open positions"""
        try:
            async with self.session.get(
                f"{self.BASE_URL}/positions",
                headers=self._get_headers()
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Get positions failed: {resp.status}")
                
                data = await resp.json()
                positions = data.get("positions", [])
                
                orders = []
                for pos in positions:
                    market = pos.get("market", {})
                    position_data = pos.get("position", {})
                    
                    orders.append(OpenOrder(
                        order_id=position_data.get("dealId", ""),
                        symbol=market.get("epic", ""),
                        side="buy" if position_data.get("direction") == "BUY" else "sell",
                        lot_size=position_data.get("size", 0),
                        entry_price=position_data.get("level", 0),
                        current_price=market.get("bid", 0),
                        stop_loss=position_data.get("stopLevel"),
                        take_profit=position_data.get("profitLevel"),
                        pnl=position_data.get("profit", 0),
                        opened_at=position_data.get("createdDate", ""),
                    ))
                
                return orders
                
        except Exception as e:
            logger.error(f"Capital.com get_open_orders error: {e}")
            return []

    async def is_connected(self) -> bool:
        """Check if session is still valid"""
        if not self.session or not self.cst_token:
            return False
        
        try:
            async with self.session.get(
                f"{self.BASE_URL}/accounts",
                headers=self._get_headers()
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def get_watchlists(self) -> list[dict]:
        """Get all watchlists"""
        try:
            async with self.session.get(
                f"{self.BASE_URL}/watchlists",
                headers=self._get_headers()
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Capital.com watchlists error: {resp.status} - {text}")
                    return []
                
                data = await resp.json()
                return data.get("watchlists", [])
                
        except Exception as e:
            logger.error(f"Capital.com watchlists error: {e}")
            return []

    async def get_watchlist_markets(self, watchlist_id: str) -> list[str]:
        """Get all market epics from a watchlist"""
        try:
            async with self.session.get(
                f"{self.BASE_URL}/watchlists/{watchlist_id}",
                headers=self._get_headers()
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Capital.com watchlist markets error: {resp.status} - {text}")
                    return []
                
                data = await resp.json()
                markets = data.get("markets", [])
                
                # Extract epic codes
                epics = [m.get("epic") for m in markets if m.get("epic")]
                logger.info(f"Found {len(epics)} markets in watchlist {watchlist_id}")
                
                return epics
                
        except Exception as e:
            logger.error(f"Capital.com watchlist markets error: {e}")
            return []

    async def load_trademinds_watchlist(self):
        """Load symbols from TradeMinds watchlist and cache them"""
        try:
            watchlists = await self.get_watchlists()
            
            # Find TradeMinds watchlist
            trademinds_wl = None
            for wl in watchlists:
                if wl.get("name", "").lower() == "trademinds":
                    trademinds_wl = wl
                    break
            
            if not trademinds_wl:
                logger.warning("TradeMinds watchlist not found on Capital.com")
                return
            
            watchlist_id = trademinds_wl.get("id")
            epics = await self.get_watchlist_markets(watchlist_id)
            
            if epics:
                self._cached_watchlist_symbols = epics
                self._watchlist_cache_time = datetime.utcnow()
                logger.info(f"Loaded {len(epics)} symbols from TradeMinds watchlist")
            
        except Exception as e:
            logger.error(f"Failed to load TradeMinds watchlist: {e}")

    def get_cached_watchlist_symbols(self) -> list[str]:
        """Get cached watchlist symbols (refresh if older than 1 hour)"""
        # Refresh cache if empty or older than 1 hour
        if not self._cached_watchlist_symbols or \
           not self._watchlist_cache_time or \
           (datetime.utcnow() - self._watchlist_cache_time).total_seconds() > 3600:
            # Trigger async refresh in background
            asyncio.create_task(self.load_trademinds_watchlist())
        
        return self._cached_watchlist_symbols

