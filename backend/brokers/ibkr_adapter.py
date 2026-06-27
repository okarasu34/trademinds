"""
Interactive Brokers adapter using Client Portal REST API.

IMPORTANT: IBKR requires IB Gateway or TWS running on the server.
Client Portal Gateway must be started before using this adapter.

Setup:
  1. Download IB Gateway from https://www.interactivebrokers.com/en/trading/ibgateway-stable.php
  2. Install and configure on the server
  3. Start gateway: java -jar clientportal.gw/bin/run.sh root/conf.yaml
  4. Gateway runs on https://localhost:5000 by default

Demo:    Paper Trading account (no real money)
Live:    Live account

API Docs: https://www.interactivebrokers.com/api/doc.html
"""
import aiohttp
import ssl
import asyncio
from typing import Optional
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta

from brokers.base_adapter import BrokerAdapter, AccountInfo, TickData, OpenOrder
from core.security import decrypt_credential
from db.models import BrokerAccount


class IBKRAdapter(BrokerAdapter):
    # Client Portal Gateway default URL (localhost)
    DEFAULT_GATEWAY = "https://localhost:5000/v1/api"

    TIMEFRAME_MAP = {
        "1m":  "1min",
        "5m":  "5min",
        "15m": "15min",
        "30m": "30min",
        "1h":  "1h",
        "4h":  "4h",
        "1d":  "1d",
        "1w":  "1w",
    }

    def __init__(self, account: BrokerAccount):
        super().__init__(account)
        self.session: Optional[aiohttp.ClientSession] = None
        self.account_id: Optional[str] = None
        self._cached_watchlist_symbols: list[str] = []
        # Gateway URL — extra alanında özel URL varsa kullan
        extra = decrypt_credential(account.encrypted_extra) if account.encrypted_extra else ""
        self.BASE_URL = extra if extra.startswith("http") else self.DEFAULT_GATEWAY

    async def connect(self) -> bool:
        try:
            # IBKR gateway self-signed cert kullanıyor
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

            self.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=ssl_ctx)
            )

            # Auth durumunu kontrol et
            async with self.session.get(f"{self.BASE_URL}/iserver/auth/status") as resp:
                if resp.status != 200:
                    logger.error(f"IBKR Gateway not reachable: {resp.status}")
                    return False
                data = await resp.json()
                if not data.get("authenticated"):
                    # Re-authenticate
                    async with self.session.post(f"{self.BASE_URL}/iserver/auth/ssodh/init") as auth_resp:
                        if auth_resp.status != 200:
                            logger.error("IBKR authentication failed")
                            return False

            # Account ID al
            async with self.session.get(f"{self.BASE_URL}/iserver/accounts") as resp:
                if resp.status != 200:
                    logger.error(f"IBKR get accounts failed: {resp.status}")
                    return False
                data = await resp.json()
                accounts = data.get("accounts", [])
                if not accounts:
                    logger.error("IBKR: No accounts found")
                    return False
                self.account_id = accounts[0]
                logger.info(f"Connected to Interactive Brokers | account={self.account_id}")

            await self.load_trademinds_watchlist()
            return True

        except aiohttp.ClientConnectorError:
            logger.error("IBKR Gateway not running! Start IB Gateway first.")
            return False
        except Exception as e:
            logger.error(f"IBKR connect error: {e}")
            return False

    async def disconnect(self):
        if self.session:
            try:
                await self.session.post(f"{self.BASE_URL}/logout")
            except Exception:
                pass
            await self.session.close()
            self.session = None

    async def get_account_info(self) -> AccountInfo:
        async with self.session.get(
            f"{self.BASE_URL}/portfolio/{self.account_id}/summary"
        ) as resp:
            data = await resp.json()

            balance = data.get("totalcashvalue", {}).get("amount", 0)
            equity = data.get("netliquidation", {}).get("amount", 0)
            available = data.get("buyingpower", {}).get("amount", 0)
            margin_used = data.get("initmarginreq", {}).get("amount", 0)
            currency = data.get("netliquidation", {}).get("currency", "USD")

            return AccountInfo(
                balance=balance,
                equity=equity,
                margin_used=margin_used,
                free_margin=available,
                currency=currency,
                leverage=1.0,
            )

    async def get_tick(self, symbol: str) -> TickData:
        # IBKR conid (contract ID) gerekiyor
        conid = await self._get_conid(symbol)
        if not conid:
            raise Exception(f"Symbol not found: {symbol}")

        async with self.session.get(
            f"{self.BASE_URL}/iserver/marketdata/snapshot",
            params={"conids": str(conid), "fields": "31,84,86"}
        ) as resp:
            data = await resp.json()
            if not data:
                raise Exception(f"No tick data for {symbol}")
            tick = data[0] if isinstance(data, list) else data
            bid = float(tick.get("84", 0) or 0)
            ask = float(tick.get("86", 0) or 0)
            last = float(tick.get("31", 0) or 0)
            return TickData(
                symbol=symbol,
                bid=bid,
                ask=ask,
                price=last if last else (bid + ask) / 2,
                spread=ask - bid if ask and bid else 0,
                timestamp=datetime.utcnow().timestamp(),
            )

    async def get_candles(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> pd.DataFrame:
        conid = await self._get_conid(symbol)
        if not conid:
            return None

        period = self.TIMEFRAME_MAP.get(timeframe, "1h")

        # IBKR period string: "200" bars
        try:
            async with self.session.get(
                f"{self.BASE_URL}/iserver/marketdata/history",
                params={
                    "conid": str(conid),
                    "period": "1w",
                    "bar": period,
                    "outsideRth": "true",
                }
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning(f"IBKR candles failed for {symbol}: {resp.status} - {text[:100]}")
                    return None
                data = await resp.json()
                bars = data.get("data", [])
                if not bars:
                    return None

                rows = []
                for bar in bars:
                    rows.append({
                        "timestamp": datetime.fromtimestamp(bar.get("t", 0) / 1000),
                        "open": bar.get("o", 0),
                        "high": bar.get("h", 0),
                        "low": bar.get("l", 0),
                        "close": bar.get("c", 0),
                        "volume": bar.get("v", 0),
                    })

                df = pd.DataFrame(rows)
                df = df.sort_values("timestamp").reset_index(drop=True)
                return df.tail(limit)

        except Exception as e:
            logger.error(f"IBKR get_candles error for {symbol}: {e}")
            return None

    async def place_order(
        self,
        symbol: str,
        side: str,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
        comment: str = "TradeMinds",
    ) -> Optional[str]:
        try:
            conid = await self._get_conid(symbol)
            if not conid:
                logger.error(f"IBKR: Symbol {symbol} not found")
                return None

            # Ana emir
            order = {
                "conid": conid,
                "orderType": "MKT",
                "side": side.upper(),
                "quantity": lot_size,
                "tif": "GTC",
            }

            orders = [order]

            # Bracket order: SL ve TP ekleme
            if stop_loss:
                sl_order = {
                    "conid": conid,
                    "orderType": "STP",
                    "side": "SELL" if side.upper() == "BUY" else "BUY",
                    "quantity": lot_size,
                    "price": stop_loss,
                    "tif": "GTC",
                    "parentId": "main",
                }
                orders.append(sl_order)

            if take_profit:
                tp_order = {
                    "conid": conid,
                    "orderType": "LMT",
                    "side": "SELL" if side.upper() == "BUY" else "BUY",
                    "quantity": lot_size,
                    "price": take_profit,
                    "tif": "GTC",
                    "parentId": "main",
                }
                orders.append(tp_order)

            async with self.session.post(
                f"{self.BASE_URL}/iserver/account/{self.account_id}/orders",
                json={"orders": orders}
            ) as resp:
                if resp.status not in [200, 201]:
                    text = await resp.text()
                    logger.error(f"IBKR place_order failed: {resp.status} - {text}")
                    return None

                data = await resp.json()

                # IBKR bazen confirmation istiyor
                if isinstance(data, list) and data and data[0].get("id"):
                    # Confirm order
                    reply_id = data[0]["id"]
                    async with self.session.post(
                        f"{self.BASE_URL}/iserver/reply/{reply_id}",
                        json={"confirmed": True}
                    ) as confirm_resp:
                        confirm_data = await confirm_resp.json()
                        if isinstance(confirm_data, list) and confirm_data:
                            order_id = str(confirm_data[0].get("order_id", ""))
                            logger.info(f"Order placed: {symbol} {side} {lot_size} | orderId={order_id}")
                            return order_id

                # Normal response
                if isinstance(data, list) and data:
                    order_id = str(data[0].get("order_id", ""))
                    logger.info(f"Order placed: {symbol} {side} {lot_size} | orderId={order_id}")
                    return order_id

                return None

        except Exception as e:
            logger.error(f"IBKR place_order error: {e}")
            return None

    async def close_order(self, order_id: str, symbol: str) -> bool:
        try:
            # Açık pozisyonları bul
            async with self.session.get(
                f"{self.BASE_URL}/portfolio/{self.account_id}/positions"
            ) as resp:
                if resp.status != 200:
                    return False
                positions = await resp.json()

            # Pozisyonu bul
            position = None
            for pos in positions:
                conid = str(pos.get("conid", ""))
                if conid == order_id or pos.get("contractDesc", "") == symbol:
                    position = pos
                    break

            if not position:
                logger.warning(f"IBKR position {order_id} not found")
                return False

            conid = position.get("conid")
            qty = abs(position.get("position", 0))
            side = "SELL" if position.get("position", 0) > 0 else "BUY"

            # Kapatma emri
            async with self.session.post(
                f"{self.BASE_URL}/iserver/account/{self.account_id}/orders",
                json={"orders": [{
                    "conid": conid,
                    "orderType": "MKT",
                    "side": side,
                    "quantity": qty,
                    "tif": "GTC",
                }]}
            ) as resp:
                if resp.status not in [200, 201]:
                    text = await resp.text()
                    logger.error(f"IBKR close_order failed: {resp.status} - {text}")
                    return False

                data = await resp.json()
                # Confirmation
                if isinstance(data, list) and data and data[0].get("id"):
                    async with self.session.post(
                        f"{self.BASE_URL}/iserver/reply/{data[0]['id']}",
                        json={"confirmed": True}
                    ) as confirm_resp:
                        pass

                logger.info(f"Position closed: {symbol}")
                return True

        except Exception as e:
            logger.error(f"IBKR close_order error: {e}")
            return False

    async def get_open_orders(self) -> list[OpenOrder]:
        try:
            async with self.session.get(
                f"{self.BASE_URL}/portfolio/{self.account_id}/positions"
            ) as resp:
                if resp.status != 200:
                    return []
                positions = await resp.json()

                orders = []
                for pos in positions:
                    if pos.get("position", 0) == 0:
                        continue
                    orders.append(OpenOrder(
                        order_id=str(pos.get("conid", "")),
                        symbol=pos.get("contractDesc", ""),
                        side="buy" if pos.get("position", 0) > 0 else "sell",
                        lot_size=abs(pos.get("position", 0)),
                        entry_price=pos.get("avgCost", 0),
                        current_price=pos.get("mktPrice", 0),
                        stop_loss=None,
                        take_profit=None,
                        pnl=pos.get("unrealizedPnl", 0),
                        opened_at="",
                    ))
                return orders
        except Exception as e:
            logger.error(f"IBKR get_open_orders error: {e}")
            return []

    async def is_connected(self) -> bool:
        if not self.session or self.session.closed:
            return False
        try:
            async with self.session.get(f"{self.BASE_URL}/iserver/auth/status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("authenticated", False)
                return False
        except Exception:
            return False

    async def load_trademinds_watchlist(self):
        """IBKR watchlist'ini yükle."""
        try:
            async with self.session.get(
                f"{self.BASE_URL}/iserver/watchlists"
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                watchlists = data.get("data", []) if isinstance(data, dict) else data

                wl = next((w for w in watchlists if "trademinds" in str(w.get("name", "")).lower()), None)
                if not wl:
                    wl = watchlists[0] if watchlists else None
                if not wl:
                    return

                wl_id = wl.get("id")

            async with self.session.get(
                f"{self.BASE_URL}/iserver/watchlists/{wl_id}"
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                instruments = data.get("instruments", []) if isinstance(data, dict) else []
                self._cached_watchlist_symbols = [
                    str(i.get("conid", "")) for i in instruments if i
                ]
                logger.info(f"Loaded {len(self._cached_watchlist_symbols)} symbols from IBKR watchlist")
        except Exception as e:
            logger.error(f"IBKR load_watchlist error: {e}")

    def get_watchlist_symbols(self) -> list[str]:
        return self._cached_watchlist_symbols

    async def get_closed_transactions(self, hours_back: int = 6) -> dict:
        """Kapanan trade'lerin PnL bilgisini çek."""
        try:
            async with self.session.get(
                f"{self.BASE_URL}/iserver/account/trades"
            ) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()
                result = {}
                for trade in data:
                    exec_id = trade.get("execution_id", "")
                    pnl = trade.get("realized_pnl", 0)
                    result[exec_id] = {
                        "pnl": pnl,
                        "symbol": trade.get("symbol", ""),
                        "date": trade.get("trade_time", ""),
                    }
                return result
        except Exception as e:
            logger.error(f"IBKR get_closed_transactions error: {e}")
            return {}

    async def _get_conid(self, symbol: str) -> Optional[int]:
        """Sembol isminden IBKR contract ID al."""
        try:
            async with self.session.get(
                f"{self.BASE_URL}/iserver/secdef/search",
                params={"symbol": symbol}
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if data and isinstance(data, list) and data[0].get("conid"):
                    return data[0]["conid"]
                return None
        except Exception:
            return None
