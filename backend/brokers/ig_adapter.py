"""
IG Markets broker adapter for forex, indices, commodities, stocks, and crypto CFDs.
Uses IG REST API.
Demo: https://demo-api.ig.com/gateway/deal
Live: https://api.ig.com/gateway/deal
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


class IGAdapter(BrokerAdapter):
    DEMO_URL = "https://demo-api.ig.com/gateway/deal"
    LIVE_URL = "https://api.ig.com/gateway/deal"

    TIMEFRAME_MAP = {
        "1m":  "MINUTE",
        "5m":  "MINUTE_5",
        "15m": "MINUTE_15",
        "30m": "MINUTE_30",
        "1h":  "HOUR",
        "4h":  "HOUR_4",
        "1d":  "DAY",
        "1w":  "WEEK",
    }

    def __init__(self, account: BrokerAccount):
        super().__init__(account)
        self.session: Optional[aiohttp.ClientSession] = None
        self.cst_token: Optional[str] = None
        self.x_security_token: Optional[str] = None
        self.account_id: Optional[str] = None
        self._cached_watchlist_symbols: list[str] = []
        self._watchlist_cache_time: Optional[datetime] = None
        # Demo veya live
        is_demo = "demo" in (account.name or "").lower()
        self.BASE_URL = self.DEMO_URL if is_demo else self.LIVE_URL

    async def connect(self) -> bool:
        try:
            api_key    = decrypt_credential(self.account.encrypted_api_key)
            password   = decrypt_credential(self.account.encrypted_api_secret)
            identifier = decrypt_credential(self.account.encrypted_extra) if self.account.encrypted_extra else ""

            self.session = aiohttp.ClientSession()

            async with self.session.post(
                f"{self.BASE_URL}/session",
                json={"identifier": identifier, "password": password},
                headers={
                    "X-IG-API-KEY": api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json; charset=UTF-8",
                    "Version": "2",
                }
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"IG login failed: {resp.status} - {text}")
                    return False

                self.cst_token = resp.headers.get("CST")
                self.x_security_token = resp.headers.get("X-SECURITY-TOKEN")

                if not self.cst_token or not self.x_security_token:
                    logger.error("IG: Missing auth tokens in response")
                    return False

                data = await resp.json()
                self.account_id = data.get("currentAccountId")
                logger.info(f"Connected to IG Markets | account={self.account_id}")

            await self.load_trademinds_watchlist()
            return True

        except Exception as e:
            logger.error(f"IG connect error: {e}")
            return False

    async def disconnect(self):
        if self.session:
            try:
                await self.session.delete(
                    f"{self.BASE_URL}/session",
                    headers=self._get_headers()
                )
            except Exception:
                pass
            await self.session.close()
            self.session = None

    def _get_headers(self, version: str = "2") -> dict:
        return {
            "X-IG-API-KEY": decrypt_credential(self.account.encrypted_api_key),
            "CST": self.cst_token or "",
            "X-SECURITY-TOKEN": self.x_security_token or "",
            "Content-Type": "application/json",
            "Accept": "application/json; charset=UTF-8",
            "Version": version,
        }

    async def get_account_info(self) -> AccountInfo:
        async with self.session.get(
            f"{self.BASE_URL}/accounts",
            headers=self._get_headers(version="1")
        ) as resp:
            data = await resp.json()
            accounts = data.get("accounts", [])
            acc = next((a for a in accounts if a.get("accountId") == self.account_id), accounts[0] if accounts else {})
            balance_info = acc.get("balance", {})

            balance = balance_info.get("balance", 0)
            equity = balance_info.get("balance", 0) + balance_info.get("profitLoss", 0)
            available = balance_info.get("available", 0)
            margin_used = balance - available if available else 0

            return AccountInfo(
                balance=balance,
                equity=equity,
                margin_used=max(margin_used, 0),
                free_margin=available,
                currency=acc.get("currency", "EUR"),
                leverage=1.0,
            )

    async def get_tick(self, symbol: str) -> TickData:
        async with self.session.get(
            f"{self.BASE_URL}/markets/{symbol}",
            headers=self._get_headers(version="3")
        ) as resp:
            data = await resp.json()
            snapshot = data.get("snapshot", {})
            bid = snapshot.get("bid", 0)
            ask = snapshot.get("offer", 0)
            return TickData(
                symbol=symbol,
                bid=bid,
                ask=ask,
                price=(bid + ask) / 2,
                spread=ask - bid,
                timestamp=datetime.utcnow().timestamp(),
            )

    async def get_candles(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> pd.DataFrame:
        tf = self.TIMEFRAME_MAP.get(timeframe, "HOUR")
        try:
            async with self.session.get(
                f"{self.BASE_URL}/prices/{symbol}",
                params={"resolution": tf, "max": str(limit), "pageSize": str(limit)},
                headers=self._get_headers(version="3")
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning(f"IG candles failed for {symbol}: {resp.status} - {text[:100]}")
                    return None
                data = await resp.json()
                prices = data.get("prices", [])
                if not prices:
                    return None

                rows = []
                for p in prices:
                    ts = p.get("snapshotTimeUTC") or p.get("snapshotTime", "")
                    o = p.get("openPrice", {})
                    h = p.get("highPrice", {})
                    l = p.get("lowPrice", {})
                    c = p.get("closePrice", {})
                    # IG mid price hesapla (bid+ask)/2
                    rows.append({
                        "timestamp": ts,
                        "open":   (o.get("bid", 0) + o.get("ask", 0)) / 2,
                        "high":   (h.get("bid", 0) + h.get("ask", 0)) / 2,
                        "low":    (l.get("bid", 0) + l.get("ask", 0)) / 2,
                        "close":  (c.get("bid", 0) + c.get("ask", 0)) / 2,
                        "volume": p.get("lastTradedVolume", 0),
                    })

                df = pd.DataFrame(rows)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                return df

        except Exception as e:
            logger.error(f"IG get_candles error for {symbol}: {e}")
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
            direction = "BUY" if side.upper() == "BUY" else "SELL"

            order_data = {
                "epic": symbol,
                "direction": direction,
                "size": lot_size,
                "orderType": "MARKET",
                "timeInForce": "FILL_OR_KILL",
                "guaranteedStop": False,
                "forceOpen": True,
                "currencyCode": self.account.currency or "EUR",
            }

            if stop_loss:
                order_data["stopLevel"] = stop_loss
            if take_profit:
                order_data["limitLevel"] = take_profit

            async with self.session.post(
                f"{self.BASE_URL}/positions/otc",
                json=order_data,
                headers=self._get_headers(version="2")
            ) as resp:
                if resp.status not in [200, 201]:
                    text = await resp.text()
                    logger.error(f"IG place_order failed: {resp.status} - {text}")
                    return None

                data = await resp.json()
                deal_reference = data.get("dealReference", "")

                # Deal confirmation al
                if deal_reference:
                    await asyncio.sleep(0.5)
                    async with self.session.get(
                        f"{self.BASE_URL}/confirms/{deal_reference}",
                        headers=self._get_headers(version="1")
                    ) as confirm_resp:
                        if confirm_resp.status == 200:
                            confirm_data = await confirm_resp.json()
                            deal_id = confirm_data.get("dealId", deal_reference)
                            status = confirm_data.get("dealStatus", "")
                            if status == "REJECTED":
                                reason = confirm_data.get("reason", "unknown")
                                logger.error(f"IG order rejected: {reason}")
                                return None
                            logger.info(f"Order placed: {symbol} {side} {lot_size} | dealId={deal_id}")
                            return deal_id

                logger.info(f"Order placed: {symbol} {side} {lot_size} | ref={deal_reference}")
                return deal_reference

        except Exception as e:
            logger.error(f"IG place_order error: {e}")
            return None

    async def close_order(self, order_id: str, symbol: str) -> bool:
        try:
            # Açık pozisyonlardan bilgi al
            async with self.session.get(
                f"{self.BASE_URL}/positions",
                headers=self._get_headers(version="2")
            ) as resp:
                if resp.status != 200:
                    logger.error(f"IG get positions failed: {resp.status}")
                    return False
                data = await resp.json()

            # deal_id ile eşleştir
            position = None
            for pos in data.get("positions", []):
                p = pos.get("position", {})
                if p.get("dealId") == order_id:
                    position = pos
                    break

            if not position:
                logger.warning(f"IG position {order_id} not found for close")
                return False

            p = position.get("position", {})
            direction = "SELL" if p.get("direction") == "BUY" else "BUY"
            size = p.get("size", 0)

            # IG delete kullanmıyor, POST ile kapatıyor
            # _method override gerekiyor
            close_headers = self._get_headers(version="1")
            close_headers["_method"] = "DELETE"

            async with self.session.post(
                f"{self.BASE_URL}/positions/otc",
                json={
                    "dealId": order_id,
                    "direction": direction,
                    "size": size,
                    "orderType": "MARKET",
                    "timeInForce": "FILL_OR_KILL",
                },
                headers=close_headers,
            ) as resp:
                if resp.status not in [200, 201]:
                    text = await resp.text()
                    logger.error(f"IG close_order failed: {resp.status} - {text}")
                    return False

                data = await resp.json()
                deal_ref = data.get("dealReference", "")

                # Confirmation bekle
                if deal_ref:
                    await asyncio.sleep(0.5)
                    async with self.session.get(
                        f"{self.BASE_URL}/confirms/{deal_ref}",
                        headers=self._get_headers(version="1")
                    ) as confirm_resp:
                        if confirm_resp.status == 200:
                            confirm_data = await confirm_resp.json()
                            if confirm_data.get("dealStatus") == "REJECTED":
                                logger.error(f"IG close rejected: {confirm_data.get('reason')}")
                                return False

                logger.info(f"Position closed: {symbol} dealId={order_id}")
                return True

        except Exception as e:
            logger.error(f"IG close_order error: {e}")
            return False

    async def get_open_orders(self) -> list[OpenOrder]:
        try:
            async with self.session.get(
                f"{self.BASE_URL}/positions",
                headers=self._get_headers(version="2")
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Get positions failed: {resp.status}")
                data = await resp.json()
                positions = data.get("positions", [])
                orders = []
                for pos in positions:
                    market = pos.get("market", {})
                    p = pos.get("position", {})
                    pnl = p.get("upl", 0) or 0

                    deal_id = p.get("dealId", "")

                    orders.append(OpenOrder(
                        order_id=deal_id,
                        symbol=market.get("epic", ""),
                        side="buy" if p.get("direction") == "BUY" else "sell",
                        lot_size=p.get("size", 0),
                        entry_price=p.get("openLevel", 0) or p.get("level", 0),
                        current_price=market.get("bid", 0),
                        stop_loss=p.get("stopLevel"),
                        take_profit=p.get("limitLevel"),
                        pnl=pnl,
                        opened_at=p.get("createdDateUTC", ""),
                    ))
                return orders
        except Exception as e:
            logger.error(f"IG get_open_orders error: {e}")
            return []

    async def is_connected(self) -> bool:
        if not self.session or self.session.closed:
            return False
        try:
            async with self.session.get(
                f"{self.BASE_URL}/session",
                headers=self._get_headers(version="1")
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def load_trademinds_watchlist(self):
        """TradeMinds watchlist'ini yükle."""
        try:
            async with self.session.get(
                f"{self.BASE_URL}/watchlists",
                headers=self._get_headers(version="1")
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                watchlists = data.get("watchlists", [])

                # TradeMinds watchlist'ini bul
                wl = next((w for w in watchlists if "trademinds" in w.get("name", "").lower()), None)
                if not wl:
                    # Yoksa ilk watchlist'i kullan
                    wl = watchlists[0] if watchlists else None
                if not wl:
                    return

                wl_id = wl.get("id")

            async with self.session.get(
                f"{self.BASE_URL}/watchlists/{wl_id}",
                headers=self._get_headers(version="1")
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                markets = data.get("markets", [])
                self._cached_watchlist_symbols = [
                    m.get("epic") for m in markets
                    if m and m.get("epic")
                ]
                self._watchlist_cache_time = datetime.utcnow()
                logger.info(
                    f"Loaded {len(self._cached_watchlist_symbols)} symbols from "
                    f"IG watchlist {wl.get('name', wl_id)}"
                )
        except Exception as e:
            logger.error(f"IG load_watchlist error: {e}")

    def get_watchlist_symbols(self) -> list[str]:
        return self._cached_watchlist_symbols

    async def get_closed_transactions(self, hours_back: int = 6) -> dict:
        """Kapanan trade'lerin PnL bilgisini çek."""
        try:
            from_date = (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S")
            to_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

            async with self.session.get(
                f"{self.BASE_URL}/history/transactions",
                params={
                    "from": from_date,
                    "to": to_date,
                    "type": "TRADE",
                    "pageSize": "50",
                },
                headers=self._get_headers(version="2")
            ) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()
                transactions = data.get("transactions", [])
                result = {}
                for txn in transactions:
                    deal_id = txn.get("reference", "")
                    pnl_str = txn.get("profitAndLoss", "0")
                    # IG format: "E12.50" veya "-E5.30"
                    pnl = float(pnl_str.replace("E", "").replace(",", ""))
                    result[deal_id] = {
                        "pnl": pnl,
                        "symbol": txn.get("instrumentName", ""),
                        "date": txn.get("dateUtc", ""),
                    }
                return result
        except Exception as e:
            logger.error(f"IG get_closed_transactions error: {e}")
            return {}

    async def get_market_rules(self, symbol: str) -> dict:
        """Market kurallarını al (min size, tick size vs)."""
        try:
            async with self.session.get(
                f"{self.BASE_URL}/markets/{symbol}",
                headers=self._get_headers(version="3")
            ) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()
                rules = data.get("dealingRules", {})
                instrument = data.get("instrument", {})
                min_size_data = rules.get("minDealSize", {})
                return {
                    "min_size": min_size_data.get("value", 0.01),
                    "max_size": rules.get("maxDealSize", {}).get("value", 100),
                    "tick_size": instrument.get("lotSize", 1),
                    "currency": instrument.get("currencies", [{}])[0].get("code", "EUR") if instrument.get("currencies") else "EUR",
                }
        except Exception as e:
            logger.error(f"IG get_market_rules error: {e}")
            return {"min_size": 0.01, "max_size": 100, "tick_size": 1}
