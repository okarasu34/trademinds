"""
MetaAPI adapter for MT4 and MT5 brokers.
Uses MetaAPI Cloud (metaapi.cloud) REST API.
Requires: pip install metaapi-cloud-sdk
"""
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from brokers.base_adapter import BrokerAdapter, AccountInfo, TickData, OpenOrder
from db.models import BrokerAccount
from core.security import decrypt_credential


METAAPI_BASE = "https://mt-client-api-v1.agiliumtrade.agiliumtrade.ai"


class MetaAPIAdapter(BrokerAdapter):
    """
    MetaAPI adapter for MT4/MT5.
    encrypted_api_key  → MetaAPI token
    encrypted_extra    → MetaAPI account ID
    """

    TIMEFRAME_MAP = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
    }

    def __init__(self, account: BrokerAccount):
        super().__init__(account)
        self.token: str = ""
        self.account_id: str = ""
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        try:
            self.token = decrypt_credential(self.account.encrypted_api_key)
            self.account_id = decrypt_credential(self.account.encrypted_extra)
            self._session = aiohttp.ClientSession(
                headers={
                    "auth-token": self.token,
                    "Content-Type": "application/json",
                }
            )
            # Verify connection
            info = await self.get_account_info()
            logger.info(f"MetaAPI connected: account {self.account_id}, balance {info.balance}")
            return True
        except Exception as e:
            logger.error(f"MetaAPI connect failed: {e}")
            return False

    async def disconnect(self):
        if self._session:
            await self._session.close()

    async def get_account_info(self) -> AccountInfo:
        url = f"{METAAPI_BASE}/users/current/accounts/{self.account_id}/account-information"
        async with self._session.get(url) as resp:
            data = await resp.json()
            return AccountInfo(
                balance=data.get("balance", 0),
                equity=data.get("equity", 0),
                margin_used=data.get("margin", 0),
                free_margin=data.get("freeMargin", 0),
                currency=data.get("currency", "USD"),
                leverage=data.get("leverage", 1),
            )

    async def get_tick(self, symbol: str) -> TickData:
        url = f"{METAAPI_BASE}/users/current/accounts/{self.account_id}/symbols/{symbol}/current-price"
        async with self._session.get(url) as resp:
            data = await resp.json()
            bid = data.get("bid", 0)
            ask = data.get("ask", 0)
            return TickData(
                symbol=symbol,
                bid=bid,
                ask=ask,
                price=(bid + ask) / 2,
                spread=ask - bid,
                timestamp=datetime.utcnow().timestamp(),
            )

    async def get_candles(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        tf = self.TIMEFRAME_MAP.get(timeframe, "1h")
        start = (datetime.utcnow() - timedelta(days=30)).isoformat() + "Z"
        url = (
            f"{METAAPI_BASE}/users/current/accounts/{self.account_id}"
            f"/historical-market-data/symbols/{symbol}/timeframes/{tf}/candles"
            f"?startTime={start}&limit={limit}"
        )
        async with self._session.get(url) as resp:
            data = await resp.json()

        rows = []
        for c in data:
            rows.append({
                "timestamp": pd.to_datetime(c["time"]),
                "open": c["open"],
                "high": c["high"],
                "low": c["low"],
                "close": c["close"],
                "volume": c.get("tickVolume", 0),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
        return df

    async def place_order(
        self, symbol: str, side: str, lot_size: float,
        stop_loss: float, take_profit: float, comment: str = "TradeMinds"
    ) -> Optional[str]:
        url = f"{METAAPI_BASE}/users/current/accounts/{self.account_id}/trade"
        payload = {
            "actionType": "ORDER_TYPE_BUY" if side == "buy" else "ORDER_TYPE_SELL",
            "symbol": symbol,
            "volume": lot_size,
            "stopLoss": stop_loss,
            "takeProfit": take_profit,
            "comment": comment,
        }
        try:
            async with self._session.post(url, json=payload) as resp:
                data = await resp.json()
                order_id = data.get("orderId") or data.get("positionId")
                if order_id:
                    logger.info(f"MT5 order placed: {symbol} {side} {lot_size} → {order_id}")
                    return str(order_id)
                logger.error(f"MT5 order failed: {data}")
                return None
        except Exception as e:
            logger.error(f"MT5 place order error: {e}")
            return None

    async def close_order(self, order_id: str, symbol: str) -> bool:
        url = f"{METAAPI_BASE}/users/current/accounts/{self.account_id}/trade"
        payload = {
            "actionType": "POSITION_CLOSE_ID",
            "positionId": order_id,
        }
        try:
            async with self._session.post(url, json=payload) as resp:
                data = await resp.json()
                return data.get("numericCode") == 10009  # TRADE_RETCODE_DONE
        except Exception as e:
            logger.error(f"MT5 close order error: {e}")
            return False

    async def get_open_orders(self) -> list[OpenOrder]:
        url = f"{METAAPI_BASE}/users/current/accounts/{self.account_id}/positions"
        try:
            async with self._session.get(url) as resp:
                positions = await resp.json()
            result = []
            for p in positions:
                result.append(OpenOrder(
                    order_id=str(p.get("id", "")),
                    symbol=p.get("symbol", ""),
                    side="buy" if p.get("type") == "POSITION_TYPE_BUY" else "sell",
                    lot_size=p.get("volume", 0),
                    entry_price=p.get("openPrice", 0),
                    current_price=p.get("currentPrice", 0),
                    stop_loss=p.get("stopLoss"),
                    take_profit=p.get("takeProfit"),
                    pnl=p.get("unrealizedProfit", 0),
                    opened_at=str(p.get("time", "")),
                ))
            return result
        except Exception as e:
            logger.error(f"MT5 get positions error: {e}")
            return []

    async def is_connected(self) -> bool:
        try:
            await self.get_account_info()
            return True
        except Exception:
            return False
