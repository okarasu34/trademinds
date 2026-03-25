"""
Interactive Brokers adapter via IBKR Web API (Client Portal API).
No TWS or IB Gateway needed — uses the browser-based REST API.
"""
import aiohttp
import pandas as pd
from datetime import datetime
from typing import Optional
from loguru import logger

from brokers.base_adapter import BrokerAdapter, AccountInfo, TickData, OpenOrder
from db.models import BrokerAccount
from core.security import decrypt_credential


IBKR_BASE = "https://localhost:5000/v1/api"  # Client Portal Gateway


class IBKRAdapter(BrokerAdapter):
    """
    Interactive Brokers via Client Portal API.
    encrypted_api_key  → IBKR username
    encrypted_api_secret → IBKR password
    encrypted_extra    → Account ID (e.g. U1234567)
    """

    TIMEFRAME_MAP = {
        "1m": "1min", "5m": "5mins", "15m": "15mins", "30m": "30mins",
        "1h": "1h", "4h": "4h", "1d": "1d",
    }

    def __init__(self, account: BrokerAccount):
        super().__init__(account)
        self.account_id: str = ""
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        try:
            self.account_id = decrypt_credential(self.account.encrypted_extra)
            # IBKR Client Portal requires SSL — use ssl=False for localhost
            connector = aiohttp.TCPConnector(ssl=False)
            self._session = aiohttp.ClientSession(connector=connector)

            # Auth tickle to keep session alive
            async with self._session.post(f"{IBKR_BASE}/tickle") as resp:
                if resp.status == 200:
                    logger.info(f"IBKR connected: account {self.account_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"IBKR connect failed: {e}")
            return False

    async def disconnect(self):
        if self._session:
            await self._session.post(f"{IBKR_BASE}/logout")
            await self._session.close()

    async def get_account_info(self) -> AccountInfo:
        async with self._session.get(
            f"{IBKR_BASE}/portfolio/{self.account_id}/summary"
        ) as resp:
            data = await resp.json()

        return AccountInfo(
            balance=float(data.get("totalcashvalue", {}).get("amount", 0)),
            equity=float(data.get("netliquidation", {}).get("amount", 0)),
            margin_used=float(data.get("initmarginreq", {}).get("amount", 0)),
            free_margin=float(data.get("availablefunds", {}).get("amount", 0)),
            currency=data.get("totalcashvalue", {}).get("currency", "USD"),
        )

    async def get_tick(self, symbol: str) -> TickData:
        conid = await self._get_conid(symbol)
        if not conid:
            raise ValueError(f"Cannot find conid for {symbol}")

        async with self._session.get(
            f"{IBKR_BASE}/iserver/marketdata/snapshot",
            params={"conids": conid, "fields": "31,84,86"}
        ) as resp:
            data = await resp.json()

        item = data[0] if data else {}
        price = float(item.get("31", 0))
        bid = float(item.get("84", price))
        ask = float(item.get("86", price))

        return TickData(
            symbol=symbol,
            bid=bid,
            ask=ask,
            price=price,
            spread=ask - bid,
            timestamp=datetime.utcnow().timestamp(),
        )

    async def get_candles(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        conid = await self._get_conid(symbol)
        if not conid:
            return pd.DataFrame()

        tf = self.TIMEFRAME_MAP.get(timeframe, "1h")
        period = f"{max(1, limit // 24)}d"

        async with self._session.get(
            f"{IBKR_BASE}/iserver/marketdata/history",
            params={"conid": conid, "period": period, "bar": tf}
        ) as resp:
            data = await resp.json()

        rows = []
        for bar in data.get("data", []):
            rows.append({
                "timestamp": pd.to_datetime(bar["t"], unit="ms"),
                "open": bar["o"],
                "high": bar["h"],
                "low": bar["l"],
                "close": bar["c"],
                "volume": bar.get("v", 0),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
        return df

    async def place_order(
        self, symbol: str, side: str, lot_size: float,
        stop_loss: float, take_profit: float, comment: str = "TradeMinds"
    ) -> Optional[str]:
        conid = await self._get_conid(symbol)
        if not conid:
            return None

        orders = [
            {
                "conid": conid,
                "orderType": "MKT",
                "side": "BUY" if side == "buy" else "SELL",
                "quantity": lot_size,
                "tif": "GTC",
                "auxPrice": stop_loss,
                "lmtPrice": take_profit,
                "outsideRTH": True,
                "cOID": comment,
            }
        ]

        try:
            async with self._session.post(
                f"{IBKR_BASE}/iserver/account/{self.account_id}/orders",
                json={"orders": orders}
            ) as resp:
                data = await resp.json()
                order_id = data[0].get("order_id") if data else None
                if order_id:
                    logger.info(f"IBKR order placed: {symbol} {side} → {order_id}")
                return str(order_id) if order_id else None
        except Exception as e:
            logger.error(f"IBKR place order error: {e}")
            return None

    async def close_order(self, order_id: str, symbol: str) -> bool:
        try:
            async with self._session.delete(
                f"{IBKR_BASE}/iserver/account/{self.account_id}/order/{order_id}"
            ) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"IBKR close order error: {e}")
            return False

    async def get_open_orders(self) -> list[OpenOrder]:
        try:
            async with self._session.get(
                f"{IBKR_BASE}/portfolio/{self.account_id}/positions/0"
            ) as resp:
                positions = await resp.json()

            result = []
            for p in positions or []:
                result.append(OpenOrder(
                    order_id=str(p.get("conid", "")),
                    symbol=p.get("contractDesc", ""),
                    side="buy" if p.get("position", 0) > 0 else "sell",
                    lot_size=abs(p.get("position", 0)),
                    entry_price=p.get("avgCost", 0),
                    current_price=p.get("mktPrice", 0),
                    stop_loss=None,
                    take_profit=None,
                    pnl=p.get("unrealizedPnl", 0),
                    opened_at="",
                ))
            return result
        except Exception as e:
            logger.error(f"IBKR get positions error: {e}")
            return []

    async def is_connected(self) -> bool:
        try:
            async with self._session.get(f"{IBKR_BASE}/tickle") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _get_conid(self, symbol: str) -> Optional[str]:
        """Resolve symbol to IBKR contract ID."""
        try:
            async with self._session.get(
                f"{IBKR_BASE}/iserver/secdef/search",
                params={"symbol": symbol.replace("/", ""), "name": False}
            ) as resp:
                data = await resp.json()
            if data:
                return str(data[0].get("conid"))
        except Exception:
            pass
        return None
