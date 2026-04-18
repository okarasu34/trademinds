from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import ccxt.async_support as ccxt
from loguru import logger
import pandas as pd
from core.security import decrypt_credential
from db.models import BrokerAccount


@dataclass
class AccountInfo:
    balance: float
    equity: float
    margin_used: float
    free_margin: float
    currency: str
    leverage: float = 1.0


@dataclass
class TickData:
    symbol: str
    bid: float
    ask: float
    price: float
    spread: float
    timestamp: float


@dataclass
class OpenOrder:
    order_id: str
    symbol: str
    side: str
    lot_size: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    pnl: float
    opened_at: str


# ─────────────────────────── BASE ADAPTER ───────────────────────────

class BrokerAdapter(ABC):

    def __init__(self, account: BrokerAccount):
        self.account = account

    @abstractmethod
    async def connect(self) -> bool:
        pass

    @abstractmethod
    async def disconnect(self):
        pass

    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        pass

    @abstractmethod
    async def get_tick(self, symbol: str) -> TickData:
        pass

    @abstractmethod
    async def get_candles(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
        comment: str = "TradeMinds",
    ) -> Optional[str]:  # returns order_id
        pass

    @abstractmethod
    async def close_order(self, order_id: str, symbol: str) -> bool:
        pass

    @abstractmethod
    async def get_open_orders(self) -> list[OpenOrder]:
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        pass


# ─────────────────────────── CCXT CRYPTO ADAPTER ───────────────────────────

class CCXTAdapter(BrokerAdapter):
    """Universal crypto exchange adapter via CCXT (Binance, Bybit, OKX etc.)"""

    TIMEFRAME_MAP = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"
    }

    def __init__(self, account: BrokerAccount):
        super().__init__(account)
        self.exchange: Optional[ccxt.Exchange] = None

    async def connect(self) -> bool:
        try:
            api_key = decrypt_credential(self.account.encrypted_api_key)
            api_secret = decrypt_credential(self.account.encrypted_api_secret)

            exchange_class = getattr(ccxt, self.account.broker_type, None)
            if not exchange_class:
                logger.error(f"Unknown exchange: {self.account.broker_type}")
                return False

            self.exchange = exchange_class({
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })
            await self.exchange.load_markets()
            logger.info(f"Connected to {self.account.broker_type}")
            return True
        except Exception as e:
            logger.error(f"CCXT connect error: {e}")
            return False

    async def disconnect(self):
        if self.exchange:
            await self.exchange.close()

    async def get_account_info(self) -> AccountInfo:
        balance = await self.exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        return AccountInfo(
            balance=usdt.get("free", 0) + usdt.get("used", 0),
            equity=usdt.get("total", 0),
            margin_used=usdt.get("used", 0),
            free_margin=usdt.get("free", 0),
            currency="USDT",
        )

    async def get_tick(self, symbol: str) -> TickData:
        ticker = await self.exchange.fetch_ticker(symbol)
        bid = ticker.get("bid", ticker["last"])
        ask = ticker.get("ask", ticker["last"])
        return TickData(
            symbol=symbol,
            bid=bid,
            ask=ask,
            price=ticker["last"],
            spread=ask - bid,
            timestamp=ticker["timestamp"] / 1000,
        )

    async def get_candles(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        tf = self.TIMEFRAME_MAP.get(timeframe, "1h")
        ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    async def place_order(
        self, symbol: str, side: str, lot_size: float,
        stop_loss: float, take_profit: float, comment: str = "TradeMinds"
    ) -> Optional[str]:
        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=lot_size,
                params={
                    "stopLoss": {"type": "market", "price": stop_loss},
                    "takeProfit": {"type": "market", "price": take_profit},
                    "clientOrderId": comment,
                }
            )
            return str(order["id"])
        except Exception as e:
            logger.error(f"Place order error on {symbol}: {e}")
            return None

    async def close_order(self, order_id: str, symbol: str) -> bool:
        try:
            positions = await self.exchange.fetch_positions([symbol])
            for pos in positions:
                if pos["contracts"] > 0:
                    side = "sell" if pos["side"] == "long" else "buy"
                    await self.exchange.create_order(
                        symbol=symbol, type="market",
                        side=side, amount=pos["contracts"],
                        params={"reduceOnly": True}
                    )
            return True
        except Exception as e:
            logger.error(f"Close order error: {e}")
            return False

    async def get_open_orders(self) -> list[OpenOrder]:
        try:
            positions = await self.exchange.fetch_positions()
            orders = []
            for pos in positions:
                if pos.get("contracts", 0) > 0:
                    orders.append(OpenOrder(
                        order_id=str(pos.get("id", "")),
                        symbol=pos["symbol"],
                        side="buy" if pos["side"] == "long" else "sell",
                        lot_size=pos["contracts"],
                        entry_price=pos.get("entryPrice", 0),
                        current_price=pos.get("markPrice", 0),
                        stop_loss=pos.get("stopLossPrice"),
                        take_profit=pos.get("takeProfitPrice"),
                        pnl=pos.get("unrealizedPnl", 0),
                        opened_at=str(pos.get("datetime", "")),
                    ))
            return orders
        except Exception as e:
            logger.error(f"Get open orders error: {e}")
            return []

    async def is_connected(self) -> bool:
        try:
            await self.exchange.fetch_status()
            return True
        except Exception:
            return False


# ─────────────────────────── ADAPTER FACTORY ───────────────────────────

def get_broker_adapter(account: BrokerAccount) -> BrokerAdapter:
    """Return the right adapter based on broker_type."""
    broker_type = account.broker_type.lower()

    # Crypto exchanges via CCXT
    ccxt_exchanges = [
        "binance", "bybit", "okx", "kraken", "coinbase",
        "kucoin", "gate", "bitget", "huobi"
    ]
    if broker_type in ccxt_exchanges:
        return CCXTAdapter(account)

    # MetaTrader (MT4/MT5) via MetaAPI
    if broker_type in ["mt4", "mt5", "metaapi"]:
        from brokers.metaapi_adapter import MetaAPIAdapter
        return MetaAPIAdapter(account)

    # Interactive Brokers
    if broker_type in ["ibkr", "interactive_brokers"]:
        from brokers.ibkr_adapter import IBKRAdapter
        return IBKRAdapter(account)

    # Capital.com CFD broker
    if broker_type in ["capital", "capital.com", "capitalcom"]:
        from brokers.capital_adapter import CapitalAdapter
        return CapitalAdapter(account)

    raise ValueError(f"Unsupported broker type: {broker_type}")
