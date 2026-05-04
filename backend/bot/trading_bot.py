"""
TradeMinds bot core — pipeline architecture.

Her sinyal şu kapılardan geçmek zorunda:
  Strategy Engine → Signal Validator → Position Guard → Risk Manager → Order Executor

Bu sıralama duplicate emirleri ve aynı sembolde çoklu pozisyon açmayı
mimari olarak imkansız hale getirir. Bir kapı geçilmezse pipeline durur.
"""
import asyncio
import hashlib
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import AsyncSessionLocal
from db.models import (
    BotConfig, BotStatus, BrokerAccount, Strategy, Trade,
    OrderSide, OrderStatus, TradeMode, MarketType, AISignalLog
)
from db.redis_client import cache_set, cache_get, get_redis
from brokers.capital_adapter import CapitalAdapter
from bot.indicators import calculate_rsi, calculate_sma


# ─────────────────────── DATA CLASSES ───────────────────────

@dataclass
class Signal:
    """Strategy engine'in ürettiği ham sinyal."""
    user_id: str
    broker_id: str
    symbol: str
    market_type: MarketType
    side: OrderSide
    confidence: float
    reasoning: str
    indicators: dict
    timestamp: datetime

    def idempotency_key(self) -> str:
        """Aynı sinyalin tekrar işlenmesini engelleyen unique key.
        5 dakikalık pencerede aynı sembol+yön = aynı key."""
        bucket = int(self.timestamp.timestamp() / 300)  # 5 dk bucket
        raw = f"{self.user_id}:{self.symbol}:{self.side.value}:{bucket}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ─────────────────────── PIPELINE GATES ───────────────────────

class SignalValidator:
    """1. Kapı: Idempotency check.
    Aynı sinyalin tekrar işlenmesini engeller (Redis tabanlı)."""

    @staticmethod
    async def validate(signal: Signal) -> tuple[bool, str]:
        key = f"signal:processed:{signal.idempotency_key()}"
        existing = await cache_get(key)
        if existing:
            return False, f"Duplicate signal (key={signal.idempotency_key()})"

        # 10 dakika boyunca işlenmiş olarak işaretle
        await cache_set(key, {"processed_at": datetime.utcnow().isoformat()}, ttl=600)
        return True, "OK"


class PositionGuard:
    """2. Kapı: Açık pozisyon kontrolü.
    Capital.com'dan canlı pozisyonları çeker, aynı symbol+side varsa reddeder.
    Ayrıca max_positions ve market_limits limitlerini kontrol eder."""

    @staticmethod
    async def check(
        signal: Signal,
        adapter: CapitalAdapter,
        config: BotConfig,
    ) -> tuple[bool, str]:
        try:
            positions = await adapter.get_open_orders()
        except Exception as e:
            return False, f"Position fetch failed: {e}"

        # Aynı symbol'de zaten pozisyon var mı?
        for pos in positions:
            if pos.symbol == signal.symbol:
                return False, f"Position already exists on {signal.symbol}"

        # Global max_positions limit
        if len(positions) >= config.max_positions:
            return False, f"Max positions reached ({len(positions)}/{config.max_positions})"

        # Market type limiti
        market_limits = config.market_limits or {}
        market_key = signal.market_type.value  # "forex", "crypto", vs.
        market_limit = market_limits.get(market_key)
        if market_limit:
            same_market_count = sum(
                1 for p in positions
                if PositionGuard._infer_market(p.symbol) == market_key
            )
            if same_market_count >= market_limit:
                return False, f"Market limit reached: {market_key} {same_market_count}/{market_limit}"

        return True, "OK"

    @staticmethod
    def _infer_market(symbol: str) -> str:
        """Capital.com epic'inden market türünü tahmin et (basit heuristic)."""
        s = symbol.upper()
        if any(c in s for c in ("BTC", "ETH", "XRP", "DOGE", "SOL", "ADA")):
            return "crypto"
        if any(c in s for c in ("XAU", "XAG", "OIL", "GAS")):
            return "commodity"
        if any(c in s for c in ("SPX", "NDX", "DJI", "DAX", "FTSE", "NKY")):
            return "index"
        if len(s) == 6 and s.isalpha():
            return "forex"
        return "stock"


class RiskManager:
    """3. Kapı: Lot size hesabı + SL/TP belirleme + balance check."""

    @staticmethod
    async def calculate(
        signal: Signal,
        adapter: CapitalAdapter,
        config: BotConfig,
    ) -> tuple[bool, str, Optional[dict]]:
        try:
            info = await adapter.get_account_info()
        except Exception as e:
            return False, f"Balance fetch failed: {e}", None

        balance = info.balance
        if balance <= 0:
            return False, f"Insufficient balance: {balance}", None

        # Tick fiyatı al (SL/TP hesabı için)
        try:
            tick = await adapter.get_tick(signal.symbol)
        except Exception as e:
            return False, f"Tick fetch failed: {e}", None

        price = tick.ask if signal.side == OrderSide.BUY else tick.bid
        if not price or price <= 0:
            return False, f"Invalid price for {signal.symbol}", None

        # Risk: balance'ın max_risk_per_trade_pct kadarı
        risk_amount = balance * (config.max_risk_per_trade_pct / 100.0)

        # Stop loss: %1 fiyat hareketi (basit, sembol başına customize edilebilir)
        sl_distance_pct = 0.01
        if signal.side == OrderSide.BUY:
            stop_loss = round(price * (1 - sl_distance_pct), 5)
            take_profit = round(price * (1 + sl_distance_pct * 2), 5)  # 1:2 risk:reward
        else:
            stop_loss = round(price * (1 + sl_distance_pct), 5)
            take_profit = round(price * (1 - sl_distance_pct * 2), 5)

        # Lot size = risk_amount / (price * sl_distance_pct)
        # Bu hesap çok basit — Capital.com'un min/max lot kurallarına uymayabilir
        # Şu an minimum güvenli değeri kullanıyoruz: 0.1
        sl_distance = abs(price - stop_loss)
        if sl_distance <= 0:
            return False, "Invalid SL distance", None

        lot_size = round(risk_amount / (sl_distance * 100), 2)
        lot_size = max(0.1, min(lot_size, 1.0))  # 0.1 ile 1.0 arası clamp

        return True, "OK", {
            "lot_size": lot_size,
            "entry_price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }


class OrderExecutor:
    """4. Kapı: Capital.com'a emir gönder + DB'ye trade kaydet."""

    @staticmethod
    async def execute(
        signal: Signal,
        adapter: CapitalAdapter,
        risk_params: dict,
        config: BotConfig,
        db: AsyncSession,
    ) -> tuple[bool, str, Optional[Trade]]:
        deal_ref = await adapter.place_order(
            symbol=signal.symbol,
            side=signal.side.value,
            lot_size=risk_params["lot_size"],
            stop_loss=risk_params["stop_loss"],
            take_profit=risk_params["take_profit"],
            comment=f"TradeMinds:{signal.idempotency_key()}",
        )

        if not deal_ref:
            return False, "Capital.com returned no dealReference", None

        # DB'ye trade kaydet
        trade = Trade(
            user_id=signal.user_id,
            broker_id=signal.broker_id,
            strategy_id=None,
            symbol=signal.symbol,
            market_type=signal.market_type,
            side=signal.side,
            status=OrderStatus.OPEN,
            trade_mode=TradeMode(config.trade_mode.value),
            entry_price=risk_params["entry_price"],
            lot_size=risk_params["lot_size"],
            stop_loss=risk_params["stop_loss"],
            take_profit=risk_params["take_profit"],
            ai_reasoning=signal.reasoning,
            ai_confidence=signal.confidence,
            signals_used=signal.indicators,
            broker_order_id=deal_ref,
            opened_at=datetime.utcnow(),
        )
        db.add(trade)
        await db.commit()
        await db.refresh(trade)

        return True, f"Order placed: {deal_ref}", trade


# ─────────────────────── STRATEGY ENGINE ───────────────────────

class StrategyEngine:
    """RSI + SMA basit strateji.
    RSI < 30 + fiyat SMA20 üstündeyse BUY
    RSI > 70 + fiyat SMA20 altındaysa SELL"""

    @staticmethod
    async def generate_signal(
        user_id: str,
        broker_id: str,
        symbol: str,
        adapter: CapitalAdapter,
    ) -> Optional[Signal]:
        try:
            df = await adapter.get_candles(symbol, "1h", limit=100)
        except Exception as e:
            logger.warning(f"[{symbol}] Candle fetch failed: {e}")
            return None

        if df is None or df.empty or len(df) < 50:
            return None

        try:
            rsi = calculate_rsi(df["close"], period=14)
            sma = calculate_sma(df["close"], period=20)
        except Exception as e:
            logger.warning(f"[{symbol}] Indicator error: {e}")
            return None

        if rsi.empty or sma.empty:
            return None

        last_rsi = float(rsi.iloc[-1])
        last_sma = float(sma.iloc[-1])
        last_close = float(df["close"].iloc[-1])

        side = None
        confidence = 0.0
        reasoning = ""

        if last_rsi < 30 and last_close > last_sma:
            side = OrderSide.BUY
            confidence = min(0.5 + (30 - last_rsi) / 60, 0.95)
            reasoning = f"RSI oversold ({last_rsi:.1f}) + price above SMA20"
        elif last_rsi > 70 and last_close < last_sma:
            side = OrderSide.SELL
            confidence = min(0.5 + (last_rsi - 70) / 60, 0.95)
            reasoning = f"RSI overbought ({last_rsi:.1f}) + price below SMA20"

        if side is None:
            return None

        return Signal(
            user_id=user_id,
            broker_id=broker_id,
            symbol=symbol,
            market_type=MarketType.FOREX,  # default; PositionGuard'ın infer'ı sınırı kontrol eder
            side=side,
            confidence=confidence,
            reasoning=reasoning,
            indicators={
                "rsi": round(last_rsi, 2),
                "sma20": round(last_sma, 5),
                "close": round(last_close, 5),
            },
            timestamp=datetime.utcnow(),
        )


# ─────────────────────── MAIN BOT ───────────────────────

class TradingBot:
    """Ana bot: scheduler her 5 dakikada bir scan() çağırır.
    Tek kullanıcı (admin), tek broker (Capital.com demo)."""

    def __init__(self):
        self._adapter_cache: dict[str, CapitalAdapter] = {}

    async def _get_adapter(self, broker: BrokerAccount) -> Optional[CapitalAdapter]:
        if broker.id in self._adapter_cache:
            adapter = self._adapter_cache[broker.id]
            if await adapter.is_connected():
                return adapter
            # Connection bozulmuş, yeniden bağlan
            await adapter.disconnect()
            del self._adapter_cache[broker.id]

        adapter = CapitalAdapter(broker)
        if not await adapter.connect():
            logger.error(f"Failed to connect to broker {broker.name}")
            return None

        self._adapter_cache[broker.id] = adapter
        return adapter

    async def scan(self):
        """Scheduler'ın çağırdığı ana giriş noktası."""
        logger.info(">>> Bot scan started")

        async with AsyncSessionLocal() as db:
            try:
                # 1. Aktif kullanıcı + bot config
                cfg_result = await db.execute(
                    select(BotConfig).where(BotConfig.status == BotStatus.RUNNING)
                )
                configs = cfg_result.scalars().all()
                if not configs:
                    logger.info("No running bots")
                    return

                for config in configs:
                    await self._scan_for_user(db, config)

            except Exception as e:
                logger.exception(f"Scan failed: {e}")
            finally:
                logger.info("<<< Bot scan finished")

    async def _scan_for_user(self, db: AsyncSession, config: BotConfig):
        # 2. Aktif broker
        br_result = await db.execute(
            select(BrokerAccount).where(
                BrokerAccount.user_id == config.user_id,
                BrokerAccount.is_active == True,
            )
        )
        broker = br_result.scalars().first()
        if not broker:
            logger.warning(f"User {config.user_id} has no active broker")
            return

        # 3. Adapter
        adapter = await self._get_adapter(broker)
        if not adapter:
            return

        # 4. Watchlist sembolleri
        symbols = adapter.get_cached_watchlist_symbols()
        if not symbols:
            logger.warning("Watchlist empty, nothing to scan")
            return

        logger.info(f"Scanning {len(symbols)} symbols for user {config.user_id}")

        # 5. Her sembol için pipeline çalıştır
        processed = 0
        for symbol in symbols:
            try:
                await self._process_symbol(
                    db, config, broker, adapter, symbol
                )
                processed += 1
            except Exception as e:
                logger.exception(f"[{symbol}] Pipeline error: {e}")

        logger.info(f"Scan complete: {processed}/{len(symbols)} symbols processed")

    async def _process_symbol(
        self,
        db: AsyncSession,
        config: BotConfig,
        broker: BrokerAccount,
        adapter: CapitalAdapter,
        symbol: str,
    ):
        # Strategy engine
        signal = await StrategyEngine.generate_signal(
            user_id=config.user_id,
            broker_id=broker.id,
            symbol=symbol,
            adapter=adapter,
        )
        if not signal:
            return  # No signal — quiet exit

        logger.info(f"[{symbol}] Signal: {signal.side.value} conf={signal.confidence:.2f}")

        # AI signal log (audit trail)
        log = AISignalLog(
            user_id=signal.user_id,
            symbol=signal.symbol,
            market_type=signal.market_type,
            signal=signal.side.value,
            confidence=signal.confidence,
            reasoning=signal.reasoning,
            indicators=signal.indicators,
            acted_on=False,
        )
        db.add(log)
        await db.commit()
        await db.refresh(log)

        # Gate 1: Validator
        ok, msg = await SignalValidator.validate(signal)
        if not ok:
            logger.info(f"[{symbol}] REJECTED at Validator: {msg}")
            return

        # Gate 2: Position Guard
        ok, msg = await PositionGuard.check(signal, adapter, config)
        if not ok:
            logger.info(f"[{symbol}] REJECTED at PositionGuard: {msg}")
            return

        # Gate 3: Risk Manager
        ok, msg, risk_params = await RiskManager.calculate(signal, adapter, config)
        if not ok:
            logger.info(f"[{symbol}] REJECTED at RiskManager: {msg}")
            return

        # Gate 4: Order Executor (only on LIVE mode)
        if config.trade_mode != TradeMode.LIVE:
            logger.info(f"[{symbol}] PAPER mode — order not sent. Risk params: {risk_params}")
            return

        ok, msg, trade = await OrderExecutor.execute(
            signal, adapter, risk_params, config, db
        )
        if ok:
            logger.success(f"[{symbol}] ORDER PLACED: {msg}")
            # Mark signal log as acted
            log.acted_on = True
            log.trade_id = trade.id
            await db.commit()
        else:
            logger.error(f"[{symbol}] ORDER FAILED: {msg}")


# Global bot instance (scheduler tarafından kullanılır)
bot_instance = TradingBot()