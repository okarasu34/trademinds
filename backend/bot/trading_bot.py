"""
TradeMinds bot core — pipeline architecture.

Tüm dashboard ayarları aktif:
- max_risk_per_trade_pct → lot size hesabında kullanılır
- max_daily_loss_pct → günlük kayıp limitini kontrol eder
- max_positions → pozisyon limiti
- market_limits → market bazında limit
- pause_on_high_impact_news → haber kontrolü (ileride)
"""
import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field

from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import AsyncSessionLocal
from db.models import (
    BotConfig, BotStatus, BrokerAccount, Trade,
    OrderSide, OrderStatus, TradeMode, MarketType, AISignalLog
)
from db.redis_client import cache_set, cache_get
from brokers.capital_adapter import CapitalAdapter
from bot.indicators import calculate_indicators


# ─────────────────────── DATA CLASSES ───────────────────────

@dataclass
class Signal:
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
        bucket = int(self.timestamp.timestamp() / 300)
        raw = f"{self.user_id}:{self.symbol}:{self.side.value}:{bucket}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class ScanContext:
    open_symbols: set = field(default_factory=set)
    open_count: int = 0
    market_counts: dict = field(default_factory=dict)


# ─────────────────────── PIPELINE GATES ───────────────────────

class SignalValidator:
    """1. Kapı: Idempotency check."""

    @staticmethod
    async def validate(signal: Signal) -> tuple[bool, str]:
        key = f"signal:processed:{signal.idempotency_key()}"
        existing = await cache_get(key)
        if existing:
            return False, f"Duplicate signal (key={signal.idempotency_key()})"
        await cache_set(key, {"processed_at": datetime.utcnow().isoformat()}, ttl=600)
        return True, "OK"


class PositionGuard:
    """2. Kapı: Local cache tabanlı pozisyon kontrolü."""

    @staticmethod
    def check(
        signal: Signal,
        ctx: ScanContext,
        config: BotConfig,
    ) -> tuple[bool, str]:
        if signal.symbol in ctx.open_symbols:
            return False, f"Position already open: {signal.symbol}"

        if ctx.open_count >= config.max_positions:
            return False, f"Max positions reached ({ctx.open_count}/{config.max_positions})"

        market_limits = config.market_limits or {}
        market_key    = signal.market_type.value
        market_limit  = market_limits.get(market_key)
        if market_limit:
            current = ctx.market_counts.get(market_key, 0)
            if current >= market_limit:
                return False, f"Market limit reached: {market_key} {current}/{market_limit}"

        return True, "OK"

    @staticmethod
    def register_open(signal: Signal, ctx: ScanContext):
        ctx.open_symbols.add(signal.symbol)
        ctx.open_count += 1
        market_key = signal.market_type.value
        ctx.market_counts[market_key] = ctx.market_counts.get(market_key, 0) + 1

    @staticmethod
    def _infer_market(symbol: str) -> str:
        s = symbol.upper()
        if any(c in s for c in ("BTC","ETH","XRP","DOGE","SOL","ADA","AAVE","AVAX","LTC","SHIB","PEPE","TRX","HBAR","XLM","ALPHA","USDT")):
            return "crypto"
        if any(c in s for c in ("XAU","XAG","OIL","GAS","GOLD","SILVER","PLATINUM","PALLADIUM","COPPER","CORN","WHEAT","NATURALGAS","BRENT")):
            return "commodity"
        if any(c in s for c in ("SPX","NDX","DJI","DAX","FTSE","NKY","US100","US500","US30","DE40")):
            return "index"
        if len(s) == 6 and s.isalpha():
            return "forex"
        return "stock"


class DailyLossGuard:
    """3. Kapı: Günlük kayıp limiti kontrolü.
    Dashboard'daki max_daily_loss_pct ayarını kullanır."""

    @staticmethod
    async def check(
        config: BotConfig,
        db: AsyncSession,
        broker: BrokerAccount,
        adapter: CapitalAdapter,
    ) -> tuple[bool, str]:
        if not config.max_daily_loss_pct or config.max_daily_loss_pct <= 0:
            return True, "OK"

        try:
            info = await adapter.get_account_info()
            balance = info.balance
        except Exception as e:
            return False, f"Balance fetch failed: {e}"

        if balance <= 0:
            return False, "Invalid balance"

        # Bugün kapanan trade'lerin toplam PnL'ini hesapla
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        result = await db.execute(
            select(func.coalesce(func.sum(Trade.pnl), 0)).where(
                Trade.user_id == config.user_id,
                Trade.status == OrderStatus.CLOSED,
                Trade.closed_at >= today_start,
                Trade.pnl.isnot(None),
            )
        )
        daily_pnl = float(result.scalar() or 0)

        # Kayıp miktarı balance'ın max_daily_loss_pct'ini aştı mı?
        max_loss = balance * (config.max_daily_loss_pct / 100.0)
        if daily_pnl < 0 and abs(daily_pnl) >= max_loss:
            return False, (
                f"Daily loss limit reached: {daily_pnl:.2f} / -{max_loss:.2f} "
                f"({config.max_daily_loss_pct}% of {balance:.2f})"
            )

        return True, "OK"


class RiskManager:
    """4. Kapı: Lot size hesabı + SL/TP.
    Dashboard'daki max_risk_per_trade_pct ayarını kullanır."""

    @staticmethod
    async def calculate(
        signal: Signal,
        adapter: CapitalAdapter,
        config: BotConfig,
    ) -> tuple[bool, str, Optional[dict]]:
        try:
            rules = await adapter.get_market_rules(signal.symbol)
        except Exception as e:
            return False, f"Market rules fetch failed: {e}", None

        min_size     = rules.get("min_size", 1.0)
        min_stop_pct = rules.get("min_stop_pct", 0.1)
        bid          = rules.get("bid", 0)
        ask          = rules.get("ask", 0)

        if not bid or not ask:
            return False, f"Invalid price for {signal.symbol}", None

        price = ask if signal.side == OrderSide.BUY else bid

        # SL/TP mesafesi
        sl_distance_pct = max(min_stop_pct / 100.0 * 1.5, 0.001)

        if signal.side == OrderSide.BUY:
            stop_loss   = round(price * (1 - sl_distance_pct), 5)
            take_profit = round(price * (1 + sl_distance_pct * 2), 5)
        else:
            stop_loss   = round(price * (1 + sl_distance_pct), 5)
            take_profit = round(price * (1 - sl_distance_pct * 2), 5)

        # Lot size: max_risk_per_trade_pct bazlı hesap
        if config.max_risk_per_trade_pct and config.max_risk_per_trade_pct > 0:
            try:
                info = await adapter.get_account_info()
                balance = info.balance
                if balance > 0:
                    risk_amount  = balance * (config.max_risk_per_trade_pct / 100.0)
                    sl_distance  = abs(price - stop_loss)
                    if sl_distance > 0:
                        calculated_lot = round(risk_amount / (sl_distance * 100), 2)
                        # min_size ile max(1 lot veya hesaplanan) arasında clamp
                        lot_size = max(float(min_size), min(calculated_lot, float(min_size) * 10))
                    else:
                        lot_size = float(min_size)
                else:
                    lot_size = float(min_size)
            except Exception:
                lot_size = float(min_size)
        else:
            lot_size = float(min_size)

        logger.info(
            f"[{signal.symbol}] Rules: min_size={min_size} risk_pct={config.max_risk_per_trade_pct}% "
            f"→ lot={lot_size} sl={stop_loss} tp={take_profit}"
        )

        return True, "OK", {
            "lot_size":    lot_size,
            "entry_price": price,
            "stop_loss":   stop_loss,
            "take_profit": take_profit,
        }


class OrderExecutor:
    """5. Kapı: Capital.com'a emir gönder + DB'ye trade kaydet."""

    @staticmethod
    async def execute(
        signal: Signal,
        adapter: CapitalAdapter,
        risk_params: dict,
        config: BotConfig,
        db: AsyncSession,
    ) -> tuple[bool, str, Optional[Trade]]:
        deal_ref = await adapter.place_order(
            symbol      = signal.symbol,
            side        = signal.side.value,
            lot_size    = risk_params["lot_size"],
            stop_loss   = risk_params["stop_loss"],
            take_profit = risk_params["take_profit"],
            comment     = f"TradeMinds:{signal.idempotency_key()}",
        )

        if not deal_ref:
            return False, "Capital.com returned no dealReference", None

        trade = Trade(
            user_id         = signal.user_id,
            broker_id       = signal.broker_id,
            strategy_id     = None,
            symbol          = signal.symbol,
            market_type     = signal.market_type,
            side            = signal.side,
            status          = OrderStatus.OPEN,
            trade_mode      = TradeMode(config.trade_mode.value),
            entry_price     = risk_params["entry_price"],
            lot_size        = risk_params["lot_size"],
            stop_loss       = risk_params["stop_loss"],
            take_profit     = risk_params["take_profit"],
            ai_reasoning    = signal.reasoning,
            ai_confidence   = signal.confidence,
            signals_used    = signal.indicators,
            broker_order_id = deal_ref,
            opened_at       = datetime.utcnow(),
        )
        db.add(trade)
        await db.commit()
        await db.refresh(trade)

        return True, f"Order placed: {deal_ref}", trade


# ─────────────────────── STRATEGY ENGINE ───────────────────────

class StrategyEngine:
    """RSI momentum + SMA trend konfirmasyonu.

    BUY:  RSI > 50 VE fiyat SMA20 üzerinde (yukarı momentum + trend)
    SELL: RSI < 50 VE fiyat SMA20 altında (aşağı momentum + trend)
    """

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
            ind = calculate_indicators(df)
        except Exception as e:
            logger.warning(f"[{symbol}] Indicator error: {e}")
            return None

        if not ind or "rsi_14" not in ind or "sma_20" not in ind:
            return None

        last_rsi   = ind["rsi_14"]
        last_sma   = ind["sma_20"]
        last_close = float(df["close"].iloc[-1])

        if last_rsi is None or last_sma is None:
            return None

        side       = None
        confidence = 0.0
        reasoning  = ""

        if last_rsi > 50 and last_close > last_sma:
            side       = OrderSide.BUY
            confidence = min(0.5 + (last_rsi - 50) / 100, 0.95)
            reasoning  = f"RSI={last_rsi:.1f} > 50 + price above SMA20 → BUY"
        elif last_rsi < 50 and last_close < last_sma:
            side       = OrderSide.SELL
            confidence = min(0.5 + (50 - last_rsi) / 100, 0.95)
            reasoning  = f"RSI={last_rsi:.1f} < 50 + price below SMA20 → SELL"

        if side is None:
            return None

        inferred   = PositionGuard._infer_market(symbol)
        market_map = {
            "crypto":    MarketType.CRYPTO,
            "commodity": MarketType.COMMODITY,
            "index":     MarketType.INDEX,
            "stock":     MarketType.STOCK,
            "forex":     MarketType.FOREX,
        }
        market_type = market_map.get(inferred, MarketType.FOREX)

        return Signal(
            user_id     = user_id,
            broker_id   = broker_id,
            symbol      = symbol,
            market_type = market_type,
            side        = side,
            confidence  = confidence,
            reasoning   = reasoning,
            indicators  = {
                "rsi":   round(last_rsi, 2),
                "sma20": round(last_sma, 5),
                "close": round(last_close, 5),
            },
            timestamp   = datetime.utcnow(),
        )


# ─────────────────────── MAIN BOT ───────────────────────

class TradingBot:
    def __init__(self):
        self._adapter_cache: dict[str, CapitalAdapter] = {}

    async def _get_adapter(self, broker: BrokerAccount) -> Optional[CapitalAdapter]:
        if broker.id in self._adapter_cache:
            adapter = self._adapter_cache[broker.id]
            if await adapter.is_connected():
                return adapter
            await adapter.disconnect()
            del self._adapter_cache[broker.id]

        adapter = CapitalAdapter(broker)
        if not await adapter.connect():
            logger.error(f"Failed to connect to broker {broker.name}")
            return None

        self._adapter_cache[broker.id] = adapter
        return adapter

    async def scan(self):
        logger.info(">>> Bot scan started")

        async with AsyncSessionLocal() as db:
            try:
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

        adapter = await self._get_adapter(broker)
        if not adapter:
            return

        symbols = adapter.get_cached_watchlist_symbols()
        if not symbols:
            logger.warning("Watchlist empty, nothing to scan")
            return

        # Günlük kayıp limiti kontrolü (tüm scan için bir kez)
        ok, msg = await DailyLossGuard.check(config, db, broker, adapter)
        if not ok:
            logger.warning(f"Daily loss limit hit — scan paused: {msg}")
            return

        try:
            live_positions = await adapter.get_open_orders()
        except Exception as e:
            logger.error(f"Failed to fetch live positions: {e}")
            return

        ctx = ScanContext(
            open_symbols  = {p.symbol for p in live_positions},
            open_count    = len(live_positions),
            market_counts = {},
        )
        for p in live_positions:
            mk = PositionGuard._infer_market(p.symbol)
            ctx.market_counts[mk] = ctx.market_counts.get(mk, 0) + 1

        logger.info(
            f"Scanning {len(symbols)} symbols | "
            f"open={ctx.open_count} | "
            f"daily_loss_pct={config.max_daily_loss_pct}% | "
            f"risk_per_trade={config.max_risk_per_trade_pct}%"
        )

        processed = 0
        for symbol in symbols:
            try:
                await self._process_symbol(db, config, broker, adapter, symbol, ctx)
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
        ctx: ScanContext,
    ):
        signal = await StrategyEngine.generate_signal(
            user_id   = config.user_id,
            broker_id = broker.id,
            symbol    = symbol,
            adapter   = adapter,
        )
        if not signal:
            return

        logger.info(f"[{symbol}] Signal: {signal.side.value} conf={signal.confidence:.2f}")

        log = AISignalLog(
            user_id     = signal.user_id,
            symbol      = signal.symbol,
            market_type = signal.market_type,
            signal      = signal.side.value,
            confidence  = signal.confidence,
            reasoning   = signal.reasoning,
            indicators  = signal.indicators,
            acted_on    = False,
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
        ok, msg = PositionGuard.check(signal, ctx, config)
        if not ok:
            logger.info(f"[{symbol}] REJECTED at PositionGuard: {msg}")
            return

        # Gate 3: Risk Manager
        ok, msg, risk_params = await RiskManager.calculate(signal, adapter, config)
        if not ok:
            logger.info(f"[{symbol}] REJECTED at RiskManager: {msg}")
            return

        # Gate 4: Order Executor (sadece LIVE modda)
        if config.trade_mode != TradeMode.LIVE:
            logger.info(f"[{symbol}] PAPER mode — order not sent")
            return

        ok, msg, trade = await OrderExecutor.execute(
            signal, adapter, risk_params, config, db
        )
        if ok:
            logger.success(f"[{symbol}] ORDER PLACED: {msg}")
            PositionGuard.register_open(signal, ctx)
            log.acted_on = True
            log.trade_id = trade.id
            await db.commit()
        else:
            logger.error(f"[{symbol}] ORDER FAILED: {msg}")


# Global bot instance
bot_instance = TradingBot()