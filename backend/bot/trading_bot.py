"""
TradeMinds bot core — pipeline architecture + 3 strateji.

Stratejiler DB'den çekilir, aktif olan kullanılır:
  1. AlphaTrendStrategy   — Alpha Trend crossover + EMA hizalaması
  2. RSIDivergenceStrategy — RSI + MACD diverjans + Fibonacci seviyeleri
  3. SmartMoneyStrategy   — Order Block + FVG + POC

Pipeline:
  Strategy Engine → Signal Validator → Position Guard → Risk Manager → Order Executor

Fix: Redis-based ordered symbol cache — aynı sembol 1 saat içinde tekrar açılmaz.
"""
import asyncio
import hashlib
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import AsyncSessionLocal
from db.models import (
    BotConfig, BotStatus, BrokerAccount, Trade, Strategy,
    OrderSide, OrderStatus, TradeMode, MarketType, AISignalLog
)
from db.redis_client import cache_set, cache_get
from brokers.capital_adapter import CapitalAdapter
from bot.indicators import calculate_indicators


# ─────────────────────── SL/TP MESAFE TABLOSU ───────────────────────

MARKET_SL_PCT = {
    "forex":     0.30,
    "crypto":    1.50,
    "index":     0.50,
    "commodity": 0.80,
    "stock":     1.00,
}
TP_RATIO = 2.0


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
    strategy_id: Optional[str] = None

    def idempotency_key(self) -> str:
        bucket = int(self.timestamp.timestamp() / 300)
        raw = f"{self.user_id}:{self.symbol}:{self.side.value}:{bucket}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class ScanContext:
    open_symbols: set = field(default_factory=set)
    open_count: int = 0
    market_counts: dict = field(default_factory=dict)


# ─────────────────────── STRATEJI SINIFLAR ───────────────────────

class AlphaTrendStrategy:
    """
    Strateji 1: Alpha Trend + EMA Hizalaması

    BUY: Alpha Trend yukarı döndü + EMA13 > EMA21 + fiyat EMA89 üzerinde
    SELL: Alpha Trend aşağı döndü + EMA13 < EMA21 + fiyat EMA89 altında
    """

    @staticmethod
    def generate(ind: dict, params: dict) -> Optional[tuple[OrderSide, float, str]]:
        at_cross_up   = ind.get("alpha_trend_cross_up", False)
        at_cross_down = ind.get("alpha_trend_cross_down", False)
        ema13_gt_21   = ind.get("ema13_above_ema21", False)
        ema13_gt_89   = ind.get("ema13_above_ema89", False)
        rsi           = ind.get("rsi_13", 50)
        macd_cross    = ind.get("macd13_crossover", "bearish")

        if at_cross_up and ema13_gt_21 and ema13_gt_89:
            confidence = 0.65
            if macd_cross == "bullish":
                confidence += 0.10
            if rsi > 50:
                confidence += 0.05
            return (
                OrderSide.BUY,
                min(confidence, 0.95),
                f"AlphaTrend cross UP | EMA13>{ind.get('ema_13','?')} > EMA21 | RSI={rsi:.1f}"
            )

        if at_cross_down and not ema13_gt_21 and not ema13_gt_89:
            confidence = 0.65
            if macd_cross == "bearish":
                confidence += 0.10
            if rsi < 50:
                confidence += 0.05
            return (
                OrderSide.SELL,
                min(confidence, 0.95),
                f"AlphaTrend cross DOWN | EMA13 < EMA21 | RSI={rsi:.1f}"
            )

        return None


class RSIDivergenceStrategy:
    """
    Strateji 2: RSI + MACD Diverjans + Fibonacci

    BUY: RSI/MACD bullish diverjans + MACD cross up + Fibonacci desteği
    SELL: RSI/MACD bearish diverjans + MACD cross down + Fibonacci direnci
    """

    @staticmethod
    def generate(ind: dict, params: dict) -> Optional[tuple[OrderSide, float, str]]:
        rsi_bull_div   = ind.get("rsi_bullish_divergence", False)
        rsi_bear_div   = ind.get("rsi_bearish_divergence", False)
        macd_bull_div  = ind.get("macd_bullish_divergence", False)
        macd_bear_div  = ind.get("macd_bearish_divergence", False)
        macd_cross_up  = ind.get("macd13_cross_up", False)
        macd_cross_dn  = ind.get("macd13_cross_down", False)
        near_fibo      = ind.get("near_fibo_level")
        rsi            = ind.get("rsi_13", 50)

        min_div        = params.get("min_divergence_count", 2)
        bull_div_count = sum([rsi_bull_div, macd_bull_div])
        bear_div_count = sum([rsi_bear_div, macd_bear_div])

        if bull_div_count >= min_div:
            confidence = 0.60 + bull_div_count * 0.10
            if near_fibo in ("fibo_382", "fibo_500", "fibo_618"):
                confidence += 0.10
            if rsi < 45:
                confidence += 0.05
            return (
                OrderSide.BUY,
                min(confidence, 0.95),
                f"RSI+MACD bullish div ({bull_div_count}) | MACD cross up | Fibo={near_fibo} | RSI={rsi:.1f}"
            )

        if bear_div_count >= min_div:
            confidence = 0.60 + bear_div_count * 0.10
            if near_fibo in ("fibo_618", "fibo_786", "fibo_1000"):
                confidence += 0.10
            if rsi > 55:
                confidence += 0.05
            return (
                OrderSide.SELL,
                min(confidence, 0.95),
                f"RSI+MACD bearish div ({bear_div_count}) | MACD cross down | Fibo={near_fibo} | RSI={rsi:.1f}"
            )

        if bull_div_count >= 1 and near_fibo in ("fibo_382", "fibo_618"):
            return (
                OrderSide.BUY,
                0.65,
                f"RSI/MACD bullish div + Fibo {near_fibo} destek | RSI={rsi:.1f}"
            )

        if bear_div_count >= 1 and near_fibo in ("fibo_618", "fibo_786"):
            return (
                OrderSide.SELL,
                0.65,
                f"RSI/MACD bearish div + Fibo {near_fibo} direnç | RSI={rsi:.1f}"
            )

        return None


class SmartMoneyStrategy:
    """
    Strateji 3: Order Block + FVG + POC

    BUY: Fiyat bullish OB içinde + POC yakın + bullish FVG + MACD bullish
    SELL: Fiyat bearish OB içinde + POC yakın + bearish FVG + MACD bearish
    """

    @staticmethod
    def generate(ind: dict, params: dict) -> Optional[tuple[OrderSide, float, str]]:
        in_bull_ob    = ind.get("in_bullish_ob", False)
        in_bear_ob    = ind.get("in_bearish_ob", False)
        near_poc      = ind.get("near_poc", False)
        bull_fvg      = ind.get("bull_fvg", False)
        bear_fvg      = ind.get("bear_fvg", False)
        macd_cross    = ind.get("macd13_crossover", "bearish")
        poc_prox      = ind.get("poc_proximity_pct", 999)
        rsi           = ind.get("rsi_13", 50)
        poc_threshold = params.get("poc_proximity_pct", 0.5)

        if in_bull_ob and poc_prox <= poc_threshold:
            confidence = 0.65
            if bull_fvg:
                confidence += 0.10
            if macd_cross == "bullish":
                confidence += 0.10
            if rsi < 50:
                confidence += 0.05
            return (
                OrderSide.BUY,
                min(confidence, 0.95),
                f"Bullish OB + POC ({poc_prox:.2f}%) | FVG={bull_fvg} | MACD={macd_cross} | RSI={rsi:.1f}"
            )

        if in_bear_ob and poc_prox <= poc_threshold:
            confidence = 0.65
            if bear_fvg:
                confidence += 0.10
            if macd_cross == "bearish":
                confidence += 0.10
            if rsi > 50:
                confidence += 0.05
            return (
                OrderSide.SELL,
                min(confidence, 0.95),
                f"Bearish OB + POC ({poc_prox:.2f}%) | FVG={bear_fvg} | MACD={macd_cross} | RSI={rsi:.1f}"
            )

        return None


# ─────────────────────── STRATEJİ FACTORY ───────────────────────

STRATEGY_MAP = {
    "Alpha Trend":    AlphaTrendStrategy,
    "RSI Divergence": RSIDivergenceStrategy,
    "Smart Money":    SmartMoneyStrategy,
}


# ─────────────────────── PIPELINE GATES ───────────────────────

class SignalValidator:
    @staticmethod
    async def validate(signal: Signal) -> tuple[bool, str]:
        key = f"signal:processed:{signal.idempotency_key()}"
        existing = await cache_get(key)
        if existing:
            return False, f"Duplicate signal (key={signal.idempotency_key()})"
        await cache_set(key, {"processed_at": datetime.utcnow().isoformat()}, ttl=3600)
        return True, "OK"


class OrderedSymbolCache:
    """Başarılı emir açılan sembolleri Redis'te saklar.
    Bir sonraki scan'de aynı sembol tekrar açılmaz (1 saat TTL)."""

    @staticmethod
    async def add(user_id: str, symbol: str, ttl: int = 3600):
        key = f"ordered:{user_id}:{symbol}"
        await cache_set(key, {"ordered_at": datetime.utcnow().isoformat()}, ttl=ttl)

    @staticmethod
    async def exists(user_id: str, symbol: str) -> bool:
        key = f"ordered:{user_id}:{symbol}"
        return await cache_get(key) is not None

    @staticmethod
    async def get_all_ordered(user_id: str, symbols: list) -> set:
        ordered = set()
        for symbol in symbols:
            if await OrderedSymbolCache.exists(user_id, symbol):
                ordered.add(symbol)
        return ordered


class PositionGuard:
    @staticmethod
    def check(signal: Signal, ctx: ScanContext, config: BotConfig) -> tuple[bool, str]:
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
    @staticmethod
    async def check(config: BotConfig, db: AsyncSession, broker: BrokerAccount, adapter: CapitalAdapter) -> tuple[bool, str]:
        if not config.max_daily_loss_pct or config.max_daily_loss_pct <= 0:
            return True, "OK"
        try:
            info    = await adapter.get_account_info()
            balance = info.balance
        except Exception as e:
            return False, f"Balance fetch failed: {e}"
        if balance <= 0:
            return False, "Invalid balance"
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
        max_loss  = balance * (config.max_daily_loss_pct / 100.0)
        if daily_pnl < 0 and abs(daily_pnl) >= max_loss:
            return False, f"Daily loss limit reached: {daily_pnl:.2f} / -{max_loss:.2f}"
        return True, "OK"


class RiskManager:
    @staticmethod
    async def calculate(signal: Signal, adapter: CapitalAdapter, config: BotConfig) -> tuple[bool, str, Optional[dict]]:
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

        market_key  = PositionGuard._infer_market(signal.symbol)
        sl_pct      = MARKET_SL_PCT.get(market_key, 0.50) / 100.0
        sl_distance = price * sl_pct
        tp_distance = sl_distance * TP_RATIO

        min_sl_distance = price * (min_stop_pct / 100.0)
        if sl_distance < min_sl_distance:
            sl_distance = min_sl_distance * 1.5
            tp_distance = sl_distance * TP_RATIO

        if signal.side == OrderSide.BUY:
            stop_loss   = round(price - sl_distance, 5)
            take_profit = round(price + tp_distance, 5)
        else:
            stop_loss   = round(price + sl_distance, 5)
            take_profit = round(price - tp_distance, 5)

        risk_pct = (config.max_risk_per_trade_pct or 1.0) / 100.0
        try:
            info    = await adapter.get_account_info()
            balance = info.balance if info.balance > 0 else 10000
        except Exception:
            balance = 10000

        risk_amount    = balance * risk_pct
        calculated_lot = risk_amount / (sl_distance * 100) if sl_distance > 0 else float(min_size)
        lot_size = max(float(min_size), min(round(calculated_lot, 2), float(min_size) * 50))

        logger.info(
            f"[{signal.symbol}] {market_key} | sl={sl_pct*100:.2f}% | "
            f"risk={config.max_risk_per_trade_pct}% | lot={lot_size} | "
            f"sl_price={stop_loss} tp_price={take_profit}"
        )

        return True, "OK", {
            "lot_size":    lot_size,
            "entry_price": price,
            "stop_loss":   stop_loss,
            "take_profit": take_profit,
        }


class OrderExecutor:
    @staticmethod
    async def execute(signal: Signal, adapter: CapitalAdapter, risk_params: dict, config: BotConfig, db: AsyncSession) -> tuple[bool, str, Optional[Trade]]:
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
            strategy_id     = signal.strategy_id,
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

    async def _get_active_strategy(self, db: AsyncSession, user_id: str) -> Optional[Strategy]:
        result = await db.execute(
            select(Strategy).where(
                Strategy.user_id == user_id,
                Strategy.is_active == True,
            )
        )
        return result.scalars().first()

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

        strategy = await self._get_active_strategy(db, config.user_id)
        if not strategy:
            logger.warning("No active strategy found")
            return

        strategy_class = STRATEGY_MAP.get(strategy.name)
        if not strategy_class:
            logger.warning(f"Unknown strategy: {strategy.name}")
            return

        logger.info(f"Active strategy: {strategy.name}")

        symbols = adapter.get_cached_watchlist_symbols()
        if not symbols:
            logger.warning("Watchlist empty, nothing to scan")
            return

        ok, msg = await DailyLossGuard.check(config, db, broker, adapter)
        if not ok:
            logger.warning(f"Daily loss limit hit: {msg}")
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

        # Redis cache'ten daha önce emir açılmış sembolleri çek
        try:
            redis_ordered = await OrderedSymbolCache.get_all_ordered(config.user_id, symbols)
            ctx.open_symbols.update(redis_ordered)
            if redis_ordered:
                logger.info(f"Redis cache: {len(redis_ordered)} symbols blocked")
        except Exception as e:
            logger.warning(f"Redis cache read failed: {e}")

        logger.info(
            f"Scanning {len(symbols)} symbols | open={ctx.open_count} | "
            f"strategy={strategy.name} | risk={config.max_risk_per_trade_pct}%"
        )

        processed = 0
        for symbol in symbols:
            try:
                await self._process_symbol(db, config, broker, adapter, symbol, ctx, strategy, strategy_class)
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
        strategy: Strategy,
        strategy_class,
    ):
        try:
            df = await adapter.get_candles(symbol, "1h", limit=200)
        except Exception as e:
            logger.warning(f"[{symbol}] Candle fetch failed: {e}")
            return

        if df is None or df.empty or len(df) < 50:
            return

        try:
            ind = calculate_indicators(df)
        except Exception as e:
            logger.warning(f"[{symbol}] Indicator error: {e}")
            return

        if not ind:
            return

        result = strategy_class.generate(ind, strategy.parameters or {})
        if not result:
            return

        side, confidence, reasoning = result

        # EMA200 trend filtresi — trend'e karşı pozisyon açma
        ema200 = ind.get("ema_89") or ind.get("ema_200")
        current_price = ind.get("current_price")
        if ema200 and current_price:
            trend_up = current_price > ema200
            if side == OrderSide.BUY and not trend_up:
                logger.info(f"[{symbol}] REJECTED by EMA200: BUY ama trend ASAGI (price={current_price:.5f} < EMA200={ema200:.5f})")
                return
            if side == OrderSide.SELL and trend_up:
                logger.info(f"[{symbol}] REJECTED by EMA200: SELL ama trend YUKARI (price={current_price:.5f} > EMA200={ema200:.5f})")
                return

        inferred   = PositionGuard._infer_market(symbol)
        market_map = {
            "crypto":    MarketType.CRYPTO,
            "commodity": MarketType.COMMODITY,
            "index":     MarketType.INDEX,
            "stock":     MarketType.STOCK,
            "forex":     MarketType.FOREX,
        }
        market_type = market_map.get(inferred, MarketType.FOREX)

        signal = Signal(
            user_id     = config.user_id,
            broker_id   = broker.id,
            symbol      = symbol,
            market_type = market_type,
            side        = side,
            confidence  = confidence,
            reasoning   = reasoning,
            indicators  = {
                "rsi":         ind.get("rsi_13"),
                "macd":        ind.get("macd13"),
                "alpha_trend": ind.get("alpha_trend"),
                "ema_13":      ind.get("ema_13"),
                "ema_21":      ind.get("ema_21"),
            },
            timestamp   = datetime.utcnow(),
            strategy_id = strategy.id,
        )

        logger.info(f"[{symbol}] [{strategy.name}] Signal: {side.value} conf={confidence:.2f} | {reasoning}")

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

        # Gate 4: Order Executor
        if config.trade_mode != TradeMode.LIVE:
            logger.info(f"[{symbol}] PAPER mode — order not sent")
            return

        ok, msg, trade = await OrderExecutor.execute(signal, adapter, risk_params, config, db)
        if ok:
            logger.success(f"[{symbol}] ORDER PLACED: {msg}")
            PositionGuard.register_open(signal, ctx)
            # Redis cache'e ekle — bir sonraki scan'de bu sembol atlanır
            await OrderedSymbolCache.add(signal.user_id, symbol, ttl=3600)
            log.acted_on = True
            log.trade_id = trade.id
            await db.commit()
        else:
            logger.error(f"[{symbol}] ORDER FAILED: {msg}")


# Global bot instance
bot_instance = TradingBot()