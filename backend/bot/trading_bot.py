import asyncio
from datetime import datetime
from typing import Optional
from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from bot.ai_engine import analyze_market
from bot.indicators import (
    calculate_indicators, detect_patterns,
    calculate_lot_size_from_risk, validate_sl_tp,
    apply_strategy_filters, get_mtf_trend,
)
from brokers.base_adapter import get_broker_adapter, BrokerAdapter
from data.calendar import calendar_client
from risk.risk_manager import RiskManager
from db.models import (
    BotConfig, BotStatus, Trade, Strategy, BrokerAccount,
    OrderStatus, OrderSide, TradeMode, AISignalLog, BotHealthLog,
)
from db.database import AsyncSessionLocal
from db.redis_client import set_bot_state, set_live_price, publish
from core.config import settings
from notifications.notifier import Notifier


class TradingBot:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.is_running = False
        self.adapters: dict[str, BrokerAdapter] = {}
        self.notifier = Notifier(user_id)
        self._loop_task: Optional[asyncio.Task] = None
        self.config: Optional[BotConfig] = None

    # ── Lifecycle ──

    async def start(self, config: BotConfig):
        if self.is_running:
            return
        self.config = config
        self.is_running = True
        logger.info(f"Bot starting | user={self.user_id} | mode={config.trade_mode.value}")
        await self._connect_brokers()
        await set_bot_state(self.user_id, {"status": "running", "started_at": datetime.utcnow().isoformat()})
        self._loop_task = asyncio.create_task(self._main_loop())

    async def stop(self):
        self.is_running = False
        if self._loop_task:
            self._loop_task.cancel()
        await self._disconnect_brokers()
        await set_bot_state(self.user_id, {"status": "stopped"})
        logger.info(f"Bot stopped | user={self.user_id}")

    async def pause(self):
        self.is_running = False
        if self._loop_task:
            self._loop_task.cancel()
        await set_bot_state(self.user_id, {"status": "paused"})

    # ── Main Loop ──

    async def _main_loop(self):
        while self.is_running:
            try:
                await self._health_check()
                await self._scan_and_execute()
                await self._sync_open_positions()
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Bot loop error | user={self.user_id}: {e}")
                await self.notifier.send(f"⚠️ Bot error: {str(e)[:200]}", level="error")
                await asyncio.sleep(30)

    # ── Scan & Execute ──

    async def _scan_and_execute(self):
        async with AsyncSessionLocal() as db:
            strategies = await self._get_active_strategies(db)
            if not strategies:
                return

            open_positions = await self._get_open_positions(db)
            account_info = await self._get_consolidated_account()
            risk_manager = RiskManager(self.config)

            if risk_manager.should_emergency_stop(account_info["balance"]):
                logger.warning(f"Emergency stop | user={self.user_id}")
                await self.notifier.send("🛑 Daily loss limit reached. Bot paused.", level="critical")
                async with AsyncSessionLocal() as db2:
                    await db2.execute(update(BotConfig).where(BotConfig.user_id == self.user_id).values(status=BotStatus.PAUSED))
                    await db2.commit()
                await self.pause()
                return

            upcoming_news = await calendar_client.get_upcoming_high_impact(minutes_ahead=self.config.news_pause_minutes)

            for strategy in strategies:
                for market_type in (strategy.markets or []):
                    for symbol in self._get_symbols(strategy, market_type):
                        try:
                            await self._analyze_and_trade(db, symbol, market_type, strategy, open_positions, account_info, upcoming_news, risk_manager)
                            await asyncio.sleep(1)
                        except Exception as e:
                            logger.error(f"Error on {symbol}: {e}")

    async def _analyze_and_trade(self, db, symbol, market_type, strategy, open_positions, account_info, upcoming_news, risk_manager):
        adapter = self.adapters.get(market_type)
        if not adapter:
            return

        try:
            tick = await adapter.get_tick(symbol)
            await set_live_price(symbol, tick.price, tick.bid, tick.ask)
            # Entry timeframe: 1h (200 bars)
            df_1h = await adapter.get_candles(symbol, "1h", 200)
            # Higher timeframe: 4h for trend filter (100 bars = ~16 days)
            df_4h = await adapter.get_candles(symbol, "4h", 100)
        except Exception as e:
            logger.warning(f"Data fetch failed {symbol}: {e}")
            return

        # Entry timeframe indicators
        indicators = calculate_indicators(df_1h)
        patterns = detect_patterns(df_1h)
        if patterns:
            indicators["patterns"] = patterns

        # Higher timeframe trend filter
        htf_indicators = calculate_indicators(df_4h) if df_4h is not None and len(df_4h) >= 50 else {}
        htf_trend = get_mtf_trend(htf_indicators)
        indicators["htf_trend_4h"] = htf_trend
        indicators["htf_adx_4h"] = htf_indicators.get("adx")
        indicators["htf_ema50_4h"] = htf_indicators.get("ema_50")
        indicators["htf_ema200_4h"] = htf_indicators.get("ema_200")

        # Apply strategy parameter filters so AI sees exact thresholds
        strategy_type = strategy.strategy_type.value if hasattr(strategy.strategy_type, "value") else str(strategy.strategy_type)
        indicators = apply_strategy_filters(indicators, strategy_type, strategy.parameters or {})

        symbol_events = await calendar_client.get_events_for_symbol(symbol, hours_ahead=4)
        news_items = [f"{e['title']} ({e['currency']}, {e['impact']}, in {e['minutes_until']:.0f}min)" for e in symbol_events]

        daily_pnl = -(self.config.daily_loss or 0.0)

        signal = await analyze_market(
            symbol=symbol, market_type=market_type,
            strategy_name=strategy.name, strategy_params=strategy.parameters or {},
            indicators=indicators, recent_news=news_items, economic_events=upcoming_news,
            current_price=tick.price, bid=tick.bid, ask=tick.ask, spread=tick.spread,
            open_positions=len(open_positions), max_positions=self.config.max_positions,
            account_balance=account_info["balance"], daily_pnl=daily_pnl,
            max_daily_loss_pct=self.config.max_daily_loss_pct,
            max_risk_pct=self.config.max_risk_per_trade_pct,
            ai_system_prompt_override=strategy.ai_system_prompt,
        )

        await self._log_signal(db, symbol, market_type, signal)

        if signal["signal"] == "hold" or signal.get("confidence", 0) < 0.65:
            return

        entry_price = tick.ask if signal["signal"] == "buy" else tick.bid
        atr = indicators.get("atr_14", entry_price * 0.01)

        # FIX: Validate AI-provided SL/TP before using them
        raw_sl = signal.get("stop_loss", 0)
        raw_tp = signal.get("take_profit", 0)
        stop_loss, take_profit = validate_sl_tp(
            signal=signal["signal"],
            entry_price=entry_price,
            stop_loss=raw_sl if raw_sl else (entry_price - atr * 1.5 if signal["signal"] == "buy" else entry_price + atr * 1.5),
            take_profit=raw_tp if raw_tp else (entry_price + atr * 3.0 if signal["signal"] == "buy" else entry_price - atr * 3.0),
            atr=atr,
        )

        # FIX: Use correct lot size calculation based on SL distance, not AI's lot_size_pct
        lot_size = calculate_lot_size_from_risk(
            account_balance=account_info["balance"],
            risk_pct=self.config.max_risk_per_trade_pct,
            entry_price=entry_price,
            stop_loss=stop_loss,
            market_type=market_type,
        )

        risk_result = await risk_manager.check_new_trade(
            user_id=self.user_id, market_type=market_type, symbol=symbol,
            lot_size=lot_size, entry_price=entry_price,
            stop_loss=stop_loss,
            account_balance=account_info["balance"],
            open_positions=open_positions, upcoming_news=upcoming_news,
        )

        if not risk_result.allowed:
            logger.info(f"Trade blocked {symbol}: {risk_result.reason}")
            return

        if risk_result.adjusted_lot_size:
            lot_size = risk_result.adjusted_lot_size

        broker_account = await self._get_broker_for_market(db, market_type)
        if not broker_account:
            return

        if self.config.trade_mode == TradeMode.LIVE:
            order_id = await adapter.place_order(
                symbol=symbol, side=signal["signal"], lot_size=lot_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
        else:
            order_id = f"PAPER_{symbol}_{datetime.utcnow().strftime('%H%M%S')}"

        if order_id:
            # Pass validated SL/TP into signal for _save_trade
            signal["stop_loss"] = stop_loss
            signal["take_profit"] = take_profit
            trade = await self._save_trade(db, broker_account.id, strategy.id, symbol, market_type, signal, entry_price, lot_size, order_id)
            await self.notifier.send_trade_opened(trade, signal)
            await publish(f"trades:{self.user_id}", {"event": "trade_opened", "symbol": symbol, "side": signal["signal"], "price": entry_price})

    # ── Position Sync ──

    async def _sync_open_positions(self):
        async with AsyncSessionLocal() as db:
            open_trades = await self._get_open_positions(db)
            for trade in open_trades:
                adapter = self.adapters.get(trade.market_type.value)
                if not adapter:
                    continue
                try:
                    broker_positions = await adapter.get_open_orders()
                    broker_ids = {p.order_id for p in broker_positions}
                    broker_pos = next((p for p in broker_positions if p.order_id == trade.broker_order_id), None)

                    if trade.broker_order_id not in broker_ids:
                        pnl = broker_pos.pnl if broker_pos else 0
                        await self._close_trade(db, trade, pnl, "bot")
                    elif broker_pos:
                        trade.pnl = broker_pos.pnl
                        await db.commit()
                except Exception as e:
                    logger.error(f"Sync error {trade.symbol}: {e}")

    # ── DB Operations ──

    async def _get_active_strategies(self, db: AsyncSession) -> list:
        r = await db.execute(select(Strategy).where(Strategy.user_id == self.user_id, Strategy.is_active == True).order_by(Strategy.priority.desc()))
        return r.scalars().all()

    async def _get_open_positions(self, db: AsyncSession) -> list:
        r = await db.execute(select(Trade).where(Trade.user_id == self.user_id, Trade.status == OrderStatus.OPEN))
        return r.scalars().all()

    async def _get_broker_for_market(self, db: AsyncSession, market_type: str):
        r = await db.execute(select(BrokerAccount).where(BrokerAccount.user_id == self.user_id, BrokerAccount.market_type == market_type, BrokerAccount.is_active == True, BrokerAccount.is_connected == True).limit(1))
        return r.scalar_one_or_none()

    async def _save_trade(self, db, broker_id, strategy_id, symbol, market_type, signal, entry_price, lot_size, order_id) -> Trade:
        trade = Trade(
            user_id=self.user_id, broker_id=broker_id, strategy_id=strategy_id,
            symbol=symbol, market_type=market_type,
            side=OrderSide.BUY if signal["signal"] == "buy" else OrderSide.SELL,
            status=OrderStatus.OPEN, trade_mode=self.config.trade_mode,
            entry_price=entry_price, lot_size=lot_size,
            stop_loss=signal.get("stop_loss"), take_profit=signal.get("take_profit"),
            ai_reasoning=signal.get("reasoning"), ai_confidence=signal.get("confidence"),
            signals_used=signal.get("key_factors", []),
            broker_order_id=order_id, currency=settings.BASE_CURRENCY,
            opened_at=datetime.utcnow(),
        )
        db.add(trade)
        await db.execute(update(Strategy).where(Strategy.id == strategy_id).values(total_trades=Strategy.total_trades + 1))
        await db.execute(update(BotConfig).where(BotConfig.user_id == self.user_id).values(daily_trades=BotConfig.daily_trades + 1))
        await db.commit()
        await db.refresh(trade)
        return trade

    async def _close_trade(self, db, trade, pnl, closed_by):
        trade.status = OrderStatus.CLOSED
        trade.pnl = round(pnl, 2)
        trade.closed_at = datetime.utcnow()
        trade.closed_by = closed_by
        if pnl < 0:
            await db.execute(update(BotConfig).where(BotConfig.user_id == self.user_id).values(daily_loss=BotConfig.daily_loss + abs(pnl)))
        await db.commit()
        await self.notifier.send_trade_closed(trade)
        await publish(f"trades:{self.user_id}", {"event": "trade_closed", "trade_id": trade.id, "symbol": trade.symbol, "pnl": pnl, "closed_by": closed_by})

    async def _log_signal(self, db, symbol, market_type, signal):
        log = AISignalLog(
            user_id=self.user_id, symbol=symbol, market_type=market_type,
            signal=signal["signal"], confidence=signal.get("confidence", 0),
            reasoning=signal.get("reasoning"), indicators=signal.get("key_factors", []),
            acted_on=signal["signal"] != "hold" and signal.get("confidence", 0) >= 0.65,
            created_at=datetime.utcnow(),
        )
        db.add(log)
        await db.commit()
        await publish(f"signals:{self.user_id}", {"symbol": symbol, "signal": signal["signal"], "confidence": signal.get("confidence", 0)})

    async def _health_check(self):
        async with AsyncSessionLocal() as db:
            open_trades = await self._get_open_positions(db)
            daily_pnl = -(self.config.daily_loss or 0.0)
            log = BotHealthLog(
                user_id=self.user_id, status="healthy",
                open_positions=len(open_trades),
                daily_pnl=daily_pnl,
                checked_at=datetime.utcnow(),
            )
            db.add(log)
            await db.commit()
        await publish(f"health:{self.user_id}", {
            "status": "healthy",
            "open_positions": len(open_trades),
            "daily_pnl": daily_pnl,
            "checked_at": datetime.utcnow().isoformat(),
        })

    async def _connect_brokers(self):
        async with AsyncSessionLocal() as db:
            r = await db.execute(select(BrokerAccount).where(BrokerAccount.user_id == self.user_id, BrokerAccount.is_active == True))
            accounts = r.scalars().all()
        for account in accounts:
            try:
                adapter = get_broker_adapter(account)
                if await adapter.connect():
                    self.adapters[account.market_type.value] = adapter
                    logger.info(f"Connected: {account.broker_type} ({account.market_type.value})")
            except Exception as e:
                logger.error(f"Broker error {account.broker_type}: {e}")

    async def _disconnect_brokers(self):
        for adapter in self.adapters.values():
            try:
                await adapter.disconnect()
            except Exception:
                pass
        self.adapters.clear()

    async def _get_consolidated_account(self) -> dict:
        balance, equity = 0.0, 0.0
        for adapter in self.adapters.values():
            try:
                info = await adapter.get_account_info()
                balance += info.balance
                equity += info.equity
            except Exception:
                pass
        return {"balance": balance, "equity": equity}

    def _get_symbols(self, strategy, market_type):
        """Get symbols to trade - from strategy config or adapter watchlist"""
        if strategy.symbols:
            return strategy.symbols
        
        # For Capital.com, try to get symbols from watchlist
        adapter = self.adapters.get(market_type)
        if adapter and hasattr(adapter, 'get_watchlist_symbols'):
            try:
                # This will be cached in the adapter
                symbols = adapter.get_cached_watchlist_symbols()
                if symbols:
                    return symbols
            except Exception as e:
                logger.warning(f"Failed to get watchlist symbols: {e}")
        
        # Fallback to default symbols
        return {
            "forex":     ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"],
            "crypto":    ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "commodity": ["XAUUSD", "XAGUSD", "USOIL"],
            "stock":     ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
            "index":     ["US500", "NAS100", "US30", "GER40"],
        }.get(market_type, [])
