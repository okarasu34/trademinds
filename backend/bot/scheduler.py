"""
Scheduler — bot job'larını zamanlar.

Job'lar:
  1. bot_scan: Her 5 dakikada bir sinyal + emir pipeline'ı
  2. balance_sync: Her 5 dakikada bir balance DB'ye yaz
  3. trade_sync: Her 5 dakikada bir Capital.com ↔ DB tam senkron
     - Capital.com'da açık ama DB'de olmayan → DB'ye ekle
     - DB'de OPEN ama Capital.com'da olmayan → CLOSED yap + PnL yaz
     - DB'deki açık pozisyonların PnL'ini güncelle
"""
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from sqlalchemy import select

from db.database import AsyncSessionLocal
from db.models import BrokerAccount, Trade, OrderStatus, MarketType, OrderSide, TradeMode
from brokers.capital_adapter import CapitalAdapter
from bot.trading_bot import bot_instance


scheduler = AsyncIOScheduler(timezone="UTC")


def _infer_market(symbol: str) -> MarketType:
    s = symbol.upper()
    if any(c in s for c in ("BTC","ETH","XRP","DOGE","SOL","ADA","AAVE","AVAX","LTC","SHIB","PEPE","TRX","HBAR","XLM","ALPHA","USDT")):
        return MarketType.CRYPTO
    if any(c in s for c in ("XAU","XAG","OIL","GAS","GOLD","SILVER","PLATINUM","PALLADIUM","COPPER","CORN","WHEAT","NATURALGAS","BRENT")):
        return MarketType.COMMODITY
    if any(c in s for c in ("SPX","NDX","DJI","DAX","FTSE","NKY","US100","US500","US30","DE40")):
        return MarketType.INDEX
    if len(s) == 6 and s.isalpha():
        return MarketType.FOREX
    return MarketType.STOCK


async def bot_scan_job():
    try:
        await bot_instance.scan()
    except Exception as e:
        logger.exception(f"bot_scan_job failed: {e}")


async def balance_sync_job():
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(
                select(BrokerAccount).where(BrokerAccount.is_active == True)
            )
            brokers = result.scalars().all()

            for broker in brokers:
                if broker.broker_type != "capitalcom":
                    continue
                try:
                    adapter = CapitalAdapter(broker)
                    if not await adapter.connect():
                        logger.warning(f"[balance_sync] {broker.name}: connect failed")
                        continue

                    info = await adapter.get_account_info()
                    broker.balance     = info.balance
                    broker.equity      = info.equity
                    broker.margin_used = info.margin_used
                    broker.currency    = info.currency
                    broker.last_sync   = datetime.utcnow()
                    broker.is_connected = True

                    await db.commit()
                    logger.info(f"[balance_sync] {broker.name}: {info.balance:.2f} {info.currency}")
                    await adapter.disconnect()

                except Exception as e:
                    logger.exception(f"[balance_sync] {broker.name} error: {e}")
                    await db.rollback()

        except Exception as e:
            logger.exception(f"balance_sync_job failed: {e}")


async def trade_sync_job():
    """Capital.com ↔ DB tam senkron:
    1. Capital.com'da açık ama DB'de olmayan → ekle
    2. DB'de OPEN ama Capital.com'da olmayan → CLOSED yap
    3. DB'deki OPEN trade'lerin PnL'ini güncelle
    """
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(
                select(BrokerAccount).where(BrokerAccount.is_active == True)
            )
            brokers = result.scalars().all()

            for broker in brokers:
                if broker.broker_type != "capitalcom":
                    continue

                try:
                    adapter = CapitalAdapter(broker)
                    if not await adapter.connect():
                        logger.warning(f"[trade_sync] {broker.name}: connect failed")
                        continue

                    # Capital.com'dan canlı pozisyonları al
                    live_positions = await adapter.get_open_orders()
                    live_map = {p.order_id: p for p in live_positions}

                    # Broker'ın user_id'sini al
                    user_result = await db.execute(
                        select(BrokerAccount.user_id).where(BrokerAccount.id == broker.id)
                    )
                    user_id = user_result.scalar()

                    # DB'deki OPEN trade'leri al
                    open_trades_result = await db.execute(
                        select(Trade).where(
                            Trade.broker_id == broker.id,
                            Trade.status == OrderStatus.OPEN,
                        )
                    )
                    open_trades = open_trades_result.scalars().all()
                    db_order_ids = {t.broker_order_id for t in open_trades if t.broker_order_id}

                    closed_count = 0
                    updated_count = 0

                    # Kapanan trade'ler varsa PnL'lerini transactions'dan çek
                    closing_trades = [
                        t for t in open_trades
                        if t.broker_order_id not in live_map
                    ]
                    
                    closed_txns = {}
                    if closing_trades:
                        closed_txns = await adapter.get_closed_transactions(hours_back=2)

                    # 1. DB'de OPEN ama Capital.com'da olmayan → CLOSED + PnL
                    for trade in closing_trades:
                        trade.status    = OrderStatus.CLOSED
                        trade.closed_at = datetime.utcnow()
                        trade.closed_by = "bot"
                        
                        # Transactions'dan PnL al
                        txn = closed_txns.get(trade.broker_order_id)
                        if txn:
                            trade.pnl = txn["pnl"]
                            logger.info(
                                f"[trade_sync] Closed: {trade.symbol} "
                                f"pnl={txn['pnl']}"
                            )
                        elif trade.pnl is not None:
                            # Son bilinen PnL'i koru (önceki sync'ten)
                            logger.info(
                                f"[trade_sync] Closed: {trade.symbol} "
                                f"pnl={trade.pnl} (last known)"
                            )
                        else:
                            logger.warning(
                                f"[trade_sync] Closed: {trade.symbol} "
                                f"pnl=UNKNOWN (no transaction found)"
                            )
                        closed_count += 1

                    # 2. DB'deki OPEN trade'lerin PnL'ini güncelle
                    for trade in open_trades:
                        if trade.broker_order_id in live_map:
                            live = live_map[trade.broker_order_id]
                            trade.pnl        = live.pnl
                            trade.exit_price = live.current_price
                            updated_count += 1

                    # 3. Capital.com'da açık ama DB'de olmayan → ekle
                    added_count = 0
                    for order_id, pos in live_map.items():
                        if order_id not in db_order_ids:
                            new_trade = Trade(
                                user_id         = user_id,
                                broker_id       = broker.id,
                                symbol          = pos.symbol,
                                market_type     = _infer_market(pos.symbol),
                                side            = OrderSide.BUY if pos.side == "buy" else OrderSide.SELL,
                                status          = OrderStatus.OPEN,
                                trade_mode      = TradeMode.LIVE,
                                entry_price     = pos.entry_price,
                                lot_size        = pos.lot_size,
                                pnl             = pos.pnl,
                                exit_price      = pos.current_price,
                                broker_order_id = order_id,
                                opened_at       = datetime.utcnow(),
                            )
                            db.add(new_trade)
                            added_count += 1
                            logger.info(f"[trade_sync] Added: {pos.symbol} {pos.side} pnl={pos.pnl}")

                    await db.commit()

                    if closed_count or added_count or updated_count:
                        logger.info(
                            f"[trade_sync] closed={closed_count} "
                            f"added={added_count} "
                            f"pnl_updated={updated_count}"
                        )

                    await adapter.disconnect()

                except Exception as e:
                    logger.exception(f"[trade_sync] {broker.name} error: {e}")
                    await db.rollback()

        except Exception as e:
            logger.exception(f"trade_sync_job failed: {e}")


def setup_scheduler():
    scheduler.add_job(
        bot_scan_job,
        IntervalTrigger(minutes=5),
        id="bot_scan",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        balance_sync_job,
        IntervalTrigger(minutes=5, start_date="2000-01-01 00:01:00"),
        id="balance_sync",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        trade_sync_job,
        IntervalTrigger(minutes=5, start_date="2000-01-01 00:02:00"),
        id="trade_sync",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()
    logger.info(">>> Scheduler started: bot_scan + balance_sync + trade_sync (every 5 min)")