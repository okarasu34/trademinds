"""
Scheduler — bot job'larını zamanlar.

Job'lar:
  1. bot_scan: Her 5 dakikada bir sinyal + emir pipeline'ı
  2. balance_sync: Her 5 dakikada bir balance DB'ye yaz
  3. trade_sync: Her 5 dakikada bir kapanan trade'leri DB'ye yaz + PnL kaydet
"""
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from sqlalchemy import select

from db.database import AsyncSessionLocal
from db.models import BrokerAccount, Trade, OrderStatus
from brokers.capital_adapter import CapitalAdapter
from bot.trading_bot import bot_instance


scheduler = AsyncIOScheduler(timezone="UTC")


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
                    broker.balance    = info.balance
                    broker.equity     = info.equity
                    broker.margin_used = info.margin_used
                    broker.currency   = info.currency
                    broker.last_sync  = datetime.utcnow()
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
    """Capital.com'da kapanan pozisyonları DB'ye yaz, PnL'i kaydet."""
    async with AsyncSessionLocal() as db:
        try:
            # Aktif broker'ları al
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
                        continue

                    # Capital.com'dan canlı pozisyonları al
                    live_positions = await adapter.get_open_orders()
                    live_order_ids = {p.order_id for p in live_positions}
                    live_pnl_map   = {p.order_id: p.pnl for p in live_positions}
                    live_price_map = {p.order_id: p.current_price for p in live_positions}

                    # DB'de OPEN olan trade'leri al
                    open_trades_result = await db.execute(
                        select(Trade).where(
                            Trade.broker_id == broker.id,
                            Trade.status == OrderStatus.OPEN,
                        )
                    )
                    open_trades = open_trades_result.scalars().all()

                    closed_count = 0
                    for trade in open_trades:
                        if trade.broker_order_id not in live_order_ids:
                            # Capital.com'da artık yok — kapanmış
                            # Kapanış fiyatını ve PnL'i hesapla
                            # Capital.com kapanmış pozisyonlar için ayrı endpoint lazım
                            # Şimdilik entry - exit tahmin ediyoruz
                            trade.status    = OrderStatus.CLOSED
                            trade.closed_at = datetime.utcnow()
                            trade.closed_by = "bot"
                            # PnL: live_pnl_map'te yoksa 0
                            trade.pnl = live_pnl_map.get(trade.broker_order_id, None)
                            closed_count += 1

                    if closed_count > 0:
                        await db.commit()
                        logger.info(f"[trade_sync] {closed_count} trade CLOSED işaretlendi")

                    # Açık trade'lerin unrealized PnL'ini güncelle
                    updated = 0
                    for trade in open_trades:
                        if trade.broker_order_id in live_pnl_map:
                            trade.pnl         = live_pnl_map[trade.broker_order_id]
                            trade.exit_price  = live_price_map.get(trade.broker_order_id)
                            updated += 1

                    if updated > 0:
                        await db.commit()

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
        IntervalTrigger(minutes=5),
        id="balance_sync",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        trade_sync_job,
        IntervalTrigger(minutes=5),
        id="trade_sync",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()
    logger.info(">>> Scheduler started: bot_scan + balance_sync + trade_sync (every 5 min)")