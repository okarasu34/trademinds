"""
Scheduler — bot job'larını zamanlar.

İki job:
  1. bot_scan: Her 5 dakikada bir bot.scan() çağırır (sinyal + emir pipeline'ı)
  2. balance_sync: Her 5 dakikada bir aktif broker'ların balance'ını DB'ye senkronlar
"""
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from sqlalchemy import select

from db.database import AsyncSessionLocal
from db.models import BrokerAccount
from brokers.capital_adapter import CapitalAdapter
from bot.trading_bot import bot_instance


scheduler = AsyncIOScheduler(timezone="UTC")


async def bot_scan_job():
    """Her 5 dakikada bir: sinyal üret + emir pipeline'ı."""
    try:
        await bot_instance.scan()
    except Exception as e:
        logger.exception(f"bot_scan_job failed: {e}")


async def balance_sync_job():
    """Her 5 dakikada bir: tüm aktif broker'ların canlı balance'ını DB'ye yaz."""
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(
                select(BrokerAccount).where(BrokerAccount.is_active == True)
            )
            brokers = result.scalars().all()

            for broker in brokers:
                if broker.broker_type != "capitalcom":
                    continue  # şimdilik sadece Capital.com

                try:
                    adapter = CapitalAdapter(broker)
                    if not await adapter.connect():
                        logger.warning(f"[balance_sync] {broker.name}: connect failed")
                        continue

                    info = await adapter.get_account_info()
                    broker.balance = info.balance
                    broker.equity = info.equity
                    broker.margin_used = info.margin_used
                    broker.currency = info.currency
                    broker.last_sync = datetime.utcnow()
                    broker.is_connected = True

                    await db.commit()
                    logger.info(
                        f"[balance_sync] {broker.name}: {info.balance:.2f} {info.currency}"
                    )

                    await adapter.disconnect()

                except Exception as e:
                    logger.exception(f"[balance_sync] {broker.name} error: {e}")
                    await db.rollback()

        except Exception as e:
            logger.exception(f"balance_sync_job failed: {e}")


def setup_scheduler():
    """Job'ları kaydet, scheduler'ı başlat."""
    # Bot scan: her 5 dakikada bir
    scheduler.add_job(
        bot_scan_job,
        IntervalTrigger(minutes=5),
        id="bot_scan",
        replace_existing=True,
        max_instances=1,  # paralel çalışma yasak (duplicate koruması)
        coalesce=True,    # gecikmiş job'lar birleşsin, üst üste binmesin
    )

    # Balance sync: her 5 dakikada bir (bot scan'den 30 sn sonra başlasın diye sapma yok ama coalesce var)
    scheduler.add_job(
        balance_sync_job,
        IntervalTrigger(minutes=5),
        id="balance_sync",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()
    logger.info(">>> Scheduler started: bot_scan + balance_sync (every 5 min)")