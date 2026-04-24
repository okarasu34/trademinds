from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
from bot.trading_bot import TradingBot

# Bot motorunu burada başlatıyoruz
scheduler = AsyncIOScheduler(timezone="UTC")
bot_instance = TradingBot()

def setup_scheduler():
    """Tüm görevleri kaydet ve botu ateşle."""

    # ─── ANA ANALİZ VE İŞLEM MOTORU (Her 60 Saniyede Bir) ───
    # Bu satır botun piyasayı taramasını sağlar
    scheduler.add_job(
        bot_instance._scan_and_execute,
        IntervalTrigger(seconds=60),
        id="main_trading_engine",
        replace_existing=True,
    )

    # ─── Bot Sağlık Kontrolü ───
    from bot.scheduler import health_check_all_bots
    scheduler.add_job(
        health_check_all_bots,
        IntervalTrigger(seconds=60),
        id="health_check",
        replace_existing=True,
    )

    # ─── Ekonomik Takvim Yenileme ───
    from bot.scheduler import refresh_calendar
    scheduler.add_job(
        refresh_calendar,
        IntervalTrigger(minutes=5),
        id="calendar_refresh",
        replace_existing=True,
    )

    # ─── Bakiye Senkronizasyonu ───
    from bot.scheduler import sync_broker_balances
    scheduler.add_job(
        sync_broker_balances,
        IntervalTrigger(minutes=30),
        id="broker_sync",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(">>> TRADING ENGINE VE SCHEDULER BAŞLATILDI <<<")

# Mevcut fonksiyonlarını aşağıya ekliyoruz (Dosya bütünlüğü için)
async def health_check_all_bots():
    from bot.scheduler_tasks import health_check_all_bots as hc
    await hc()

async def refresh_calendar():
    from data.calendar import calendar_client
    try:
        await calendar_client.get_calendar(hours_ahead=24, impact_filter=["high", "medium"])
    except Exception as e:
        logger.warning(f"Calendar refresh failed: {e}")

async def sync_broker_balances():
    # Mevcut fonksiyonun içeriği buraya gelecek veya import edilecek
    pass
EOF