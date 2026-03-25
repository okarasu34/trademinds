from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from datetime import datetime
from sqlalchemy import select, update

from db.database import AsyncSessionLocal
from db.models import BotConfig, BotStatus, BotHealthLog, User, Trade, OrderStatus
from db.redis_client import get_redis, publish
from data.calendar import calendar_client

scheduler = AsyncIOScheduler(timezone="UTC")


def setup_scheduler():
    """Register all scheduled jobs."""

    # ─── Every 60s: bot health check for all running bots ───
    scheduler.add_job(
        health_check_all_bots,
        IntervalTrigger(seconds=60),
        id="health_check",
        replace_existing=True,
    )

    # ─── Every 5min: refresh economic calendar cache ───
    scheduler.add_job(
        refresh_calendar,
        IntervalTrigger(minutes=5),
        id="calendar_refresh",
        replace_existing=True,
    )

    # ─── 00:01 UTC daily: reset daily P&L counters ───
    scheduler.add_job(
        reset_daily_stats,
        CronTrigger(hour=0, minute=1),
        id="daily_reset",
        replace_existing=True,
    )

    # ─── Every Sunday 02:00 UTC: database backup ───
    scheduler.add_job(
        backup_database,
        CronTrigger(day_of_week="sun", hour=2, minute=0),
        id="weekly_backup",
        replace_existing=True,
    )

    # ─── Every 30min: sync broker account balances ───
    scheduler.add_job(
        sync_broker_balances,
        IntervalTrigger(minutes=30),
        id="broker_sync",
        replace_existing=True,
    )

    # ─── Every 1h: update strategy performance stats ───
    scheduler.add_job(
        update_strategy_stats,
        IntervalTrigger(hours=1),
        id="strategy_stats",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Scheduler started with all jobs")


async def health_check_all_bots():
    """Check health of all running bots, log status."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(BotConfig).where(BotConfig.status == BotStatus.RUNNING)
        )
        configs = result.scalars().all()

        for config in configs:
            try:
                open_trades = await db.execute(
                    select(Trade).where(
                        Trade.user_id == config.user_id,
                        Trade.status == OrderStatus.OPEN,
                    )
                )
                open_count = len(open_trades.scalars().all())

                log = BotHealthLog(
                    user_id=config.user_id,
                    status="healthy",
                    open_positions=open_count,
                    daily_pnl=-(config.daily_loss or 0),
                    checked_at=datetime.utcnow(),
                )
                db.add(log)

                # Publish health to WebSocket
                await publish(f"health:{config.user_id}", {
                    "status": "healthy",
                    "open_positions": open_count,
                    "daily_pnl": -(config.daily_loss or 0),
                    "checked_at": datetime.utcnow().isoformat(),
                })

            except Exception as e:
                logger.error(f"Health check failed for {config.user_id}: {e}")
                log = BotHealthLog(
                    user_id=config.user_id,
                    status="error",
                    message=str(e),
                    checked_at=datetime.utcnow(),
                )
                db.add(log)

        await db.commit()


async def reset_daily_stats():
    """Reset daily loss and trade counters for all users at midnight UTC."""
    logger.info("Running daily stats reset")
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(BotConfig).values(
                daily_loss=0.0,
                daily_trades=0,
                daily_reset_at=datetime.utcnow(),
            )
        )
        await db.commit()
    logger.info("Daily stats reset complete")


async def refresh_calendar():
    """Pre-fetch and cache economic calendar."""
    try:
        await calendar_client.get_calendar(hours_ahead=24, impact_filter=["high", "medium"])
    except Exception as e:
        logger.warning(f"Calendar refresh failed: {e}")


async def backup_database():
    """
    Weekly PostgreSQL backup to local file.
    In production, also upload to S3/Vultr Object Storage.
    """
    import subprocess
    import os

    backup_dir = "/var/backups/trademinds"
    os.makedirs(backup_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{backup_dir}/trademinds_{timestamp}.sql.gz"

    try:
        result = subprocess.run(
            f"pg_dump trademinds | gzip > {filename}",
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info(f"Database backup created: {filename}")
            # Keep only last 4 backups
            backups = sorted([
                f for f in os.listdir(backup_dir) if f.endswith(".sql.gz")
            ])
            for old in backups[:-4]:
                os.remove(os.path.join(backup_dir, old))
        else:
            logger.error(f"Backup failed: {result.stderr}")
    except Exception as e:
        logger.error(f"Backup error: {e}")


async def sync_broker_balances():
    """Sync account balances from all active brokers."""
    from brokers.base_adapter import get_broker_adapter
    from db.models import BrokerAccount

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(BrokerAccount).where(BrokerAccount.is_active == True)
        )
        brokers = result.scalars().all()

        for broker in brokers:
            try:
                adapter = get_broker_adapter(broker)
                await adapter.connect()
                info = await adapter.get_account_info()
                broker.balance = info.balance
                broker.equity = info.equity
                broker.margin_used = info.margin_used
                broker.is_connected = True
                broker.last_sync = datetime.utcnow()
                await adapter.disconnect()
            except Exception as e:
                broker.is_connected = False
                logger.warning(f"Broker sync failed {broker.name}: {e}")

        await db.commit()


async def update_strategy_stats():
    """Recalculate performance stats for all strategies."""
    from db.models import Strategy
    from sqlalchemy import func

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Strategy))
        strategies = result.scalars().all()

        for strategy in strategies:
            trades_r = await db.execute(
                select(Trade).where(
                    Trade.strategy_id == strategy.id,
                    Trade.status == OrderStatus.CLOSED,
                )
            )
            trades = trades_r.scalars().all()
            if not trades:
                continue

            pnls = [t.pnl or 0 for t in trades]
            wins = [p for p in pnls if p > 0]
            strategy.total_trades = len(trades)
            strategy.win_rate = round(len(wins) / len(trades) * 100, 2)
            strategy.total_pnl = round(sum(pnls), 2)
            strategy.avg_pnl_per_trade = round(sum(pnls) / len(trades), 2)

            # Drawdown
            equity, peak, max_dd = 0, 0, 0
            for p in pnls:
                equity += p
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / max(peak, 1) * 100
                if dd > max_dd:
                    max_dd = dd
            strategy.max_drawdown = round(max_dd, 2)

        await db.commit()
