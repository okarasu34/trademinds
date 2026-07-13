from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from db.database import get_db
from db.models import BotConfig, BotStatus, TradeMode, BotHealthLog, BrokerAccount, User
from db.redis_client import get_bot_state
from api.auth import get_current_user

router = APIRouter()


class BotConfigUpdate(BaseModel):
    max_positions: Optional[int] = None
    max_daily_loss_pct: Optional[float] = None
    max_risk_per_trade_pct: Optional[float] = None
    news_pause_minutes: Optional[int] = None
    pause_on_high_impact_news: Optional[bool] = None
    market_limits: Optional[dict] = None


class TradeModeUpdate(BaseModel):
    mode: str


@router.get("/status")
async def get_bot_status(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Bot config
    result = await db.execute(select(BotConfig).where(BotConfig.user_id == user.id))
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="Bot config not found")

    # Balance — aktif broker'dan çek
    broker_result = await db.execute(
        select(BrokerAccount).where(
            BrokerAccount.user_id == user.id,
            BrokerAccount.is_active == True,
        )
    )
    broker = broker_result.scalars().first()

    redis_state = await get_bot_state(user.id) or {}

    return {
        "status": config.status.value,
        "trade_mode": config.trade_mode.value,
        "balance": broker.balance if broker else 0,
        "equity": broker.equity if broker else 0,
        "currency": broker.currency if broker else "EUR",
        "last_sync": broker.last_sync.isoformat() if broker and broker.last_sync else None,
        "is_connected": broker.is_connected if broker else False,
        "redis_state": redis_state,
        "max_positions": config.max_positions,
        "max_daily_loss_pct": config.max_daily_loss_pct,
        "max_risk_per_trade_pct": config.max_risk_per_trade_pct,
        "pause_on_high_impact_news": config.pause_on_high_impact_news,
        "news_pause_minutes": config.news_pause_minutes,
        "market_limits": config.market_limits,
        "daily_loss": config.daily_loss,
        "daily_trades": config.daily_trades,
    }


@router.post("/start")
async def start_bot(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(BotConfig).where(BotConfig.user_id == user.id))
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="Bot config not found")

    config.status = BotStatus.RUNNING
    await db.commit()

    return {"message": "Bot started", "mode": config.trade_mode.value}


@router.post("/stop")
async def stop_bot(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(BotConfig).where(BotConfig.user_id == user.id))
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="Bot config not found")

    config.status = BotStatus.STOPPED
    await db.commit()

    return {"message": "Bot stopped"}


@router.post("/pause")
async def pause_bot(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(BotConfig).where(BotConfig.user_id == user.id))
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="Bot config not found")

    config.status = BotStatus.PAUSED
    await db.commit()

    return {"message": "Bot paused"}


@router.put("/config")
async def update_config(
    body: BotConfigUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(BotConfig).where(BotConfig.user_id == user.id))
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="Not found")

    update_data = body.dict(exclude_none=True)
    for key, value in update_data.items():
        setattr(config, key, value)

    await db.commit()
    return {"message": "Config updated", "config": update_data}


@router.put("/mode")
async def set_trade_mode(
    body: TradeModeUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(BotConfig).where(BotConfig.user_id == user.id))
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="Not found")

    try:
        config.trade_mode = TradeMode(body.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {body.mode}")

    await db.commit()
    return {"message": f"Mode set to {body.mode}"}


@router.get("/health-logs")
async def get_health_logs(
    limit: int = 50,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(BotHealthLog)
        .where(BotHealthLog.user_id == user.id)
        .order_by(BotHealthLog.checked_at.desc())
        .limit(limit)
    )
    logs = result.scalars().all()
    return [
        {
            "status": l.status,
            "message": l.message,
            "open_positions": l.open_positions,
            "daily_pnl": l.daily_pnl,
            "checked_at": l.checked_at.isoformat() if l.checked_at else None,
        }
        for l in logs
    ]


@router.get("/symbols")
async def get_watchlist_symbols(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Returns watchlist symbols from active broker."""
    from brokers.capital_adapter import CapitalAdapter
    broker_result = await db.execute(
        select(BrokerAccount).where(
            BrokerAccount.user_id == user.id,
            BrokerAccount.is_active == True,
        )
    )
    broker = broker_result.scalars().first()
    if not broker:
        return {"symbols": []}
    try:
        adapter = CapitalAdapter(broker)
        await adapter.connect()
        await adapter.load_trademinds_watchlist()
        symbols = adapter._cached_watchlist_symbols or []
        await adapter.disconnect()
        return {"symbols": sorted(symbols)}
    except Exception as e:
        return {"symbols": []}
async def reset_daily_stats(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(BotConfig).where(BotConfig.user_id == user.id))
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="Not found")

    config.daily_loss = 0.0
    config.daily_trades = 0
    config.daily_reset_at = datetime.utcnow()
    await db.commit()
    return {"message": "Daily stats reset"}