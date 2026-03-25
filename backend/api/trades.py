from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta

from db.database import get_db
from db.models import Trade, OrderStatus, MarketType, User, BotConfig
from db.redis_client import publish
from api.auth import get_current_user
from notifications.notifier import Notifier

router = APIRouter()


@router.get("")
async def get_trades(
    status: Optional[str] = None,
    market_type: Optional[str] = None,
    symbol: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    page: int = 1,
    page_size: int = 50,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = select(Trade).where(Trade.user_id == user.id)

    if status:
        query = query.where(Trade.status == status)
    if market_type:
        query = query.where(Trade.market_type == market_type)
    if symbol:
        query = query.where(Trade.symbol == symbol)
    if from_date:
        query = query.where(Trade.opened_at >= from_date)
    if to_date:
        query = query.where(Trade.opened_at <= to_date)

    # Count total
    count_q = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_q)).scalar()

    # Paginate
    query = query.order_by(desc(Trade.opened_at)).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    trades = result.scalars().all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "trades": [_serialize_trade(t) for t in trades],
    }


@router.get("/open")
async def get_open_trades(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Trade).where(
            Trade.user_id == user.id,
            Trade.status == OrderStatus.OPEN,
        ).order_by(desc(Trade.opened_at))
    )
    trades = result.scalars().all()
    return [_serialize_trade(t) for t in trades]


@router.get("/stats")
async def get_trade_stats(
    period: str = "all",  # all, today, week, month, year
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    base_q = select(Trade).where(
        Trade.user_id == user.id,
        Trade.status == OrderStatus.CLOSED,
    )

    now = datetime.utcnow()
    if period == "today":
        base_q = base_q.where(Trade.closed_at >= now.replace(hour=0, minute=0, second=0))
    elif period == "week":
        base_q = base_q.where(Trade.closed_at >= now - timedelta(days=7))
    elif period == "month":
        base_q = base_q.where(Trade.closed_at >= now - timedelta(days=30))
    elif period == "year":
        base_q = base_q.where(Trade.closed_at >= now - timedelta(days=365))

    result = await db.execute(base_q)
    trades = result.scalars().all()

    if not trades:
        return _empty_stats()

    pnls = [t.pnl or 0 for t in trades]
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p < 0]
    total_pnl = sum(pnls)
    gross_profit = sum(winning)
    gross_loss = abs(sum(losing))

    # Drawdown calculation
    equity = 0
    peak = 0
    max_dd = 0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = (peak - equity) / max(peak, 1) * 100
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (simplified)
    import numpy as np
    pnl_arr = [t.pnl or 0 for t in trades]
    sharpe = (np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-10)) * (252 ** 0.5) if len(pnl_arr) > 1 else 0

    # Per market breakdown
    by_market = {}
    for t in trades:
        m = t.market_type.value
        if m not in by_market:
            by_market[m] = {"count": 0, "pnl": 0, "wins": 0}
        by_market[m]["count"] += 1
        by_market[m]["pnl"] += t.pnl or 0
        if (t.pnl or 0) > 0:
            by_market[m]["wins"] += 1

    return {
        "period": period,
        "total_trades": len(trades),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": round(len(winning) / len(trades) * 100, 2) if trades else 0,
        "total_pnl": round(total_pnl, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
        "best_trade": round(max(pnls), 2) if pnls else 0,
        "worst_trade": round(min(pnls), 2) if pnls else 0,
        "avg_pnl": round(total_pnl / len(trades), 2) if trades else 0,
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "by_market": by_market,
    }


@router.get("/{trade_id}")
async def get_trade(
    trade_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Trade).where(Trade.id == trade_id, Trade.user_id == user.id)
    )
    trade = result.scalar_one_or_none()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    return _serialize_trade(trade)


@router.post("/{trade_id}/close")
async def manual_close_trade(
    trade_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Only available user action — manually close an open position."""
    result = await db.execute(
        select(Trade).where(
            Trade.id == trade_id,
            Trade.user_id == user.id,
            Trade.status == OrderStatus.OPEN,
        )
    )
    trade = result.scalar_one_or_none()
    if not trade:
        raise HTTPException(status_code=404, detail="Open trade not found")

    # TODO: Call broker adapter to close the position at market
    # adapter = get_broker_adapter(trade.broker)
    # success = await adapter.close_order(trade.broker_order_id, trade.symbol)

    trade.status = OrderStatus.CLOSED
    trade.closed_at = datetime.utcnow()
    trade.closed_by = "manual"

    # Update daily loss in bot config
    result2 = await db.execute(select(BotConfig).where(BotConfig.user_id == user.id))
    config = result2.scalar_one_or_none()
    if config and trade.pnl and trade.pnl < 0:
        config.daily_loss = (config.daily_loss or 0) + abs(trade.pnl)

    await db.commit()

    # Notify
    notifier = Notifier(user.id)
    await notifier.send_trade_closed(trade)

    # Broadcast via WebSocket
    await publish(f"trades:{user.id}", {
        "event": "trade_closed",
        "trade_id": trade_id,
        "symbol": trade.symbol,
        "pnl": trade.pnl,
        "closed_by": "manual",
    })

    return {"message": "Position closed", "trade": _serialize_trade(trade)}


def _serialize_trade(t: Trade) -> dict:
    return {
        "id": t.id,
        "symbol": t.symbol,
        "market_type": t.market_type.value,
        "side": t.side.value,
        "status": t.status.value,
        "trade_mode": t.trade_mode.value,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "lot_size": t.lot_size,
        "stop_loss": t.stop_loss,
        "take_profit": t.take_profit,
        "pnl": t.pnl,
        "pnl_pct": t.pnl_pct,
        "commission": t.commission,
        "currency": t.currency,
        "ai_reasoning": t.ai_reasoning,
        "ai_confidence": t.ai_confidence,
        "signals_used": t.signals_used,
        "news_context": t.news_context,
        "opened_at": t.opened_at.isoformat() if t.opened_at else None,
        "closed_at": t.closed_at.isoformat() if t.closed_at else None,
        "closed_by": t.closed_by,
        "broker_order_id": t.broker_order_id,
    }


def _empty_stats() -> dict:
    return {
        "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
        "win_rate": 0, "total_pnl": 0, "gross_profit": 0, "gross_loss": 0,
        "profit_factor": 0, "best_trade": 0, "worst_trade": 0,
        "avg_pnl": 0, "max_drawdown_pct": 0, "sharpe_ratio": 0, "by_market": {},
    }
