from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from datetime import datetime, timedelta

from db.database import get_db
from db.models import Trade, BotConfig, OrderStatus, AISignalLog, User, BrokerAccount
from api.auth import get_current_user
from db.redis_client import get_bot_state
from data.calendar import calendar_client

router = APIRouter()


@router.get("/summary")
async def get_summary(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Bot config
    config_r = await db.execute(select(BotConfig).where(BotConfig.user_id == user.id))
    config = config_r.scalar_one_or_none()
    bot_state = await get_bot_state(user.id) or {}

    # Aktif broker — balance buradan
    broker_r = await db.execute(
        select(BrokerAccount).where(
            BrokerAccount.user_id == user.id,
            BrokerAccount.is_active == True,
        )
    )
    broker = broker_r.scalars().first()

    # Open positions
    open_r = await db.execute(
        select(Trade).where(
            Trade.user_id == user.id,
            Trade.status == OrderStatus.OPEN,
        )
    )
    open_trades = open_r.scalars().all()

    # Today's closed trades
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_r = await db.execute(
        select(Trade).where(
            Trade.user_id == user.id,
            Trade.status == OrderStatus.CLOSED,
            Trade.closed_at >= today_start,
        )
    )
    today_trades = today_r.scalars().all()

    # All-time closed trades
    all_r = await db.execute(
        select(Trade).where(
            Trade.user_id == user.id,
            Trade.status == OrderStatus.CLOSED,
        )
    )
    all_trades = all_r.scalars().all()

    # Recent AI signals
    sig_r = await db.execute(
        select(AISignalLog)
        .where(AISignalLog.user_id == user.id)
        .order_by(desc(AISignalLog.created_at))
        .limit(10)
    )
    signals = sig_r.scalars().all()

    # Calendar
    try:
        events = await calendar_client.get_calendar(hours_ahead=8, impact_filter=["high", "medium"])
    except Exception:
        events = []

    # PnL hesapları
    all_pnls    = [t.pnl for t in all_trades if t.pnl is not None]
    today_pnls  = [t.pnl for t in today_trades if t.pnl is not None]
    unrealized  = sum(t.pnl for t in open_trades if t.pnl is not None)
    winning     = [p for p in all_pnls if p > 0]

    return {
        "bot": {
            "status":             config.status.value if config else "stopped",
            "trade_mode":         config.trade_mode.value if config else "paper",
            "daily_loss":         config.daily_loss if config else 0,
            "daily_trades":       config.daily_trades if config else 0,
            "max_positions":      config.max_positions if config else 25,
            "max_daily_loss_pct": config.max_daily_loss_pct if config else 5.0,
            "max_risk_per_trade": config.max_risk_per_trade_pct if config else 1.0,
        },
        "account": {
            "balance":      broker.balance if broker else 0,
            "equity":       broker.equity if broker else 0,
            "currency":     broker.currency if broker else "EUR",
            "last_sync":    broker.last_sync.isoformat() if broker and broker.last_sync else None,
            "is_connected": broker.is_connected if broker else False,
        },
        "positions": {
            "open_count":    len(open_trades),
            "max_positions": config.max_positions if config else 25,
            "unrealized_pnl": round(unrealized, 2),
            "trades": [_serialize_trade_brief(t) for t in open_trades],
        },
        "today": {
            "trades": len(today_trades),
            "pnl":    round(sum(today_pnls), 2),
            "wins":   len([p for p in today_pnls if p > 0]),
            "losses": len([p for p in today_pnls if p < 0]),
        },
        "all_time": {
            "trades":   len(all_trades),
            "pnl":      round(sum(all_pnls), 2),
            "win_rate": round(len(winning) / max(len(all_pnls), 1) * 100, 2),
        },
        "signals": [
            {
                "symbol":     s.symbol,
                "signal":     s.signal,
                "confidence": s.confidence,
                "acted_on":   s.acted_on,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in signals
        ],
        "calendar": events[:6],
    }


@router.get("/equity-curve")
async def get_equity_curve(
    days: int = 30,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    r = await db.execute(
        select(Trade).where(
            Trade.user_id == user.id,
            Trade.status == OrderStatus.CLOSED,
            Trade.closed_at >= since,
            Trade.pnl.isnot(None),
        ).order_by(Trade.closed_at)
    )
    trades = r.scalars().all()

    curve      = []
    cumulative = 0
    for t in trades:
        cumulative += t.pnl or 0
        curve.append({
            "date":       t.closed_at.strftime("%Y-%m-%d") if t.closed_at else "",
            "pnl":        round(t.pnl or 0, 2),
            "cumulative": round(cumulative, 2),
        })
    return curve


@router.get("/market-breakdown")
async def get_market_breakdown(
    days: int = 30,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    r = await db.execute(
        select(Trade).where(
            Trade.user_id == user.id,
            Trade.status == OrderStatus.CLOSED,
            Trade.pnl.isnot(None),
            Trade.closed_at >= since,
        )
    )
    trades = r.scalars().all()

    breakdown = {}
    for t in trades:
        m = t.market_type.value
        if m not in breakdown:
            breakdown[m] = {"trades": 0, "pnl": 0, "wins": 0}
        breakdown[m]["trades"] += 1
        breakdown[m]["pnl"]     = round(breakdown[m]["pnl"] + (t.pnl or 0), 2)
        if (t.pnl or 0) > 0:
            breakdown[m]["wins"] += 1

    for m in breakdown:
        t = breakdown[m]["trades"]
        breakdown[m]["win_rate"] = round(breakdown[m]["wins"] / t * 100, 2) if t else 0

    return breakdown


def _serialize_trade_brief(t: Trade) -> dict:
    return {
        "id":           t.id,
        "symbol":       t.symbol,
        "side":         t.side.value,
        "entry_price":  t.entry_price,
        "current_price": t.exit_price,
        "lot_size":     t.lot_size,
        "pnl":          t.pnl,
        "stop_loss":    t.stop_loss,
        "take_profit":  t.take_profit,
        "ai_confidence": t.ai_confidence,
        "opened_at":    t.opened_at.isoformat() if t.opened_at else None,
    }