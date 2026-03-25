from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta

from db.database import get_db
from db.models import Trade, OrderStatus, User
from api.auth import get_current_user
from api.trades import _serialize_trade
from reports.generator import generate_pdf_report, generate_excel_report

router = APIRouter()


def _build_summary(trades, currency="USD"):
    if not trades:
        return {}
    pnls = [t.pnl or 0 for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 2),
        "total_pnl": round(sum(pnls), 2),
        "best_trade": round(max(pnls), 2) if pnls else 0,
        "worst_trade": round(min(pnls), 2) if pnls else 0,
        "profit_factor": round(sum(wins) / max(abs(sum(losses)), 0.01), 2),
        "currency": currency,
        "initial_balance": 10000,
    }


async def _fetch_closed_trades(user_id, period, db):
    q = select(Trade).where(Trade.user_id == user_id, Trade.status == OrderStatus.CLOSED)
    now = datetime.utcnow()
    periods = {
        "today": now.replace(hour=0, minute=0, second=0),
        "week": now - timedelta(days=7),
        "month": now - timedelta(days=30),
        "year": now - timedelta(days=365),
    }
    if period in periods:
        q = q.where(Trade.closed_at >= periods[period])
    result = await db.execute(q.order_by(Trade.opened_at))
    return result.scalars().all()


@router.get("/pdf")
async def download_pdf(
    period: str = Query("month", enum=["today", "week", "month", "year", "all"]),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    trades = await _fetch_closed_trades(user.id, period, db)
    summary = _build_summary(trades)
    pdf = generate_pdf_report([_serialize_trade(t) for t in trades], summary, period, user.email)
    return Response(content=pdf, media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=trademinds_{period}.pdf"})


@router.get("/excel")
async def download_excel(
    period: str = Query("month", enum=["today", "week", "month", "year", "all"]),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    trades = await _fetch_closed_trades(user.id, period, db)
    summary = _build_summary(trades)
    excel = generate_excel_report([_serialize_trade(t) for t in trades], summary, period)
    return Response(content=excel,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=trademinds_{period}.xlsx"})
