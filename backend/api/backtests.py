from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

from db.database import get_db
from db.models import Backtest, Strategy, User
from api.auth import get_current_user
from bot.backtest_engine import BacktestEngine

router = APIRouter()


class BacktestCreate(BaseModel):
    name: str
    strategy_id: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    currency: str = "USD"


@router.get("")
async def list_backtests(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Backtest)
        .where(Backtest.user_id == user.id)
        .order_by(desc(Backtest.created_at))
    )
    backtests = result.scalars().all()
    return [_serialize(b) for b in backtests]


@router.post("", status_code=201)
async def create_backtest(
    body: BacktestCreate,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Validate strategy
    strat_r = await db.execute(
        select(Strategy).where(Strategy.id == body.strategy_id, Strategy.user_id == user.id)
    )
    strategy = strat_r.scalar_one_or_none()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Validate date range
    if body.end_date <= body.start_date:
        raise HTTPException(status_code=400, detail="end_date must be after start_date")
    days = (body.end_date - body.start_date).days
    if days < 7:
        raise HTTPException(status_code=400, detail="Minimum backtest period is 7 days")

    backtest = Backtest(
        user_id=user.id,
        strategy_id=body.strategy_id,
        name=body.name,
        symbol=body.symbol,
        timeframe=body.timeframe,
        start_date=body.start_date,
        end_date=body.end_date,
        initial_balance=body.initial_balance,
        currency=body.currency,
        status="pending",
    )
    db.add(backtest)
    await db.commit()
    await db.refresh(backtest)

    # Run in background
    background_tasks.add_task(_run_backtest, backtest.id, strategy)

    return {"id": backtest.id, "status": "pending", "message": "Backtest queued"}


@router.get("/{backtest_id}")
async def get_backtest(
    backtest_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Backtest).where(Backtest.id == backtest_id, Backtest.user_id == user.id)
    )
    b = result.scalar_one_or_none()
    if not b:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return _serialize(b)


@router.delete("/{backtest_id}", status_code=204)
async def delete_backtest(
    backtest_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Backtest).where(Backtest.id == backtest_id, Backtest.user_id == user.id)
    )
    b = result.scalar_one_or_none()
    if not b:
        raise HTTPException(status_code=404, detail="Not found")
    await db.delete(b)
    await db.commit()


async def _run_backtest(backtest_id: str, strategy: Strategy):
    """Background task to run backtest and save results."""
    from db.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Backtest).where(Backtest.id == backtest_id))
        backtest = result.scalar_one_or_none()
        if not backtest:
            return

        backtest.status = "running"
        backtest.started_at = datetime.utcnow()
        await db.commit()

        try:
            engine = BacktestEngine(backtest, strategy)
            results = await engine.run()

            if "error" in results:
                backtest.status = "failed"
                backtest.error_message = results["error"]
            else:
                backtest.status = "completed"
                backtest.final_balance = results["final_balance"]
                backtest.total_return_pct = results["total_return_pct"]
                backtest.total_trades = results["total_trades"]
                backtest.winning_trades = results["winning_trades"]
                backtest.losing_trades = results["losing_trades"]
                backtest.win_rate = results["win_rate"]
                backtest.max_drawdown_pct = results["max_drawdown_pct"]
                backtest.sharpe_ratio = results["sharpe_ratio"]
                backtest.profit_factor = results["profit_factor"]
                backtest.avg_trade_duration_hours = results["avg_trade_duration_hours"]
                backtest.trade_log = results["trade_log"]
                backtest.equity_curve = results["equity_curve"]

            backtest.completed_at = datetime.utcnow()
            await db.commit()

        except Exception as e:
            backtest.status = "failed"
            backtest.error_message = str(e)
            backtest.completed_at = datetime.utcnow()
            await db.commit()


def _serialize(b: Backtest) -> dict:
    return {
        "id": b.id,
        "name": b.name,
        "strategy_id": b.strategy_id,
        "symbol": b.symbol,
        "timeframe": b.timeframe,
        "start_date": b.start_date.isoformat(),
        "end_date": b.end_date.isoformat(),
        "initial_balance": b.initial_balance,
        "currency": b.currency,
        "status": b.status,
        "final_balance": b.final_balance,
        "total_return_pct": b.total_return_pct,
        "total_trades": b.total_trades,
        "winning_trades": b.winning_trades,
        "losing_trades": b.losing_trades,
        "win_rate": b.win_rate,
        "max_drawdown_pct": b.max_drawdown_pct,
        "sharpe_ratio": b.sharpe_ratio,
        "profit_factor": b.profit_factor,
        "avg_trade_duration_hours": b.avg_trade_duration_hours,
        "equity_curve": b.equity_curve,
        "trade_log": b.trade_log,
        "error_message": b.error_message,
        "started_at": b.started_at.isoformat() if b.started_at else None,
        "completed_at": b.completed_at.isoformat() if b.completed_at else None,
        "created_at": b.created_at.isoformat() if b.created_at else None,
    }
