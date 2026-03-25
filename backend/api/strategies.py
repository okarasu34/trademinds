from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional

from db.database import get_db
from db.models import Strategy, StrategyType, User
from api.auth import get_current_user

router = APIRouter()

# ─── Built-in strategies (seeded on first run) ───

BUILTIN_STRATEGIES = [
    {
        "name": "Trend Following",
        "description": "Follows EMA crossovers and ADX trend strength. Enters on pullbacks in the direction of the primary trend. Best for trending forex and index markets.",
        "strategy_type": StrategyType.TREND_FOLLOWING,
        "markets": ["forex", "index", "stock"],
        "parameters": {
            "ema_fast": 21, "ema_slow": 50, "ema_long": 200,
            "adx_threshold": 25, "rsi_min": 40, "rsi_max": 70,
            "min_confidence": 0.70,
        },
        "is_builtin": True,
    },
    {
        "name": "Momentum",
        "description": "Captures strong momentum moves using RSI, MACD, and volume confirmation. High confidence threshold. Best for crypto and volatile assets.",
        "strategy_type": StrategyType.MOMENTUM,
        "markets": ["crypto", "stock"],
        "parameters": {
            "rsi_buy_threshold": 55, "rsi_sell_threshold": 45,
            "macd_confirm": True, "volume_ratio_min": 1.5,
            "min_confidence": 0.75,
        },
        "is_builtin": True,
    },
    {
        "name": "Mean Reversion",
        "description": "Trades overbought/oversold conditions using Bollinger Bands and Stochastic. Best in ranging, non-trending markets.",
        "strategy_type": StrategyType.MEAN_REVERSION,
        "markets": ["commodity", "forex"],
        "parameters": {
            "bb_position_buy": 0.15, "bb_position_sell": 0.85,
            "rsi_oversold": 30, "rsi_overbought": 70,
            "stoch_oversold": 20, "stoch_overbought": 80,
            "min_confidence": 0.68,
        },
        "is_builtin": True,
    },
    {
        "name": "Sentiment Analysis",
        "description": "NLP-based news and social media sentiment analysis via Claude AI. Reads and interprets market narrative to trade in sentiment direction.",
        "strategy_type": StrategyType.SENTIMENT,
        "markets": ["forex", "crypto", "stock"],
        "parameters": {
            "sentiment_threshold": 0.6, "news_weight": 0.7,
            "technical_weight": 0.3, "min_confidence": 0.72,
        },
        "is_builtin": True,
    },
    {
        "name": "News Based",
        "description": "Trades economic data releases. Waits for MyFXBook high-impact event, then enters on post-release momentum. Strict risk management.",
        "strategy_type": StrategyType.NEWS_BASED,
        "markets": ["forex", "commodity"],
        "parameters": {
            "entry_delay_seconds": 30, "max_spread_pips": 5,
            "impact_filter": ["high"], "min_confidence": 0.73,
        },
        "is_builtin": True,
    },
]


# ─── Schemas ───

class StrategyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    strategy_type: StrategyType
    markets: list[str]
    symbols: list[str] = []
    parameters: dict = {}
    ai_system_prompt: Optional[str] = None
    priority: int = 0


class StrategyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    markets: Optional[list[str]] = None
    symbols: Optional[list[str]] = None
    parameters: Optional[dict] = None
    ai_system_prompt: Optional[str] = None
    priority: Optional[int] = None


# ─── Routes ───

@router.get("")
async def list_strategies(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Seed built-ins if none exist
    result = await db.execute(
        select(Strategy).where(Strategy.user_id == user.id, Strategy.is_builtin == True)
    )
    if not result.scalars().all():
        await _seed_builtin_strategies(user.id, db)

    result = await db.execute(
        select(Strategy).where(Strategy.user_id == user.id).order_by(Strategy.priority.desc(), Strategy.created_at)
    )
    strategies = result.scalars().all()
    return [_serialize(s) for s in strategies]


@router.post("", status_code=201)
async def create_strategy(
    body: StrategyCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    strategy = Strategy(
        user_id=user.id,
        name=body.name,
        description=body.description,
        strategy_type=body.strategy_type,
        markets=body.markets,
        symbols=body.symbols,
        parameters=body.parameters,
        ai_system_prompt=body.ai_system_prompt,
        priority=body.priority,
        is_builtin=False,
    )
    db.add(strategy)
    await db.commit()
    await db.refresh(strategy)
    return _serialize(strategy)


@router.get("/{strategy_id}")
async def get_strategy(
    strategy_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    s = await _get_or_404(strategy_id, user.id, db)
    return _serialize(s)


@router.put("/{strategy_id}")
async def update_strategy(
    strategy_id: str,
    body: StrategyUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    s = await _get_or_404(strategy_id, user.id, db)
    for key, value in body.dict(exclude_none=True).items():
        setattr(s, key, value)
    await db.commit()
    return _serialize(s)


@router.delete("/{strategy_id}", status_code=204)
async def delete_strategy(
    strategy_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    s = await _get_or_404(strategy_id, user.id, db)
    if s.is_builtin:
        raise HTTPException(status_code=400, detail="Cannot delete built-in strategies. Disable them instead.")
    await db.delete(s)
    await db.commit()


@router.post("/{strategy_id}/toggle")
async def toggle_strategy(
    strategy_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    s = await _get_or_404(strategy_id, user.id, db)
    s.is_active = not s.is_active
    await db.commit()
    return {"is_active": s.is_active}


# ─── Helpers ───

async def _get_or_404(strategy_id: str, user_id: str, db: AsyncSession) -> Strategy:
    result = await db.execute(
        select(Strategy).where(Strategy.id == strategy_id, Strategy.user_id == user_id)
    )
    s = result.scalar_one_or_none()
    if not s:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return s


async def _seed_builtin_strategies(user_id: str, db: AsyncSession):
    for data in BUILTIN_STRATEGIES:
        strategy = Strategy(user_id=user_id, **data)
        db.add(strategy)
    await db.commit()


def _serialize(s: Strategy) -> dict:
    return {
        "id": s.id,
        "name": s.name,
        "description": s.description,
        "strategy_type": s.strategy_type.value,
        "is_active": s.is_active,
        "is_builtin": s.is_builtin,
        "markets": s.markets,
        "symbols": s.symbols,
        "parameters": s.parameters,
        "ai_system_prompt": s.ai_system_prompt,
        "priority": s.priority,
        "total_trades": s.total_trades,
        "win_rate": s.win_rate,
        "total_pnl": s.total_pnl,
        "max_drawdown": s.max_drawdown,
        "sharpe_ratio": s.sharpe_ratio,
        "created_at": s.created_at.isoformat() if s.created_at else None,
    }
