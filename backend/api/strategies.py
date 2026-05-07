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
        "name": "Alpha Trend",
        "description": "Alpha Trend crossover + EMA hizalaması. Trend takip stratejisi. ATR + MFI tabanlı dinamik trend çizgisi ile EMA13/21 crossover kombinasyonu.",
        "strategy_type": StrategyType.TREND_FOLLOWING,
        "markets": ["forex", "crypto", "commodity", "index"],
        "parameters": {
            "atr_period": 13,
            "mfi_period": 13,
            "coeff": 1.1,
            "ema_fast": 13,
            "ema_slow": 21,
            "ema_trend": 89,
            "min_confidence": 0.6,
        },
        "is_builtin": True,
        "priority": 1,
    },
    {
        "name": "RSI Divergence",
        "description": "RSI + MACD diverjans tespiti + Fibonacci seviyeleri. Dönüş stratejisi. En az 2 indikatörde aynı anda diverjans şartı.",
        "strategy_type": StrategyType.MOMENTUM,
        "markets": ["forex", "crypto", "commodity", "index"],
        "parameters": {
            "rsi_length": 13,
            "macd_fast": 13,
            "macd_slow": 21,
            "macd_signal": 8,
            "fibo_period": 144,
            "min_divergence_count": 2,
            "min_confidence": 0.65,
        },
        "is_builtin": True,
        "priority": 2,
    },
    {
        "name": "Smart Money",
        "description": "Order Block + FVG + POC. Kurumsal trader mantığı. Fiyat order block içinde ve POC yakınındayken işlem açar.",
        "strategy_type": StrategyType.CUSTOM,
        "markets": ["forex", "crypto", "commodity", "index"],
        "parameters": {
            "ob_period": 8,
            "fvg_atr_mult": 0.5,
            "poc_period": 89,
            "poc_proximity_pct": 0.5,
            "min_confidence": 0.70,
        },
        "is_builtin": True,
        "priority": 3,
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
    existing = result.scalars().all()
    if not existing:
        await _seed_builtin_strategies(user.id, db)

    result = await db.execute(
        select(Strategy)
        .where(Strategy.user_id == user.id)
        .order_by(Strategy.priority, Strategy.created_at)
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
        raise HTTPException(status_code=400, detail="Cannot delete built-in strategies.")
    await db.delete(s)
    await db.commit()


@router.post("/{strategy_id}/toggle")
async def toggle_strategy(
    strategy_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Bir strateji aktif edilince diğerleri otomatik pasif olur.
    Zaten aktif olan strateji tekrar tıklanırsa değişmez."""

    s = await _get_or_404(strategy_id, user.id, db)

    # Zaten aktifse dokunma
    if s.is_active:
        return {"is_active": True, "message": f"{s.name} already active"}

    # Tüm stratejileri pasif yap
    all_result = await db.execute(
        select(Strategy).where(Strategy.user_id == user.id)
    )
    all_strategies = all_result.scalars().all()
    for strategy in all_strategies:
        strategy.is_active = False

    # Bu stratejiyi aktif et
    s.is_active = True
    await db.commit()

    return {"is_active": True, "message": f"{s.name} activated"}


# ─── Helpers ───

async def _get_or_404(strategy_id: str, user_id: str, db: AsyncSession) -> Strategy:
    result = await db.execute(
        select(Strategy).where(
            Strategy.id == strategy_id,
            Strategy.user_id == user_id,
        )
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