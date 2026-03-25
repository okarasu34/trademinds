from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from db.database import get_db
from db.models import BrokerAccount, MarketType, User
from api.auth import get_current_user
from core.security import encrypt_credential

router = APIRouter()


class BrokerCreate(BaseModel):
    name: str
    broker_type: str
    market_type: MarketType
    api_key: str
    api_secret: str
    extra: Optional[str] = None


@router.get("")
async def list_brokers(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    r = await db.execute(select(BrokerAccount).where(BrokerAccount.user_id == user.id))
    return [_serialize(b) for b in r.scalars().all()]


@router.post("", status_code=201)
async def add_broker(body: BrokerCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    broker = BrokerAccount(
        user_id=user.id, name=body.name,
        broker_type=body.broker_type.lower(), market_type=body.market_type,
        encrypted_api_key=encrypt_credential(body.api_key),
        encrypted_api_secret=encrypt_credential(body.api_secret),
        encrypted_extra=encrypt_credential(body.extra) if body.extra else None,
    )
    db.add(broker)
    await db.commit()
    await db.refresh(broker)
    return _serialize(broker)


@router.delete("/{broker_id}", status_code=204)
async def remove_broker(broker_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    r = await db.execute(select(BrokerAccount).where(BrokerAccount.id == broker_id, BrokerAccount.user_id == user.id))
    broker = r.scalar_one_or_none()
    if not broker:
        raise HTTPException(404, "Broker not found")
    await db.delete(broker)
    await db.commit()


@router.post("/{broker_id}/test")
async def test_broker(broker_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    r = await db.execute(select(BrokerAccount).where(BrokerAccount.id == broker_id, BrokerAccount.user_id == user.id))
    broker = r.scalar_one_or_none()
    if not broker:
        raise HTTPException(404, "Broker not found")

    from brokers.base_adapter import get_broker_adapter
    adapter = get_broker_adapter(broker)
    try:
        connected = await adapter.connect()
        if connected:
            info = await adapter.get_account_info()
            broker.balance = info.balance
            broker.equity = info.equity
            broker.is_connected = True
            broker.last_sync = datetime.utcnow()
            await db.commit()
            return {"connected": True, "balance": info.balance, "currency": info.currency}
        return {"connected": False, "error": "Connection failed"}
    except Exception as e:
        return {"connected": False, "error": str(e)}
    finally:
        try:
            await adapter.disconnect()
        except Exception:
            pass


def _serialize(b: BrokerAccount) -> dict:
    return {
        "id": b.id, "name": b.name, "broker_type": b.broker_type,
        "market_type": b.market_type.value, "is_active": b.is_active,
        "is_connected": b.is_connected, "balance": b.balance,
        "equity": b.equity, "currency": b.currency,
        "last_sync": b.last_sync.isoformat() if b.last_sync else None,
    }
