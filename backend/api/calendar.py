from fastapi import APIRouter, Depends, Query
from typing import Optional
from db.models import User
from api.auth import get_current_user
from data.calendar import calendar_client

router = APIRouter()


@router.get("")
async def get_calendar(
    hours_ahead: int = Query(24, ge=1, le=168),
    impact: Optional[str] = Query(None, description="Comma-separated: low,medium,high"),
    currencies: Optional[str] = Query(None, description="Comma-separated: USD,EUR,GBP"),
    user: User = Depends(get_current_user),
):
    impact_filter = impact.split(",") if impact else None
    currency_filter = currencies.split(",") if currencies else None
    return await calendar_client.get_calendar(
        hours_ahead=hours_ahead,
        impact_filter=impact_filter,
        currency_filter=currency_filter,
    )


@router.get("/high-impact")
async def get_high_impact(
    minutes: int = Query(60, ge=5, le=240),
    user: User = Depends(get_current_user),
):
    return await calendar_client.get_upcoming_high_impact(minutes_ahead=minutes)


@router.get("/symbol/{symbol}")
async def get_events_for_symbol(
    symbol: str,
    hours_ahead: int = Query(4, ge=1, le=48),
    user: User = Depends(get_current_user),
):
    return await calendar_client.get_events_for_symbol(symbol, hours_ahead=hours_ahead)


@router.get("/status")
async def calendar_status(user: User = Depends(get_current_user)):
    available = await calendar_client.is_available()
    return {"available": available, "source": "MyFXBook XML"}
