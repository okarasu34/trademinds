from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr
from datetime import datetime
from loguru import logger

from db.database import get_db
from db.models import User, UserSession, NotificationConfig, BotConfig
from db.redis_client import check_rate_limit
from core.security import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, verify_access_token,
    generate_totp_secret, get_totp_uri, generate_qr_code_base64, verify_totp
)

router = APIRouter()
bearer = HTTPBearer()


# ─── Schemas ───

class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    totp_code: str | None = None

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

class RefreshRequest(BaseModel):
    refresh_token: str

class Enable2FARequest(BaseModel):
    totp_code: str


# ─── Dependency: get current user ───

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    token = credentials.credentials
    user_id = verify_access_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


# ─── Routes ───

@router.post("/register", status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    await db.flush()

    # Create default bot config
    bot_config = BotConfig(user_id=user.id)
    db.add(bot_config)

    # Create default notification config
    notif = NotificationConfig(user_id=user.id, notification_email=body.email)
    db.add(notif)

    await db.commit()
    logger.info(f"New user registered: {body.email}")
    return {"message": "Registration successful"}


@router.post("/login")
async def login(body: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)):
    # Rate limiting: max 10 login attempts per minute per IP
    ip = request.client.host
    allowed = await check_rate_limit(f"login:{ip}", 10, 60)
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many login attempts")

    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    # 2FA check
    if user.is_2fa_enabled:
        if not body.totp_code:
            raise HTTPException(status_code=200, detail="2FA_REQUIRED")
        if not verify_totp(user.totp_secret, body.totp_code):
            raise HTTPException(status_code=401, detail="Invalid 2FA code")

    # Create tokens
    access_token = create_access_token(user.id)
    refresh_token, refresh_expires = create_refresh_token()

    session = UserSession(
        user_id=user.id,
        refresh_token=refresh_token,
        expires_at=refresh_expires,
        ip_address=ip,
        user_agent=request.headers.get("user-agent", ""),
    )
    db.add(session)

    user.last_login = datetime.utcnow()
    await db.commit()

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": {"id": user.id, "email": user.email, "is_2fa_enabled": user.is_2fa_enabled, "base_currency": user.base_currency},
    }


@router.post("/refresh")
async def refresh_token(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(UserSession).where(
            UserSession.refresh_token == body.refresh_token,
            UserSession.expires_at > datetime.utcnow(),
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    new_access = create_access_token(session.user_id)
    new_refresh, new_expires = create_refresh_token()

    session.refresh_token = new_refresh
    session.expires_at = new_expires
    await db.commit()

    return {"access_token": new_access, "refresh_token": new_refresh}


@router.post("/logout")
async def logout(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(UserSession).where(UserSession.refresh_token == body.refresh_token))
    session = result.scalar_one_or_none()
    if session:
        await db.delete(session)
        await db.commit()
    return {"message": "Logged out"}


@router.get("/2fa/setup")
async def setup_2fa(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    secret = generate_totp_secret()
    uri = get_totp_uri(secret, user.email)
    qr = generate_qr_code_base64(uri)

    # Temporarily store secret (will be confirmed via /2fa/enable)
    user.totp_secret = secret
    await db.commit()

    return {"secret": secret, "qr_code": qr, "uri": uri}


@router.post("/2fa/enable")
async def enable_2fa(
    body: Enable2FARequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not user.totp_secret:
        raise HTTPException(status_code=400, detail="Run /2fa/setup first")
    if not verify_totp(user.totp_secret, body.totp_code):
        raise HTTPException(status_code=401, detail="Invalid TOTP code")

    user.is_2fa_enabled = True
    await db.commit()
    return {"message": "2FA enabled successfully"}


@router.post("/2fa/disable")
async def disable_2fa(
    body: Enable2FARequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not verify_totp(user.totp_secret, body.totp_code):
        raise HTTPException(status_code=401, detail="Invalid TOTP code")

    user.is_2fa_enabled = False
    user.totp_secret = None
    await db.commit()
    return {"message": "2FA disabled"}


@router.get("/me")
async def get_me(user: User = Depends(get_current_user)):
    return {
        "id": user.id,
        "email": user.email,
        "is_2fa_enabled": user.is_2fa_enabled,
        "base_currency": user.base_currency,
        "created_at": user.created_at,
        "last_login": user.last_login,
    }
