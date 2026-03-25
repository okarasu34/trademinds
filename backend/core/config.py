from pydantic_settings import BaseSettings
from typing import Optional
import secrets


class Settings(BaseSettings):
    # App
    APP_NAME: str = "TradeMinds"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    SECRET_KEY: str = secrets.token_urlsafe(64)
    API_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://trademinds:password@localhost:5432/trademinds"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_POOL_SIZE: int = 20

    # JWT
    JWT_SECRET_KEY: str = secrets.token_urlsafe(64)
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # 2FA
    TOTP_ISSUER: str = "TradeMinds"

    # Anthropic
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    ANTHROPIC_MAX_TOKENS: int = 4096

    # MyFXBook

    # Broker API Keys (encrypted in DB, these are master encryption keys)
    BROKER_ENCRYPTION_KEY: str = secrets.token_urlsafe(32)

    # Bot Settings
    BOT_HEALTH_CHECK_INTERVAL: int = 60       # seconds
    BOT_MAX_POSITIONS: int = 25
    BOT_MAX_DAILY_LOSS_PCT: float = 5.0       # % of balance
    BOT_MAX_RISK_PER_TRADE_PCT: float = 1.0   # % of balance
    BOT_NEWS_PAUSE_MINUTES: int = 30          # pause before high-impact news

    # Currency
    BASE_CURRENCY: str = "USD"               # USD or EUR

    # Notifications
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    NOTIFICATION_EMAIL: str = ""

    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "https://yourdomain.com"]

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
