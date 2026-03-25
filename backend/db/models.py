from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text,
    Enum, ForeignKey, JSON, BigInteger, Index
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import enum

Base = declarative_base()


def gen_uuid():
    return str(uuid.uuid4())


# ─────────────────────────── ENUMS ───────────────────────────

class MarketType(str, enum.Enum):
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    STOCK = "stock"
    INDEX = "index"

class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    PENDING = "pending"

class BotStatus(str, enum.Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

class StrategyType(str, enum.Enum):
    TREND_FOLLOWING = "trend_following"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    SENTIMENT = "sentiment"
    NEWS_BASED = "news_based"
    CUSTOM = "custom"

class TradeMode(str, enum.Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

class Currency(str, enum.Enum):
    USD = "USD"
    EUR = "EUR"


# ─────────────────────────── USER ───────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_2fa_enabled = Column(Boolean, default=False)
    totp_secret = Column(String(64), nullable=True)
    base_currency = Column(Enum(Currency), default=Currency.USD)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    sessions = relationship("UserSession", back_populates="user")
    bot_config = relationship("BotConfig", back_populates="user", uselist=False)
    brokers = relationship("BrokerAccount", back_populates="user")
    trades = relationship("Trade", back_populates="user")
    strategies = relationship("Strategy", back_populates="user")
    notifications = relationship("NotificationConfig", back_populates="user", uselist=False)


class UserSession(Base):
    __tablename__ = "user_sessions"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    refresh_token = Column(String(512), unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(String(512), nullable=True)

    user = relationship("User", back_populates="sessions")


# ─────────────────────────── BOT CONFIG ───────────────────────────

class BotConfig(Base):
    __tablename__ = "bot_configs"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), unique=True, nullable=False)
    status = Column(Enum(BotStatus), default=BotStatus.STOPPED)
    trade_mode = Column(Enum(TradeMode), default=TradeMode.PAPER)

    # Global limits
    max_positions = Column(Integer, default=25)
    max_daily_loss_pct = Column(Float, default=5.0)
    max_risk_per_trade_pct = Column(Float, default=1.0)
    news_pause_minutes = Column(Integer, default=30)
    pause_on_high_impact_news = Column(Boolean, default=True)

    # Per-market position limits
    market_limits = Column(JSON, default={
        "forex": 10,
        "crypto": 5,
        "commodity": 4,
        "stock": 4,
        "index": 2
    })

    # Daily stats (reset at 00:00 UTC)
    daily_loss = Column(Float, default=0.0)
    daily_trades = Column(Integer, default=0)
    daily_reset_at = Column(DateTime, nullable=True)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="bot_config")


# ─────────────────────────── BROKER ACCOUNTS ───────────────────────────

class BrokerAccount(Base):
    __tablename__ = "broker_accounts"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)           # e.g. "My Binance"
    broker_type = Column(String(50), nullable=False)     # binance, mt4, mt5, ibkr
    market_type = Column(Enum(MarketType), nullable=False)
    is_active = Column(Boolean, default=True)

    # Encrypted credentials (AES-256)
    encrypted_api_key = Column(Text, nullable=True)
    encrypted_api_secret = Column(Text, nullable=True)
    encrypted_extra = Column(Text, nullable=True)        # MetaAPI token, IBKR credentials etc.

    # Account info (fetched from broker)
    account_id = Column(String(100), nullable=True)
    balance = Column(Float, default=0.0)
    equity = Column(Float, default=0.0)
    margin_used = Column(Float, default=0.0)
    currency = Column(String(10), default="USD")

    is_connected = Column(Boolean, default=False)
    last_sync = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="brokers")
    trades = relationship("Trade", back_populates="broker")

    __table_args__ = (Index("ix_broker_user", "user_id", "broker_type"),)


# ─────────────────────────── STRATEGIES ───────────────────────────

class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    strategy_type = Column(Enum(StrategyType), nullable=False)
    is_active = Column(Boolean, default=True)
    is_builtin = Column(Boolean, default=False)          # pre-built or user-created

    # Markets this strategy applies to
    markets = Column(JSON, default=[])                   # ["forex", "crypto"]
    symbols = Column(JSON, default=[])                   # ["EURUSD", "BTCUSDT"] empty=all

    # Strategy parameters (flexible JSON)
    parameters = Column(JSON, default={})

    # AI prompt override (for custom strategies)
    ai_system_prompt = Column(Text, nullable=True)

    # Performance stats
    total_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    avg_pnl_per_trade = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, nullable=True)

    priority = Column(Integer, default=0)                # higher = more priority
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="strategies")
    trades = relationship("Trade", back_populates="strategy")
    backtests = relationship("Backtest", back_populates="strategy")


# ─────────────────────────── TRADES ───────────────────────────

class Trade(Base):
    __tablename__ = "trades"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    broker_id = Column(String, ForeignKey("broker_accounts.id"), nullable=False)
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=True)

    # Trade info
    symbol = Column(String(30), nullable=False)
    market_type = Column(Enum(MarketType), nullable=False)
    side = Column(Enum(OrderSide), nullable=False)
    status = Column(Enum(OrderStatus), default=OrderStatus.OPEN)
    trade_mode = Column(Enum(TradeMode), default=TradeMode.LIVE)

    # Execution
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    lot_size = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    trailing_stop = Column(Float, nullable=True)

    # P&L
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    commission = Column(Float, default=0.0)
    swap = Column(Float, default=0.0)
    currency = Column(String(10), default="USD")

    # AI reasoning
    ai_reasoning = Column(Text, nullable=True)           # why bot opened this trade
    ai_confidence = Column(Float, nullable=True)         # 0-1 confidence score
    signals_used = Column(JSON, default=[])              # which signals triggered
    news_context = Column(Text, nullable=True)           # relevant news at open time

    # Timestamps
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    closed_by = Column(String(20), nullable=True)        # "bot" or "manual"

    # External IDs
    broker_order_id = Column(String(100), nullable=True)

    user = relationship("User", back_populates="trades")
    broker = relationship("BrokerAccount", back_populates="trades")
    strategy = relationship("Strategy", back_populates="trades")

    __table_args__ = (
        Index("ix_trade_user_status", "user_id", "status"),
        Index("ix_trade_user_opened", "user_id", "opened_at"),
        Index("ix_trade_symbol", "symbol"),
    )


# ─────────────────────────── BACKTESTS ───────────────────────────

class Backtest(Base):
    __tablename__ = "backtests"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=False)
    name = Column(String(100), nullable=False)

    # Test parameters
    symbol = Column(String(30), nullable=False)
    timeframe = Column(String(10), nullable=False)       # 1m, 5m, 1h, 1d
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_balance = Column(Float, default=10000.0)
    currency = Column(String(10), default="USD")

    # Results
    final_balance = Column(Float, nullable=True)
    total_return_pct = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    winning_trades = Column(Integer, nullable=True)
    losing_trades = Column(Integer, nullable=True)
    win_rate = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    avg_trade_duration_hours = Column(Float, nullable=True)

    # Full trade log as JSON
    trade_log = Column(JSON, default=[])
    equity_curve = Column(JSON, default=[])

    status = Column(String(20), default="pending")       # pending, running, completed, failed
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    strategy = relationship("Strategy", back_populates="backtests")

    __table_args__ = (Index("ix_backtest_user", "user_id", "created_at"),)


# ─────────────────────────── MARKET DATA ───────────────────────────

class MarketCandle(Base):
    __tablename__ = "market_candles"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(30), nullable=False)
    timeframe = Column(String(10), nullable=False)
    open_time = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, default=0.0)

    __table_args__ = (
        Index("ix_candle_symbol_tf_time", "symbol", "timeframe", "open_time", unique=True),
    )


# ─────────────────────────── ECONOMIC CALENDAR ───────────────────────────

class EconomicEvent(Base):
    __tablename__ = "economic_events"

    id = Column(String, primary_key=True, default=gen_uuid)
    title = Column(String(255), nullable=False)
    country = Column(String(10), nullable=False)
    currency = Column(String(10), nullable=False)
    impact = Column(String(10), nullable=False)          # low, medium, high
    scheduled_at = Column(DateTime, nullable=False)
    previous = Column(String(50), nullable=True)
    forecast = Column(String(50), nullable=True)
    actual = Column(String(50), nullable=True)
    affected_symbols = Column(JSON, default=[])

    __table_args__ = (Index("ix_event_scheduled", "scheduled_at", "impact"),)


# ─────────────────────────── BOT HEALTH ───────────────────────────

class BotHealthLog(Base):
    __tablename__ = "bot_health_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    status = Column(String(20), nullable=False)
    message = Column(Text, nullable=True)
    open_positions = Column(Integer, default=0)
    daily_pnl = Column(Float, default=0.0)
    checked_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_health_user_time", "user_id", "checked_at"),)


# ─────────────────────────── AI SIGNAL LOG ───────────────────────────

class AISignalLog(Base):
    __tablename__ = "ai_signal_logs"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    symbol = Column(String(30), nullable=False)
    market_type = Column(Enum(MarketType), nullable=False)
    signal = Column(String(10), nullable=False)          # buy, sell, hold
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=True)
    indicators = Column(JSON, default={})
    news_impact = Column(Text, nullable=True)
    acted_on = Column(Boolean, default=False)
    trade_id = Column(String, ForeignKey("trades.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_signal_user_time", "user_id", "created_at"),)


# ─────────────────────────── NOTIFICATIONS ───────────────────────────

class NotificationConfig(Base):
    __tablename__ = "notification_configs"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), unique=True, nullable=False)
    telegram_enabled = Column(Boolean, default=False)
    telegram_chat_id = Column(String(100), nullable=True)
    email_enabled = Column(Boolean, default=True)
    notification_email = Column(String(255), nullable=True)

    # What to notify
    on_trade_open = Column(Boolean, default=True)
    on_trade_close = Column(Boolean, default=True)
    on_daily_limit_hit = Column(Boolean, default=True)
    on_bot_error = Column(Boolean, default=True)
    on_high_impact_news = Column(Boolean, default=True)

    user = relationship("User", back_populates="notifications")
