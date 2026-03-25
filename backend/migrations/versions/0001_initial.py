"""Initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-03-25
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # users
    op.create_table("users",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("is_2fa_enabled", sa.Boolean(), default=False),
        sa.Column("totp_secret", sa.String(64), nullable=True),
        sa.Column("base_currency", sa.String(10), default="USD"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("last_login", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"])

    # user_sessions
    op.create_table("user_sessions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("refresh_token", sa.String(512), unique=True, nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("ip_address", sa.String(64), nullable=True),
        sa.Column("user_agent", sa.String(512), nullable=True),
    )

    # bot_configs
    op.create_table("bot_configs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), unique=True, nullable=False),
        sa.Column("status", sa.String(20), default="stopped"),
        sa.Column("trade_mode", sa.String(20), default="paper"),
        sa.Column("max_positions", sa.Integer(), default=25),
        sa.Column("max_daily_loss_pct", sa.Float(), default=5.0),
        sa.Column("max_risk_per_trade_pct", sa.Float(), default=1.0),
        sa.Column("news_pause_minutes", sa.Integer(), default=30),
        sa.Column("pause_on_high_impact_news", sa.Boolean(), default=True),
        sa.Column("market_limits", postgresql.JSON(), nullable=True),
        sa.Column("daily_loss", sa.Float(), default=0.0),
        sa.Column("daily_trades", sa.Integer(), default=0),
        sa.Column("daily_reset_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )

    # broker_accounts
    op.create_table("broker_accounts",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("broker_type", sa.String(50), nullable=False),
        sa.Column("market_type", sa.String(20), nullable=False),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("encrypted_api_key", sa.Text(), nullable=True),
        sa.Column("encrypted_api_secret", sa.Text(), nullable=True),
        sa.Column("encrypted_extra", sa.Text(), nullable=True),
        sa.Column("account_id", sa.String(100), nullable=True),
        sa.Column("balance", sa.Float(), default=0.0),
        sa.Column("equity", sa.Float(), default=0.0),
        sa.Column("margin_used", sa.Float(), default=0.0),
        sa.Column("currency", sa.String(10), default="USD"),
        sa.Column("is_connected", sa.Boolean(), default=False),
        sa.Column("last_sync", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_broker_user", "broker_accounts", ["user_id", "broker_type"])

    # strategies
    op.create_table("strategies",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("strategy_type", sa.String(30), nullable=False),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("is_builtin", sa.Boolean(), default=False),
        sa.Column("markets", postgresql.JSON(), nullable=True),
        sa.Column("symbols", postgresql.JSON(), nullable=True),
        sa.Column("parameters", postgresql.JSON(), nullable=True),
        sa.Column("ai_system_prompt", sa.Text(), nullable=True),
        sa.Column("total_trades", sa.Integer(), default=0),
        sa.Column("win_rate", sa.Float(), default=0.0),
        sa.Column("total_pnl", sa.Float(), default=0.0),
        sa.Column("avg_pnl_per_trade", sa.Float(), default=0.0),
        sa.Column("max_drawdown", sa.Float(), default=0.0),
        sa.Column("sharpe_ratio", sa.Float(), nullable=True),
        sa.Column("priority", sa.Integer(), default=0),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )

    # trades
    op.create_table("trades",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("broker_id", sa.String(), sa.ForeignKey("broker_accounts.id"), nullable=False),
        sa.Column("strategy_id", sa.String(), sa.ForeignKey("strategies.id"), nullable=True),
        sa.Column("symbol", sa.String(30), nullable=False),
        sa.Column("market_type", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("status", sa.String(20), default="open"),
        sa.Column("trade_mode", sa.String(20), default="live"),
        sa.Column("entry_price", sa.Float(), nullable=False),
        sa.Column("exit_price", sa.Float(), nullable=True),
        sa.Column("lot_size", sa.Float(), nullable=False),
        sa.Column("stop_loss", sa.Float(), nullable=True),
        sa.Column("take_profit", sa.Float(), nullable=True),
        sa.Column("trailing_stop", sa.Float(), nullable=True),
        sa.Column("pnl", sa.Float(), nullable=True),
        sa.Column("pnl_pct", sa.Float(), nullable=True),
        sa.Column("commission", sa.Float(), default=0.0),
        sa.Column("swap", sa.Float(), default=0.0),
        sa.Column("currency", sa.String(10), default="USD"),
        sa.Column("ai_reasoning", sa.Text(), nullable=True),
        sa.Column("ai_confidence", sa.Float(), nullable=True),
        sa.Column("signals_used", postgresql.JSON(), nullable=True),
        sa.Column("news_context", sa.Text(), nullable=True),
        sa.Column("opened_at", sa.DateTime(), nullable=True),
        sa.Column("closed_at", sa.DateTime(), nullable=True),
        sa.Column("closed_by", sa.String(20), nullable=True),
        sa.Column("broker_order_id", sa.String(100), nullable=True),
    )
    op.create_index("ix_trade_user_status", "trades", ["user_id", "status"])
    op.create_index("ix_trade_user_opened", "trades", ["user_id", "opened_at"])
    op.create_index("ix_trade_symbol", "trades", ["symbol"])

    # backtests
    op.create_table("backtests",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("strategy_id", sa.String(), sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("symbol", sa.String(30), nullable=False),
        sa.Column("timeframe", sa.String(10), nullable=False),
        sa.Column("start_date", sa.DateTime(), nullable=False),
        sa.Column("end_date", sa.DateTime(), nullable=False),
        sa.Column("initial_balance", sa.Float(), default=10000.0),
        sa.Column("currency", sa.String(10), default="USD"),
        sa.Column("final_balance", sa.Float(), nullable=True),
        sa.Column("total_return_pct", sa.Float(), nullable=True),
        sa.Column("total_trades", sa.Integer(), nullable=True),
        sa.Column("winning_trades", sa.Integer(), nullable=True),
        sa.Column("losing_trades", sa.Integer(), nullable=True),
        sa.Column("win_rate", sa.Float(), nullable=True),
        sa.Column("max_drawdown_pct", sa.Float(), nullable=True),
        sa.Column("sharpe_ratio", sa.Float(), nullable=True),
        sa.Column("profit_factor", sa.Float(), nullable=True),
        sa.Column("avg_trade_duration_hours", sa.Float(), nullable=True),
        sa.Column("trade_log", postgresql.JSON(), nullable=True),
        sa.Column("equity_curve", postgresql.JSON(), nullable=True),
        sa.Column("status", sa.String(20), default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_backtest_user", "backtests", ["user_id", "created_at"])

    # market_candles
    op.create_table("market_candles",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(30), nullable=False),
        sa.Column("timeframe", sa.String(10), nullable=False),
        sa.Column("open_time", sa.DateTime(), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), default=0.0),
    )
    op.create_index("ix_candle_symbol_tf_time", "market_candles", ["symbol", "timeframe", "open_time"], unique=True)

    # economic_events
    op.create_table("economic_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("country", sa.String(10), nullable=False),
        sa.Column("currency", sa.String(10), nullable=False),
        sa.Column("impact", sa.String(10), nullable=False),
        sa.Column("scheduled_at", sa.DateTime(), nullable=False),
        sa.Column("previous", sa.String(50), nullable=True),
        sa.Column("forecast", sa.String(50), nullable=True),
        sa.Column("actual", sa.String(50), nullable=True),
        sa.Column("affected_symbols", postgresql.JSON(), nullable=True),
    )
    op.create_index("ix_event_scheduled", "economic_events", ["scheduled_at", "impact"])

    # bot_health_logs
    op.create_table("bot_health_logs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("open_positions", sa.Integer(), default=0),
        sa.Column("daily_pnl", sa.Float(), default=0.0),
        sa.Column("checked_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_health_user_time", "bot_health_logs", ["user_id", "checked_at"])

    # ai_signal_logs
    op.create_table("ai_signal_logs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("symbol", sa.String(30), nullable=False),
        sa.Column("market_type", sa.String(20), nullable=False),
        sa.Column("signal", sa.String(10), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("indicators", postgresql.JSON(), nullable=True),
        sa.Column("news_impact", sa.Text(), nullable=True),
        sa.Column("acted_on", sa.Boolean(), default=False),
        sa.Column("trade_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_signal_user_time", "ai_signal_logs", ["user_id", "created_at"])

    # notification_configs
    op.create_table("notification_configs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), unique=True, nullable=False),
        sa.Column("telegram_enabled", sa.Boolean(), default=False),
        sa.Column("telegram_chat_id", sa.String(100), nullable=True),
        sa.Column("email_enabled", sa.Boolean(), default=True),
        sa.Column("notification_email", sa.String(255), nullable=True),
        sa.Column("on_trade_open", sa.Boolean(), default=True),
        sa.Column("on_trade_close", sa.Boolean(), default=True),
        sa.Column("on_daily_limit_hit", sa.Boolean(), default=True),
        sa.Column("on_bot_error", sa.Boolean(), default=True),
        sa.Column("on_high_impact_news", sa.Boolean(), default=True),
    )


def downgrade() -> None:
    for table in [
        "notification_configs", "ai_signal_logs", "bot_health_logs",
        "economic_events", "market_candles", "backtests", "trades",
        "strategies", "broker_accounts", "bot_configs", "user_sessions", "users",
    ]:
        op.drop_table(table)
