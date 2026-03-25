"""
TradeMinds Test Suite
Run with: pytest tests/ -v
"""
import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta


# ─── Indicators ───

class TestIndicators:
    def _make_df(self, n=200, trend="up"):
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        base = 1.0800
        if trend == "up":
            close = [base + i * 0.0001 + np.random.normal(0, 0.0002) for i in range(n)]
        elif trend == "down":
            close = [base - i * 0.0001 + np.random.normal(0, 0.0002) for i in range(n)]
        else:
            close = [base + np.random.normal(0, 0.001) for _ in range(n)]

        high = [c + abs(np.random.normal(0, 0.0003)) for c in close]
        low = [c - abs(np.random.normal(0, 0.0003)) for c in close]
        open_ = [c + np.random.normal(0, 0.0002) for c in close]
        volume = [np.random.uniform(100, 1000) for _ in range(n)]

        return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)

    def test_indicators_return_dict(self):
        from bot.indicators import calculate_indicators
        df = self._make_df()
        result = calculate_indicators(df)
        assert isinstance(result, dict)
        assert len(result) > 10

    def test_rsi_in_range(self):
        from bot.indicators import calculate_indicators
        df = self._make_df()
        result = calculate_indicators(df)
        assert 0 <= result["rsi_14"] <= 100

    def test_rsi_overbought_on_uptrend(self):
        from bot.indicators import calculate_indicators
        df = self._make_df(trend="up")
        result = calculate_indicators(df)
        # Strong uptrend should push RSI high
        assert result["rsi_14"] > 50

    def test_bollinger_bands_order(self):
        from bot.indicators import calculate_indicators
        df = self._make_df()
        result = calculate_indicators(df)
        assert result["bb_upper"] > result["bb_middle"] > result["bb_lower"]

    def test_support_resistance_exist(self):
        from bot.indicators import calculate_indicators
        df = self._make_df()
        result = calculate_indicators(df)
        assert "support_1" in result
        assert "resistance_1" in result

    def test_insufficient_data_returns_empty(self):
        from bot.indicators import calculate_indicators
        df = self._make_df(n=20)
        result = calculate_indicators(df)
        assert result == {}

    def test_position_size_calculation(self):
        from bot.indicators import calculate_position_size
        lot = calculate_position_size(
            account_balance=10000,
            risk_pct=1.0,
            entry_price=1.0850,
            stop_loss=1.0800,
        )
        assert 0.01 <= lot <= 100

    def test_pattern_detection_returns_list(self):
        from bot.indicators import detect_patterns
        df = self._make_df()
        patterns = detect_patterns(df)
        assert isinstance(patterns, list)


# ─── Risk Manager ───

class TestRiskManager:
    def _make_config(self, **kwargs):
        config = MagicMock()
        config.max_positions = kwargs.get("max_positions", 25)
        config.max_daily_loss_pct = kwargs.get("max_daily_loss_pct", 5.0)
        config.max_risk_per_trade_pct = kwargs.get("max_risk_per_trade_pct", 1.0)
        config.news_pause_minutes = kwargs.get("news_pause_minutes", 30)
        config.pause_on_high_impact_news = kwargs.get("pause_on_high_impact_news", True)
        config.market_limits = kwargs.get("market_limits", {"forex": 10, "crypto": 5})
        config.daily_loss = kwargs.get("daily_loss", 0.0)
        return config

    def _make_trade(self, market_type="forex", status="open"):
        t = MagicMock()
        t.status.value = status
        t.market_type.value = market_type
        t.symbol = "EURUSD"
        return t

    @pytest.mark.asyncio
    async def test_allows_valid_trade(self):
        from risk.risk_manager import RiskManager
        config = self._make_config()
        rm = RiskManager(config)
        result = await rm.check_new_trade(
            user_id="test",
            market_type="forex",
            symbol="GBPUSD",
            lot_size=0.1,
            entry_price=1.29,
            stop_loss=1.285,
            account_balance=10000,
            open_positions=[],
            upcoming_news=[],
        )
        assert result.allowed

    @pytest.mark.asyncio
    async def test_blocks_max_positions(self):
        from risk.risk_manager import RiskManager
        config = self._make_config(max_positions=2)
        rm = RiskManager(config)
        open_pos = [self._make_trade() for _ in range(2)]
        result = await rm.check_new_trade(
            user_id="test",
            market_type="forex",
            symbol="GBPUSD",
            lot_size=0.1,
            entry_price=1.29,
            stop_loss=1.285,
            account_balance=10000,
            open_positions=open_pos,
            upcoming_news=[],
        )
        assert not result.allowed
        assert "Max positions" in result.reason

    @pytest.mark.asyncio
    async def test_blocks_daily_loss_limit(self):
        from risk.risk_manager import RiskManager
        config = self._make_config(max_daily_loss_pct=5.0, daily_loss=510.0)
        rm = RiskManager(config)
        result = await rm.check_new_trade(
            user_id="test",
            market_type="forex",
            symbol="GBPUSD",
            lot_size=0.1,
            entry_price=1.29,
            stop_loss=1.285,
            account_balance=10000,
            open_positions=[],
            upcoming_news=[],
        )
        assert not result.allowed
        assert "loss limit" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_blocks_high_impact_news(self):
        from risk.risk_manager import RiskManager
        config = self._make_config()
        rm = RiskManager(config)
        upcoming = [{"impact": "high", "minutes_until": 15, "title": "NFP"}]
        result = await rm.check_new_trade(
            user_id="test",
            market_type="forex",
            symbol="EURUSD",
            lot_size=0.1,
            entry_price=1.085,
            stop_loss=1.080,
            account_balance=10000,
            open_positions=[],
            upcoming_news=upcoming,
        )
        assert not result.allowed
        assert "news" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_blocks_duplicate_symbol(self):
        from risk.risk_manager import RiskManager
        config = self._make_config()
        rm = RiskManager(config)
        existing = self._make_trade()
        existing.symbol = "EURUSD"
        result = await rm.check_new_trade(
            user_id="test",
            market_type="forex",
            symbol="EURUSD",
            lot_size=0.1,
            entry_price=1.085,
            stop_loss=1.080,
            account_balance=10000,
            open_positions=[existing],
            upcoming_news=[],
        )
        assert not result.allowed

    def test_emergency_stop_triggers(self):
        from risk.risk_manager import RiskManager
        config = self._make_config(max_daily_loss_pct=5.0, daily_loss=600.0)
        rm = RiskManager(config)
        assert rm.should_emergency_stop(10000) is True

    def test_emergency_stop_not_triggered(self):
        from risk.risk_manager import RiskManager
        config = self._make_config(max_daily_loss_pct=5.0, daily_loss=100.0)
        rm = RiskManager(config)
        assert rm.should_emergency_stop(10000) is False


# ─── Security ───

class TestSecurity:
    def test_password_hash_and_verify(self):
        from core.security import hash_password, verify_password
        pwd = "TestPassword123!"
        hashed = hash_password(pwd)
        assert hashed != pwd
        assert verify_password(pwd, hashed)
        assert not verify_password("wrong", hashed)

    def test_access_token_create_and_verify(self):
        from core.security import create_access_token, verify_access_token
        user_id = "test-user-123"
        token = create_access_token(user_id)
        assert token
        result = verify_access_token(token)
        assert result == user_id

    def test_invalid_token_returns_none(self):
        from core.security import verify_access_token
        assert verify_access_token("invalid.token.here") is None

    def test_credential_encryption(self):
        from core.security import encrypt_credential, decrypt_credential
        secret = "my-super-secret-api-key-12345"
        encrypted = encrypt_credential(secret)
        assert encrypted != secret
        decrypted = decrypt_credential(encrypted)
        assert decrypted == secret

    def test_totp_verify(self):
        from core.security import generate_totp_secret, verify_totp
        import pyotp
        secret = generate_totp_secret()
        totp = pyotp.TOTP(secret)
        valid_code = totp.now()
        assert verify_totp(secret, valid_code)
        assert not verify_totp(secret, "000000")


# ─── Backtest Engine (basic smoke test) ───

class TestBacktestEngine:
    def _make_mock_backtest(self):
        bt = MagicMock()
        bt.id = "test-backtest"
        bt.symbol = "EURUSD"
        bt.timeframe = "1h"
        bt.start_date = datetime(2024, 1, 1)
        bt.end_date = datetime(2024, 3, 1)
        bt.initial_balance = 10000.0
        return bt

    def _make_mock_strategy(self, strategy_type="trend_following"):
        s = MagicMock()
        s.name = "Test Strategy"
        s.strategy_type.value = strategy_type
        s.parameters = {}
        s.ai_system_prompt = None
        return s

    @pytest.mark.asyncio
    async def test_backtest_returns_results_with_data(self):
        from bot.backtest_engine import BacktestEngine
        import pandas as pd
        import numpy as np

        bt = self._make_mock_backtest()
        strategy = self._make_mock_strategy()
        engine = BacktestEngine(bt, strategy)

        # Mock _load_candles to return synthetic data
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        close = [1.0800 + i * 0.00005 + np.random.normal(0, 0.0003) for i in range(n)]
        df = pd.DataFrame({
            "open": close,
            "high": [c + 0.0005 for c in close],
            "low": [c - 0.0005 for c in close],
            "close": close,
            "volume": [1000.0] * n,
        }, index=dates)

        engine._load_candles = AsyncMock(return_value=df)

        results = await engine.run()
        assert "final_balance" in results
        assert "win_rate" in results
        assert "equity_curve" in results
        assert isinstance(results["total_trades"], int)


# ─── Run ───
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
