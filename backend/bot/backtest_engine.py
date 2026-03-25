"""
Backtest Engine
Runs strategy simulations on historical OHLCV data.
Produces equity curve, win rate, drawdown, Sharpe ratio.
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Callable
from loguru import logger

from bot.indicators import calculate_indicators, detect_patterns, calculate_position_size
from bot.ai_engine import analyze_market
from db.models import Backtest, Strategy
from db.database import AsyncSessionLocal


class BacktestEngine:

    def __init__(self, backtest: Backtest, strategy: Strategy):
        self.backtest = backtest
        self.strategy = strategy

    async def run(self) -> dict:
        """
        Run full backtest simulation.
        Returns results dict with equity curve and trade log.
        """
        logger.info(f"Starting backtest {self.backtest.id} — {self.backtest.symbol} {self.backtest.timeframe}")

        # Load historical data (from DB or fetch)
        df = await self._load_candles()
        if df is None or len(df) < 100:
            return {"error": "Insufficient historical data"}

        balance = self.backtest.initial_balance
        equity = balance
        peak_equity = balance
        max_drawdown = 0.0
        trades = []
        equity_curve = []
        open_position = None

        # Simulate bar by bar
        for i in range(50, len(df)):
            current_bar = df.iloc[i]
            window = df.iloc[max(0, i-200):i+1]
            indicators = calculate_indicators(window)
            patterns = detect_patterns(window)
            if patterns:
                indicators["patterns"] = patterns

            price = current_bar["close"]

            # Check if we have an open position to manage
            if open_position:
                # Check SL/TP hit
                if open_position["side"] == "buy":
                    if current_bar["low"] <= open_position["stop_loss"]:
                        result = self._close_position(open_position, open_position["stop_loss"], i, df)
                        balance += result["pnl"]
                        trades.append(result)
                        open_position = None
                    elif current_bar["high"] >= open_position["take_profit"]:
                        result = self._close_position(open_position, open_position["take_profit"], i, df)
                        balance += result["pnl"]
                        trades.append(result)
                        open_position = None
                else:  # sell
                    if current_bar["high"] >= open_position["stop_loss"]:
                        result = self._close_position(open_position, open_position["stop_loss"], i, df)
                        balance += result["pnl"]
                        trades.append(result)
                        open_position = None
                    elif current_bar["low"] <= open_position["take_profit"]:
                        result = self._close_position(open_position, open_position["take_profit"], i, df)
                        balance += result["pnl"]
                        trades.append(result)
                        open_position = None

            # Only open new position if none open
            if not open_position and balance > 0:
                signal = await self._get_signal(indicators, price, balance)

                if signal["signal"] in ("buy", "sell") and signal.get("confidence", 0) >= 0.65:
                    stop_loss = signal.get("stop_loss", price * (0.99 if signal["signal"] == "buy" else 1.01))
                    take_profit = signal.get("take_profit", price * (1.02 if signal["signal"] == "buy" else 0.98))
                    lot_size = calculate_position_size(balance, 1.0, price, stop_loss)

                    open_position = {
                        "symbol": self.backtest.symbol,
                        "side": signal["signal"],
                        "entry_price": price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "lot_size": lot_size,
                        "open_bar": i,
                        "open_time": df.index[i],
                        "confidence": signal.get("confidence", 0),
                        "reasoning": signal.get("reasoning", ""),
                    }

            # Update equity curve
            unrealized = 0
            if open_position:
                if open_position["side"] == "buy":
                    unrealized = (price - open_position["entry_price"]) * open_position["lot_size"] * 100000
                else:
                    unrealized = (open_position["entry_price"] - price) * open_position["lot_size"] * 100000

            equity = balance + unrealized
            equity_curve.append(round(equity, 2))

            if equity > peak_equity:
                peak_equity = equity
            drawdown = (peak_equity - equity) / peak_equity * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Close any remaining position at last bar
        if open_position:
            last_price = df.iloc[-1]["close"]
            result = self._close_position(open_position, last_price, len(df)-1, df)
            balance += result["pnl"]
            trades.append(result)

        # ─── Compute results ───
        if not trades:
            return {"error": "No trades generated"}

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        durations = [t.get("duration_bars", 1) for t in trades]

        sharpe = (np.mean(pnls) / (np.std(pnls) + 1e-10)) * (252 ** 0.5) if len(pnls) > 1 else 0

        results = {
            "final_balance": round(balance, 2),
            "total_return_pct": round((balance - self.backtest.initial_balance) / self.backtest.initial_balance * 100, 2),
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(len(wins) / len(trades) * 100, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
            "avg_trade_duration_hours": round(np.mean(durations) if durations else 0, 1),
            "trade_log": trades[:500],  # cap for DB storage
            "equity_curve": equity_curve[::max(1, len(equity_curve)//200)],  # sample to 200 points
        }

        logger.info(
            f"Backtest complete: {results['total_trades']} trades, "
            f"WR={results['win_rate']}%, return={results['total_return_pct']}%"
        )
        return results

    def _close_position(self, position: dict, exit_price: float, bar: int, df: pd.DataFrame) -> dict:
        if position["side"] == "buy":
            pnl = (exit_price - position["entry_price"]) * position["lot_size"] * 100000
        else:
            pnl = (position["entry_price"] - exit_price) * position["lot_size"] * 100000

        duration = bar - position["open_bar"]
        return {
            "symbol": position["symbol"],
            "side": position["side"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "lot_size": position["lot_size"],
            "pnl": round(pnl, 2),
            "stop_loss": position["stop_loss"],
            "take_profit": position["take_profit"],
            "open_time": str(position["open_time"]),
            "close_time": str(df.index[bar]),
            "duration_bars": duration,
            "confidence": position.get("confidence", 0),
            "reasoning": position.get("reasoning", ""),
        }

    async def _get_signal(self, indicators: dict, price: float, balance: float) -> dict:
        """
        Use Claude AI or simplified rule-based signal for backtesting.
        For performance, use rule-based by default; optionally enable AI.
        """
        params = self.strategy.parameters or {}
        strategy_type = self.strategy.strategy_type.value

        if strategy_type == "trend_following":
            return self._trend_following_signal(indicators, price, params)
        elif strategy_type == "momentum":
            return self._momentum_signal(indicators, price, params)
        elif strategy_type == "mean_reversion":
            return self._mean_reversion_signal(indicators, price, params)
        else:
            # AI-powered (slower but more accurate)
            return await analyze_market(
                symbol=self.backtest.symbol,
                market_type="forex",
                strategy_name=self.strategy.name,
                strategy_params=params,
                indicators=indicators,
                recent_news=[],
                economic_events=[],
                current_price=price,
                bid=price * 0.9999,
                ask=price * 1.0001,
                spread=price * 0.0002,
                open_positions=0,
                max_positions=25,
                account_balance=balance,
                daily_pnl=0,
                max_daily_loss_pct=5.0,
                max_risk_pct=1.0,
            )

    def _trend_following_signal(self, ind: dict, price: float, params: dict) -> dict:
        ema_fast = ind.get("ema_21", price)
        ema_slow = ind.get("ema_50", price)
        adx = ind.get("adx", 0)
        rsi = ind.get("rsi_14", 50)

        if ema_fast > ema_slow and adx > params.get("adx_threshold", 25) and 40 < rsi < 70:
            sl = price * 0.988
            return {"signal": "buy", "confidence": 0.72, "stop_loss": sl, "take_profit": price + (price - sl) * 2}
        elif ema_fast < ema_slow and adx > params.get("adx_threshold", 25) and 30 < rsi < 60:
            sl = price * 1.012
            return {"signal": "sell", "confidence": 0.72, "stop_loss": sl, "take_profit": price - (sl - price) * 2}
        return {"signal": "hold", "confidence": 0.3}

    def _momentum_signal(self, ind: dict, price: float, params: dict) -> dict:
        rsi = ind.get("rsi_14", 50)
        macd_hist = ind.get("macd_histogram", 0)

        if rsi > 55 and macd_hist > 0:
            sl = price * 0.985
            return {"signal": "buy", "confidence": 0.74, "stop_loss": sl, "take_profit": price + (price - sl) * 2.5}
        elif rsi < 45 and macd_hist < 0:
            sl = price * 1.015
            return {"signal": "sell", "confidence": 0.74, "stop_loss": sl, "take_profit": price - (sl - price) * 2.5}
        return {"signal": "hold", "confidence": 0.3}

    def _mean_reversion_signal(self, ind: dict, price: float, params: dict) -> dict:
        rsi = ind.get("rsi_14", 50)
        bb_pos = ind.get("bb_position", 0.5)
        bb_lower = ind.get("bb_lower", price * 0.98)
        bb_upper = ind.get("bb_upper", price * 1.02)

        if rsi < 30 and bb_pos < 0.15:
            sl = bb_lower * 0.995
            return {"signal": "buy", "confidence": 0.70, "stop_loss": sl, "take_profit": ind.get("bb_middle", price * 1.01)}
        elif rsi > 70 and bb_pos > 0.85:
            sl = bb_upper * 1.005
            return {"signal": "sell", "confidence": 0.70, "stop_loss": sl, "take_profit": ind.get("bb_middle", price * 0.99)}
        return {"signal": "hold", "confidence": 0.3}

    async def _load_candles(self) -> pd.DataFrame:
        """Load candles from DB cache or return None to fetch from broker."""
        from sqlalchemy import select
        from db.models import MarketCandle

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(MarketCandle).where(
                    MarketCandle.symbol == self.backtest.symbol,
                    MarketCandle.timeframe == self.backtest.timeframe,
                    MarketCandle.open_time >= self.backtest.start_date,
                    MarketCandle.open_time <= self.backtest.end_date,
                ).order_by(MarketCandle.open_time)
            )
            candles = result.scalars().all()

        if not candles:
            return None

        df = pd.DataFrame([{
            "timestamp": c.open_time,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        } for c in candles])
        df.set_index("timestamp", inplace=True)
        return df
