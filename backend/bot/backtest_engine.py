"""
Backtest Engine
Runs strategy simulations on historical OHLCV data.
Produces equity curve, win rate, drawdown, Sharpe ratio.
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from bot.indicators import (
    calculate_indicators, detect_patterns,
    calculate_lot_size_from_risk, validate_sl_tp,
    apply_strategy_filters, get_mtf_trend,
)
from bot.ai_engine import analyze_market
from db.models import Backtest, Strategy
from db.database import AsyncSessionLocal


# Commission/spread costs per market type (round-trip, in price units per lot)
# These are conservative estimates for demo/retail accounts
COMMISSION_PER_LOT = {
    "forex":     7.0,    # ~$7 round-trip per standard lot (spread + commission)
    "crypto":    0.001,  # 0.1% taker fee per side → applied as price fraction
    "stock":     0.02,   # $0.02/share commission
    "index":     2.0,    # $2 per contract round-trip
    "commodity": 5.0,    # $5 per lot round-trip (gold, oil)
}

SPREAD_PIPS = {
    "forex":     1.5,    # 1.5 pip average spread
    "crypto":    0.05,   # 0.05% spread
    "stock":     0.01,   # $0.01 spread
    "index":     0.5,    # 0.5 point spread
    "commodity": 0.3,    # 0.3 point spread (gold ~$0.30)
}


class BacktestEngine:

    def __init__(self, backtest: Backtest, strategy: Strategy):
        self.backtest = backtest
        self.strategy = strategy
        # Infer market type from symbol for cost calculations
        self.market_type = self._infer_market_type(backtest.symbol)

    def _infer_market_type(self, symbol: str) -> str:
        symbol = symbol.upper()
        if any(c in symbol for c in ["BTC", "ETH", "BNB", "SOL", "XRP", "USDT"]):
            return "crypto"
        if any(c in symbol for c in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]):
            return "stock"
        if any(c in symbol for c in ["US500", "NAS100", "US30", "GER40", "SPX", "NDX"]):
            return "index"
        if any(c in symbol for c in ["XAUUSD", "XAGUSD", "USOIL", "UKOIL"]):
            return "commodity"
        return "forex"

    def _calculate_trade_cost(self, lot_size: float, price: float) -> float:
        """
        Calculate realistic round-trip trading cost (commission + spread).
        Deducted from PnL at trade open.
        """
        market = self.market_type
        commission = COMMISSION_PER_LOT.get(market, 7.0) * lot_size

        spread_pips = SPREAD_PIPS.get(market, 1.5)
        if market == "forex":
            spread_cost = spread_pips * 0.0001 * lot_size * 100_000
        elif market == "crypto":
            spread_cost = spread_pips * price * lot_size
        elif market in ("stock", "index", "commodity"):
            spread_cost = spread_pips * lot_size
        else:
            spread_cost = spread_pips * 0.0001 * lot_size * 100_000

        return round(commission + spread_cost, 4)

    async def run(self) -> dict:
        """
        Run full backtest simulation.
        Returns results dict with equity curve and trade log.
        """
        logger.info(f"Starting backtest {self.backtest.id} — {self.backtest.symbol} {self.backtest.timeframe}")

        df = await self._load_candles()
        if df is None or len(df) < 100:
            return {"error": "Insufficient historical data"}

        # Load higher timeframe candles for trend filter (4h)
        df_htf = await self._load_candles_htf()

        balance = self.backtest.initial_balance
        equity = balance
        peak_equity = balance
        max_drawdown = 0.0
        trades = []
        equity_curve = []
        open_position = None
        total_commission = 0.0

        strategy_type = self.strategy.strategy_type.value
        params = self.strategy.parameters or {}

        for i in range(50, len(df)):
            current_bar = df.iloc[i]
            window = df.iloc[max(0, i - 200):i + 1]
            indicators = calculate_indicators(window)
            patterns = detect_patterns(window)
            if patterns:
                indicators["patterns"] = patterns

            # Multi-timeframe: map 1h bar index to 4h bar index and get HTF trend
            if df_htf is not None and len(df_htf) >= 50:
                current_time = df.index[i]
                htf_window = df_htf[df_htf.index <= current_time].tail(200)
                if len(htf_window) >= 50:
                    htf_ind = calculate_indicators(htf_window)
                    htf_trend = get_mtf_trend(htf_ind)
                    indicators["htf_trend_4h"] = htf_trend
                    indicators["htf_adx_4h"] = htf_ind.get("adx")
                else:
                    indicators["htf_trend_4h"] = "neutral"
            else:
                indicators["htf_trend_4h"] = "neutral"

            # Apply strategy parameter filters
            indicators = apply_strategy_filters(indicators, strategy_type, params)

            price = current_bar["close"]
            atr = indicators.get("atr_14", price * 0.01)

            # Check if we have an open position to manage
            if open_position:
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
                    # Multi-timeframe filter: only trade in direction of HTF trend
                    htf_trend = indicators.get("htf_trend_4h", "neutral")
                    if htf_trend == "bullish" and signal["signal"] == "sell":
                        pass  # skip counter-trend trade
                    elif htf_trend == "bearish" and signal["signal"] == "buy":
                        pass  # skip counter-trend trade
                    else:
                        raw_sl = signal.get("stop_loss", 0)
                        raw_tp = signal.get("take_profit", 0)
                        stop_loss, take_profit = validate_sl_tp(
                            signal=signal["signal"],
                            entry_price=price,
                            stop_loss=raw_sl if raw_sl else (price - atr * 1.5 if signal["signal"] == "buy" else price + atr * 1.5),
                            take_profit=raw_tp if raw_tp else (price + atr * 3.0 if signal["signal"] == "buy" else price - atr * 3.0),
                            atr=atr,
                        )

                        lot_size = calculate_lot_size_from_risk(
                            account_balance=balance,
                            risk_pct=1.0,
                            entry_price=price,
                            stop_loss=stop_loss,
                            market_type=self.market_type,
                        )

                        # Deduct commission/spread cost immediately at open
                        trade_cost = self._calculate_trade_cost(lot_size, price)
                        balance -= trade_cost
                        total_commission += trade_cost

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
                            "commission": trade_cost,
                            "htf_trend": htf_trend,
                        }

            # Update equity curve
            unrealized = 0.0
            if open_position:
                if open_position["side"] == "buy":
                    unrealized = (price - open_position["entry_price"]) * open_position["lot_size"] * 100_000
                else:
                    unrealized = (open_position["entry_price"] - price) * open_position["lot_size"] * 100_000

            equity = balance + unrealized
            equity_curve.append(round(equity, 2))

            if equity > peak_equity:
                peak_equity = equity
            drawdown = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Close any remaining position at last bar
        if open_position:
            last_price = df.iloc[-1]["close"]
            result = self._close_position(open_position, last_price, len(df) - 1, df)
            balance += result["pnl"]
            trades.append(result)

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
            "total_commission": round(total_commission, 2),
            "trade_log": trades[:500],
            "equity_curve": equity_curve[::max(1, len(equity_curve) // 200)],
        }

        logger.info(
            f"Backtest complete: {results['total_trades']} trades, "
            f"WR={results['win_rate']}%, return={results['total_return_pct']}%, "
            f"commission={results['total_commission']}"
        )
        return results

    def _close_position(self, position: dict, exit_price: float, bar: int, df: pd.DataFrame) -> dict:
        if position["side"] == "buy":
            pnl = (exit_price - position["entry_price"]) * position["lot_size"] * 100_000
        else:
            pnl = (position["entry_price"] - exit_price) * position["lot_size"] * 100_000

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
            "commission": position.get("commission", 0),
            "htf_trend": position.get("htf_trend", "neutral"),
        }

    async def _get_signal(self, indicators: dict, price: float, balance: float) -> dict:
        """
        Use rule-based signals for backtesting (fast).
        Strategy parameters are now actually used in the calculations.
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
        # FIX: actually use strategy parameters in the logic
        ema_fast = ind.get("ema_21", price)
        ema_slow = ind.get("ema_50", price)
        ema_long = ind.get("ema_200", price)
        adx = ind.get("adx", 0)
        rsi = ind.get("rsi_14", 50)
        atr = ind.get("atr_14", price * 0.01)

        adx_threshold = params.get("adx_threshold", 25)
        rsi_min = params.get("rsi_min", 40)
        rsi_max = params.get("rsi_max", 70)
        min_confidence = params.get("min_confidence", 0.70)

        # Require price above long-term EMA for buys (trend filter)
        long_term_ok_buy = (ema_long is None) or (price > ema_long)
        long_term_ok_sell = (ema_long is None) or (price < ema_long)

        if (ema_fast > ema_slow and adx > adx_threshold
                and rsi_min < rsi < rsi_max and long_term_ok_buy):
            sl = price - atr * 1.5
            tp = price + atr * 3.0
            return {"signal": "buy", "confidence": min_confidence, "stop_loss": sl, "take_profit": tp}

        if (ema_fast < ema_slow and adx > adx_threshold
                and (100 - rsi_max) < rsi < (100 - rsi_min) and long_term_ok_sell):
            sl = price + atr * 1.5
            tp = price - atr * 3.0
            return {"signal": "sell", "confidence": min_confidence, "stop_loss": sl, "take_profit": tp}

        return {"signal": "hold", "confidence": 0.3}

    def _momentum_signal(self, ind: dict, price: float, params: dict) -> dict:
        rsi = ind.get("rsi_14", 50)
        macd_hist = ind.get("macd_histogram", 0)
        volume_ratio = ind.get("volume_ratio", 1.0)
        atr = ind.get("atr_14", price * 0.01)

        rsi_buy = params.get("rsi_buy_threshold", 55)
        rsi_sell = params.get("rsi_sell_threshold", 45)
        vol_min = params.get("volume_ratio_min", 1.5)
        min_confidence = params.get("min_confidence", 0.75)

        if rsi > rsi_buy and macd_hist > 0 and volume_ratio >= vol_min:
            sl = price - atr * 1.5
            tp = price + atr * 2.5
            return {"signal": "buy", "confidence": min_confidence, "stop_loss": sl, "take_profit": tp}

        if rsi < rsi_sell and macd_hist < 0 and volume_ratio >= vol_min:
            sl = price + atr * 1.5
            tp = price - atr * 2.5
            return {"signal": "sell", "confidence": min_confidence, "stop_loss": sl, "take_profit": tp}

        return {"signal": "hold", "confidence": 0.3}

    def _mean_reversion_signal(self, ind: dict, price: float, params: dict) -> dict:
        rsi = ind.get("rsi_14", 50)
        bb_pos = ind.get("bb_position", 0.5)
        bb_lower = ind.get("bb_lower", price * 0.98)
        bb_upper = ind.get("bb_upper", price * 1.02)
        bb_middle = ind.get("bb_middle", price)
        stoch_k = ind.get("stoch_k", 50)
        atr = ind.get("atr_14", price * 0.01)

        bb_buy = params.get("bb_position_buy", 0.15)
        bb_sell = params.get("bb_position_sell", 0.85)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_overbought = params.get("rsi_overbought", 70)
        stoch_oversold = params.get("stoch_oversold", 20)
        stoch_overbought = params.get("stoch_overbought", 80)
        min_confidence = params.get("min_confidence", 0.68)

        if rsi < rsi_oversold and bb_pos < bb_buy and stoch_k < stoch_oversold:
            sl = bb_lower - atr * 0.5
            tp = bb_middle
            return {"signal": "buy", "confidence": min_confidence, "stop_loss": sl, "take_profit": tp}

        if rsi > rsi_overbought and bb_pos > bb_sell and stoch_k > stoch_overbought:
            sl = bb_upper + atr * 0.5
            tp = bb_middle
            return {"signal": "sell", "confidence": min_confidence, "stop_loss": sl, "take_profit": tp}

        return {"signal": "hold", "confidence": 0.3}

    async def _load_candles(self) -> pd.DataFrame:
        """Load entry timeframe candles from DB."""
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

    async def _load_candles_htf(self) -> pd.DataFrame:
        """Load higher timeframe (4h) candles for trend filter."""
        from sqlalchemy import select
        from db.models import MarketCandle

        # Map entry timeframe to higher timeframe
        htf_map = {"1m": "15m", "5m": "1h", "15m": "1h", "30m": "4h", "1h": "4h", "4h": "1d", "1d": "1w"}
        htf = htf_map.get(self.backtest.timeframe, "4h")

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(MarketCandle).where(
                    MarketCandle.symbol == self.backtest.symbol,
                    MarketCandle.timeframe == htf,
                    MarketCandle.open_time >= self.backtest.start_date,
                    MarketCandle.open_time <= self.backtest.end_date,
                ).order_by(MarketCandle.open_time)
            )
            candles = result.scalars().all()

        if not candles or len(candles) < 50:
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
