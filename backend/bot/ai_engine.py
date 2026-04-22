"""
AI Engine — Technical analysis without Anthropic API.
Uses EMA + RSI + ADX combination.
"""
from loguru import logger
from typing import Optional


async def analyze_market(
    symbol: str,
    market_type: str,
    strategy_name: str,
    strategy_params: dict,
    indicators: dict,
    recent_news: list,
    economic_events: list,
    current_price: float,
    bid: float,
    ask: float,
    spread: float,
    open_positions: int,
    max_positions: int,
    account_balance: float,
    daily_pnl: float,
    max_daily_loss_pct: float,
    max_risk_pct: float,
    ai_system_prompt_override: Optional[str] = None,
) -> dict:
    try:
        rsi       = indicators.get("rsi_14", 50)
        adx       = indicators.get("adx", 0)
        ema_fast  = indicators.get("ema_21", 0)
        ema_slow  = indicators.get("ema_50", 0)
        ema_long  = indicators.get("ema_200", 0)
        atr       = indicators.get("atr_14", current_price * 0.01)
        htf_trend = indicators.get("htf_trend_4h", "neutral")

        signal     = "hold"
        confidence = 0.3
        reason     = "No signal"

        if adx < 15:
            return {"signal": "hold", "confidence": 0.2, "reason": "Low ADX — no trend",
                    "stop_loss": 0, "take_profit": 0, "reasoning": "Low ADX", "key_factors": []}

        ema_bull = ema_fast > ema_slow > 0
        ema_bear = ema_fast < ema_slow and ema_slow > 0
        htf_bull = htf_trend in ["strong_bull", "bull", "bullish"]
        htf_bear = htf_trend in ["strong_bear", "bear", "bearish"]

        # BUY
        if (
            ema_bull and
            htf_bull and
            28 < rsi < 68 and
            adx > 18 and
            (ema_long is None or ema_long == 0 or current_price > ema_long)
        ):
            signal     = "buy"
            confidence = min(0.95, 0.65 + (adx - 18) / 100 + (68 - rsi) / 200)
            reason     = f"EMA bull + HTF bull + RSI {rsi:.0f} + ADX {adx:.0f}"

        # SELL
        elif (
            ema_bear and
            htf_bear and
            30 < rsi < 72 and
            adx > 18 and
            (ema_long is None or ema_long == 0 or current_price < ema_long)
        ):
            signal     = "sell"
            confidence = min(0.95, 0.65 + (adx - 18) / 100 + (rsi - 30) / 200)
            reason     = f"EMA bear + HTF bear + RSI {rsi:.0f} + ADX {adx:.0f}"

        sl = tp = 0
        if signal == "buy":
            sl = round(current_price - atr * 2.0, 5)
            tp = round(current_price + atr * 4.0, 5)
        elif signal == "sell":
            sl = round(current_price + atr * 2.0, 5)
            tp = round(current_price - atr * 4.0, 5)

        if signal != "hold":
            logger.info(f"Signal {symbol}: {signal.upper()} | conf={confidence:.2f} | {reason}")

        return {
            "signal":     signal,
            "confidence": round(confidence, 2),
            "reason":     reason,
            "reasoning":  reason,
            "stop_loss":  sl,
            "take_profit": tp,
            "key_factors": [reason],
            "risk_level": "medium",
        }

    except Exception as e:
        logger.error(f"Analysis error for {symbol}: {e}")
        return {"signal": "hold", "confidence": 0.0, "reason": str(e),
                "reasoning": str(e), "stop_loss": 0, "take_profit": 0, "key_factors": []}


async def analyze_news_sentiment(symbol: str, currency: str, news_headlines: list) -> dict:
    return {"sentiment": "neutral", "score": 0.0, "key_themes": [], "impact_level": "low"}


async def generate_daily_market_brief(market_summary: dict, top_opportunities: list, economic_calendar: list) -> str:
    return "Market brief unavailable — Anthropic not connected."