import anthropic
from core.config import settings
from loguru import logger
from typing import Optional
import json
import pandas as pd
import numpy as np


client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)


SYSTEM_PROMPT = """You are TradeMinds AI, an expert autonomous trading system.
You analyze financial markets across Forex, Crypto, Commodities, Stocks, and Indices.

Your role:
- Analyze technical indicators, price action, market sentiment, and news events
- Generate precise BUY, SELL, or HOLD signals with confidence scores
- Provide clear reasoning for every decision
- Apply the assigned trading strategy strictly
- Respect risk parameters provided

Always respond in valid JSON only. No markdown, no explanation outside JSON.

JSON format:
{
  "signal": "buy" | "sell" | "hold",
  "confidence": 0.0-1.0,
  "entry_price": float or null,
  "stop_loss": float,
  "take_profit": float,
  "lot_size_pct": 0.0-1.0,
  "reasoning": "detailed explanation",
  "key_factors": ["factor1", "factor2"],
  "risk_level": "low" | "medium" | "high",
  "expected_duration_hours": float
}
"""


async def analyze_market(
    symbol: str,
    market_type: str,
    strategy_name: str,
    strategy_params: dict,
    indicators: dict,
    recent_news: list[str],
    economic_events: list[dict],
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
    """
    Core AI analysis function.
    Returns structured signal with entry, SL, TP, reasoning.
    """

    system = ai_system_prompt_override or SYSTEM_PROMPT

    user_message = f"""
MARKET ANALYSIS REQUEST
=======================
Symbol: {symbol}
Market Type: {market_type}
Current Price: {current_price}
Bid: {bid} | Ask: {ask} | Spread: {spread:.5f}

STRATEGY
--------
Name: {strategy_name}
Parameters: {json.dumps(strategy_params, indent=2)}

TECHNICAL INDICATORS
--------------------
{json.dumps(indicators, indent=2)}

RECENT NEWS & SENTIMENT
-----------------------
{chr(10).join(f"- {n}" for n in recent_news) if recent_news else "No recent news"}

UPCOMING ECONOMIC EVENTS (next 4 hours)
----------------------------------------
{json.dumps(economic_events, indent=2) if economic_events else "None"}

ACCOUNT & RISK STATUS
---------------------
Account Balance: {account_balance:.2f} USD
Daily P&L: {daily_pnl:.2f} USD
Daily Loss Limit: {max_daily_loss_pct}% of balance
Max Risk Per Trade: {max_risk_pct}%
Open Positions: {open_positions}/{max_positions}

TASK
----
Analyze all data and provide a trading signal.
If open positions are at max, signal HOLD unless a critical exit signal exists.
If daily loss limit is near (within 1%), be very conservative.
Adjust lot_size_pct based on confidence and risk environment.
"""

    try:
        response = await client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=settings.ANTHROPIC_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )

        raw = response.content[0].text.strip()
        result = json.loads(raw)
        logger.info(f"AI signal for {symbol}: {result['signal']} (confidence: {result['confidence']:.2f})")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"AI response JSON parse error for {symbol}: {e}")
        return {"signal": "hold", "confidence": 0.0, "reasoning": "AI parse error", "risk_level": "high"}
    except Exception as e:
        logger.error(f"AI analysis error for {symbol}: {e}")
        return {"signal": "hold", "confidence": 0.0, "reasoning": str(e), "risk_level": "high"}


async def analyze_news_sentiment(
    symbol: str,
    currency: str,
    news_headlines: list[str],
) -> dict:
    """Analyze news sentiment for a specific symbol/currency."""

    response = await client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=512,
        system="You are a financial news sentiment analyzer. Respond in JSON only.",
        messages=[{
            "role": "user",
            "content": f"""
Analyze sentiment for {symbol} ({currency}) based on these headlines:
{chr(10).join(f"- {h}" for h in news_headlines)}

Respond as JSON:
{{
  "sentiment": "bullish" | "bearish" | "neutral",
  "score": -1.0 to 1.0,
  "key_themes": ["theme1"],
  "impact_level": "low" | "medium" | "high"
}}
"""
        }]
    )

    try:
        return json.loads(response.content[0].text.strip())
    except Exception:
        return {"sentiment": "neutral", "score": 0.0, "key_themes": [], "impact_level": "low"}


async def generate_daily_market_brief(
    market_summary: dict,
    top_opportunities: list[dict],
    economic_calendar: list[dict],
) -> str:
    """Generate a daily AI market brief for the dashboard."""

    response = await client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=1000,
        system="You are a professional market analyst. Write concise, insightful daily briefs.",
        messages=[{
            "role": "user",
            "content": f"""
Write a professional daily market brief (max 300 words) covering:

Market Summary: {json.dumps(market_summary)}
Top Opportunities: {json.dumps(top_opportunities)}
Today's Key Events: {json.dumps(economic_calendar)}

Focus on actionable insights. Be concise and professional.
"""
        }]
    )

    return response.content[0].text.strip()
