import pandas as pd
import numpy as np
from typing import Optional


def calculate_indicators(df: pd.DataFrame) -> dict:
    """
    Calculate comprehensive technical indicators from OHLCV data.
    df must have columns: open, high, low, close, volume
    Returns dict of indicator values (latest bar).
    """
    if len(df) < 50:
        return {}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"] if "volume" in df.columns else pd.Series([0] * len(df))

    indicators = {}

    # ─── Moving Averages ───
    indicators["sma_20"] = round(close.rolling(20).mean().iloc[-1], 5)
    indicators["sma_50"] = round(close.rolling(50).mean().iloc[-1], 5)
    indicators["ema_9"] = round(close.ewm(span=9).mean().iloc[-1], 5)
    indicators["ema_21"] = round(close.ewm(span=21).mean().iloc[-1], 5)
    indicators["ema_50"] = round(close.ewm(span=50).mean().iloc[-1], 5)
    indicators["ema_200"] = round(close.ewm(span=200).mean().iloc[-1], 5) if len(df) >= 200 else None

    # ─── RSI ───
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    indicators["rsi_14"] = round(rsi.iloc[-1], 2)

    # ─── MACD ───
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    histogram = macd_line - signal_line
    indicators["macd"] = round(macd_line.iloc[-1], 5)
    indicators["macd_signal"] = round(signal_line.iloc[-1], 5)
    indicators["macd_histogram"] = round(histogram.iloc[-1], 5)
    indicators["macd_crossover"] = "bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "bearish"

    # ─── Bollinger Bands ───
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + (2 * std20)
    bb_lower = sma20 - (2 * std20)
    current_close = close.iloc[-1]
    bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / sma20.iloc[-1]
    indicators["bb_upper"] = round(bb_upper.iloc[-1], 5)
    indicators["bb_middle"] = round(sma20.iloc[-1], 5)
    indicators["bb_lower"] = round(bb_lower.iloc[-1], 5)
    indicators["bb_width"] = round(bb_width, 5)
    indicators["bb_position"] = round(
        (current_close - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]), 3
    )

    # ─── ATR (Average True Range) ───
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    indicators["atr_14"] = round(tr.rolling(14).mean().iloc[-1], 5)

    # ─── Stochastic ───
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14 + 1e-10)
    stoch_d = stoch_k.rolling(3).mean()
    indicators["stoch_k"] = round(stoch_k.iloc[-1], 2)
    indicators["stoch_d"] = round(stoch_d.iloc[-1], 2)

    # ─── ADX (Average Directional Index) ───
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr_smooth = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / tr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / tr_smooth.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(14).mean()
    indicators["adx"] = round(adx.iloc[-1], 2)
    indicators["plus_di"] = round(plus_di.iloc[-1], 2)
    indicators["minus_di"] = round(minus_di.iloc[-1], 2)
    indicators["trend_strength"] = "strong" if adx.iloc[-1] > 25 else "weak"

    # ─── Volume analysis ───
    if volume.sum() > 0:
        vol_sma20 = volume.rolling(20).mean()
        indicators["volume_ratio"] = round(volume.iloc[-1] / vol_sma20.iloc[-1], 2)
        indicators["volume_trend"] = "above_avg" if volume.iloc[-1] > vol_sma20.iloc[-1] else "below_avg"

    # ─── Support & Resistance ───
    recent = df.tail(50)
    pivot = (recent["high"].iloc[-1] + recent["low"].iloc[-1] + recent["close"].iloc[-1]) / 3
    r1 = 2 * pivot - recent["low"].iloc[-1]
    s1 = 2 * pivot - recent["high"].iloc[-1]
    r2 = pivot + (recent["high"].iloc[-1] - recent["low"].iloc[-1])
    s2 = pivot - (recent["high"].iloc[-1] - recent["low"].iloc[-1])
    indicators["pivot"] = round(pivot, 5)
    indicators["resistance_1"] = round(r1, 5)
    indicators["resistance_2"] = round(r2, 5)
    indicators["support_1"] = round(s1, 5)
    indicators["support_2"] = round(s2, 5)

    # ─── Price Action ───
    indicators["current_price"] = round(current_close, 5)
    indicators["price_change_1bar"] = round(close.pct_change().iloc[-1] * 100, 4)
    indicators["price_change_5bar"] = round(close.pct_change(5).iloc[-1] * 100, 4)
    indicators["price_change_20bar"] = round(close.pct_change(20).iloc[-1] * 100, 4)

    # ─── Trend direction ───
    above_ema50 = current_close > indicators["ema_50"]
    above_ema200 = current_close > indicators.get("ema_200", 0) if indicators.get("ema_200") else None
    indicators["trend_direction"] = "bullish" if above_ema50 else "bearish"
    if above_ema200 is not None:
        indicators["long_term_trend"] = "bullish" if above_ema200 else "bearish"

    # ─── Overbought/Oversold ───
    indicators["rsi_condition"] = (
        "overbought" if indicators["rsi_14"] > 70
        else "oversold" if indicators["rsi_14"] < 30
        else "neutral"
    )

    return indicators


def calculate_position_size(
    account_balance: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    pip_value: float = 10.0,
) -> float:
    """Calculate lot size based on risk percentage."""
    risk_amount = account_balance * (risk_pct / 100)
    price_diff = abs(entry_price - stop_loss)
    if price_diff == 0:
        return 0.01
    pips = price_diff / 0.0001  # For forex pairs
    lot_size = risk_amount / (pips * pip_value)
    return round(max(0.01, min(lot_size, 100.0)), 2)


def detect_patterns(df: pd.DataFrame) -> list[str]:
    """Simple candlestick pattern detection."""
    patterns = []
    if len(df) < 3:
        return patterns

    o, h, l, c = (
        df["open"].values, df["high"].values,
        df["low"].values, df["close"].values
    )

    # Doji
    body = abs(c[-1] - o[-1])
    total_range = h[-1] - l[-1]
    if total_range > 0 and body / total_range < 0.1:
        patterns.append("doji")

    # Bullish engulfing
    if (c[-2] < o[-2] and c[-1] > o[-1] and
            c[-1] > o[-2] and o[-1] < c[-2]):
        patterns.append("bullish_engulfing")

    # Bearish engulfing
    if (c[-2] > o[-2] and c[-1] < o[-1] and
            c[-1] < o[-2] and o[-1] > c[-2]):
        patterns.append("bearish_engulfing")

    # Hammer
    lower_wick = min(o[-1], c[-1]) - l[-1]
    upper_wick = h[-1] - max(o[-1], c[-1])
    if lower_wick > 2 * body and upper_wick < body:
        patterns.append("hammer")

    # Shooting star
    if upper_wick > 2 * body and lower_wick < body:
        patterns.append("shooting_star")

    return patterns
