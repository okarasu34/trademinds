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
    indicators["ema_9"] = round(close.ewm(span=9, adjust=False).mean().iloc[-1], 5)
    indicators["ema_21"] = round(close.ewm(span=21, adjust=False).mean().iloc[-1], 5)
    indicators["ema_50"] = round(close.ewm(span=50, adjust=False).mean().iloc[-1], 5)
    indicators["ema_200"] = round(close.ewm(span=200, adjust=False).mean().iloc[-1], 5) if len(df) >= 200 else None

    # ─── RSI ───
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    indicators["rsi_14"] = round(rsi.iloc[-1], 2)

    # ─── MACD ───
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
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
    bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
    bb_width = bb_range / sma20.iloc[-1] if sma20.iloc[-1] != 0 else 0
    indicators["bb_upper"] = round(bb_upper.iloc[-1], 5)
    indicators["bb_middle"] = round(sma20.iloc[-1], 5)
    indicators["bb_lower"] = round(bb_lower.iloc[-1], 5)
    indicators["bb_width"] = round(bb_width, 5)
    indicators["bb_position"] = round(
        (current_close - bb_lower.iloc[-1]) / bb_range, 3
    ) if bb_range > 0 else 0.5

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
    stoch_range = high14 - low14
    stoch_k = 100 * (close - low14) / stoch_range.replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()
    indicators["stoch_k"] = round(stoch_k.iloc[-1], 2)
    indicators["stoch_d"] = round(stoch_d.iloc[-1], 2)

    # ─── ADX (Wilder's smoothing — correct implementation) ───
    plus_dm = high.diff()
    minus_dm = -low.diff()
    # Only keep positive DM when it's greater than the other
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    period = 14
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    tr_smooth = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr_smooth.replace(0, np.nan))
    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    indicators["adx"] = round(adx.iloc[-1], 2)
    indicators["plus_di"] = round(plus_di.iloc[-1], 2)
    indicators["minus_di"] = round(minus_di.iloc[-1], 2)
    indicators["trend_strength"] = "strong" if adx.iloc[-1] > 25 else "weak"

    # ─── Volume analysis ───
    if volume.sum() > 0:
        vol_sma20 = volume.rolling(20).mean()
        vol_sma20_val = vol_sma20.iloc[-1]
        indicators["volume_ratio"] = round(volume.iloc[-1] / vol_sma20_val, 2) if vol_sma20_val > 0 else 1.0
        indicators["volume_trend"] = "above_avg" if volume.iloc[-1] > vol_sma20_val else "below_avg"

    # ─── Support & Resistance (pivot points) ───
    recent = df.tail(50)
    prev_high = recent["high"].iloc[-2]
    prev_low = recent["low"].iloc[-2]
    prev_close = recent["close"].iloc[-2]
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
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
    above_ema200 = (current_close > indicators["ema_200"]) if indicators.get("ema_200") else None
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
    """
    Calculate lot size based on risk percentage and stop loss distance.
    Uses pip-based calculation for forex pairs.
    """
    risk_amount = account_balance * (risk_pct / 100)
    price_diff = abs(entry_price - stop_loss)
    if price_diff == 0:
        return 0.01
    pips = price_diff / 0.0001  # For forex pairs
    lot_size = risk_amount / (pips * pip_value)
    return round(max(0.01, min(lot_size, 100.0)), 2)


def calculate_lot_size_from_risk(
    account_balance: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    market_type: str = "forex",
) -> float:
    """
    Accurate lot size calculation using stop loss distance and market type.
    This is the canonical lot size function — use this everywhere.

    For forex:  1 lot = 100,000 units, pip value ~$10 for USD pairs
    For crypto: position sizing based on price difference directly
    For stocks/indices: risk / price_diff gives share count
    """
    risk_amount = account_balance * (risk_pct / 100)
    price_diff = abs(entry_price - stop_loss)
    if price_diff == 0 or entry_price == 0:
        return 0.01

    if market_type == "forex":
        # Standard forex: 1 lot = 100,000 units
        lot_size = risk_amount / (price_diff * 100_000)
    elif market_type == "crypto":
        # Crypto: lot = risk / price_diff (in base currency units)
        lot_size = risk_amount / price_diff
    elif market_type in ("stock", "index"):
        # Stocks/indices: number of shares/contracts
        lot_size = risk_amount / price_diff
    elif market_type == "commodity":
        # Commodities (e.g. gold): similar to forex but contract size varies
        lot_size = risk_amount / (price_diff * 100)
    else:
        lot_size = risk_amount / (price_diff * 100_000)

    return round(max(0.001, min(lot_size, 100.0)), 3)


def validate_sl_tp(
    signal: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    atr: float,
) -> tuple[float, float]:
    """
    Validate and auto-correct stop loss and take profit values from AI.
    Ensures SL/TP are on the correct side of entry and within sane ATR multiples.
    Returns corrected (stop_loss, take_profit).
    """
    min_sl_distance = atr * 0.5
    max_sl_distance = atr * 10.0

    if signal == "buy":
        # SL must be below entry
        if stop_loss >= entry_price:
            stop_loss = entry_price - max(atr * 1.5, entry_price * 0.01)
        # TP must be above entry
        if take_profit <= entry_price:
            take_profit = entry_price + abs(entry_price - stop_loss) * 2.0
        # SL not too tight or too wide
        sl_dist = entry_price - stop_loss
        if sl_dist < min_sl_distance:
            stop_loss = entry_price - min_sl_distance
        elif sl_dist > max_sl_distance:
            stop_loss = entry_price - max_sl_distance

    elif signal == "sell":
        # SL must be above entry
        if stop_loss <= entry_price:
            stop_loss = entry_price + max(atr * 1.5, entry_price * 0.01)
        # TP must be below entry
        if take_profit >= entry_price:
            take_profit = entry_price - abs(stop_loss - entry_price) * 2.0
        # SL not too tight or too wide
        sl_dist = stop_loss - entry_price
        if sl_dist < min_sl_distance:
            stop_loss = entry_price + min_sl_distance
        elif sl_dist > max_sl_distance:
            stop_loss = entry_price + max_sl_distance

    return round(stop_loss, 5), round(take_profit, 5)


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
    if body > 0 and lower_wick > 2 * body and upper_wick < body:
        patterns.append("hammer")

    # Shooting star
    if body > 0 and upper_wick > 2 * body and lower_wick < body:
        patterns.append("shooting_star")

    return patterns


def apply_strategy_filters(indicators: dict, strategy_type: str, params: dict) -> dict:
    """
    Apply strategy-specific parameter filters to indicators.
    Returns a filtered/annotated copy of indicators so the AI and rule-based
    signals both use the exact thresholds defined in the strategy parameters.
    """
    filtered = dict(indicators)

    if strategy_type == "trend_following":
        filtered["_filter_adx_threshold"] = params.get("adx_threshold", 25)
        filtered["_filter_rsi_min"] = params.get("rsi_min", 40)
        filtered["_filter_rsi_max"] = params.get("rsi_max", 70)
        filtered["_filter_min_confidence"] = params.get("min_confidence", 0.70)
        # Annotate whether current values pass the strategy's own thresholds
        filtered["_strategy_adx_ok"] = indicators.get("adx", 0) > params.get("adx_threshold", 25)
        filtered["_strategy_rsi_ok"] = params.get("rsi_min", 40) < indicators.get("rsi_14", 50) < params.get("rsi_max", 70)

    elif strategy_type == "momentum":
        filtered["_filter_rsi_buy"] = params.get("rsi_buy_threshold", 55)
        filtered["_filter_rsi_sell"] = params.get("rsi_sell_threshold", 45)
        filtered["_filter_volume_ratio_min"] = params.get("volume_ratio_min", 1.5)
        filtered["_filter_min_confidence"] = params.get("min_confidence", 0.75)
        filtered["_strategy_volume_ok"] = indicators.get("volume_ratio", 0) >= params.get("volume_ratio_min", 1.5)
        filtered["_strategy_macd_confirm"] = params.get("macd_confirm", True)

    elif strategy_type == "mean_reversion":
        filtered["_filter_bb_buy"] = params.get("bb_position_buy", 0.15)
        filtered["_filter_bb_sell"] = params.get("bb_position_sell", 0.85)
        filtered["_filter_rsi_oversold"] = params.get("rsi_oversold", 30)
        filtered["_filter_rsi_overbought"] = params.get("rsi_overbought", 70)
        filtered["_filter_stoch_oversold"] = params.get("stoch_oversold", 20)
        filtered["_filter_stoch_overbought"] = params.get("stoch_overbought", 80)
        filtered["_filter_min_confidence"] = params.get("min_confidence", 0.68)
        filtered["_strategy_bb_oversold"] = indicators.get("bb_position", 0.5) < params.get("bb_position_buy", 0.15)
        filtered["_strategy_bb_overbought"] = indicators.get("bb_position", 0.5) > params.get("bb_position_sell", 0.85)

    elif strategy_type == "sentiment":
        filtered["_filter_sentiment_threshold"] = params.get("sentiment_threshold", 0.6)
        filtered["_filter_news_weight"] = params.get("news_weight", 0.7)
        filtered["_filter_technical_weight"] = params.get("technical_weight", 0.3)
        filtered["_filter_min_confidence"] = params.get("min_confidence", 0.72)

    elif strategy_type == "news_based":
        filtered["_filter_entry_delay_seconds"] = params.get("entry_delay_seconds", 30)
        filtered["_filter_max_spread_pips"] = params.get("max_spread_pips", 5)
        filtered["_filter_impact_filter"] = params.get("impact_filter", ["high"])
        filtered["_filter_min_confidence"] = params.get("min_confidence", 0.73)

    return filtered


def get_mtf_trend(htf_indicators: dict) -> str:
    """
    Determine higher timeframe trend direction.
    Returns 'bullish', 'bearish', or 'neutral'.
    """
    ema_50 = htf_indicators.get("ema_50")
    ema_200 = htf_indicators.get("ema_200")
    adx = htf_indicators.get("adx", 0)
    price = htf_indicators.get("current_price", 0)

    if not ema_50 or not price:
        return "neutral"

    above_ema50 = price > ema_50
    above_ema200 = (price > ema_200) if ema_200 else None
    strong_trend = adx > 20

    if above_ema50 and (above_ema200 is None or above_ema200) and strong_trend:
        return "bullish"
    if not above_ema50 and (above_ema200 is None or not above_ema200) and strong_trend:
        return "bearish"
    return "neutral"
