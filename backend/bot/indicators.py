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

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"] if "open" in df.columns else close
    volume = df["volume"] if "volume" in df.columns else pd.Series([1] * len(df))

    indicators = {}

    # ─── Moving Averages ───
    indicators["sma_20"]  = round(close.rolling(20).mean().iloc[-1], 5)
    indicators["sma_50"]  = round(close.rolling(50).mean().iloc[-1], 5)
    indicators["ema_9"]   = round(close.ewm(span=9,   adjust=False).mean().iloc[-1], 5)
    indicators["ema_13"]  = round(close.ewm(span=13,  adjust=False).mean().iloc[-1], 5)
    indicators["ema_21"]  = round(close.ewm(span=21,  adjust=False).mean().iloc[-1], 5)
    indicators["ema_50"]  = round(close.ewm(span=50,  adjust=False).mean().iloc[-1], 5)
    indicators["ema_89"]  = round(close.ewm(span=89,  adjust=False).mean().iloc[-1], 5)
    indicators["ema_144"] = round(close.ewm(span=144, adjust=False).mean().iloc[-1], 5) if len(df) >= 144 else None
    indicators["ema_200"] = round(close.ewm(span=200, adjust=False).mean().iloc[-1], 5) if len(df) >= 200 else None

    # ─── RSI (14 ve 13) ───
    def calc_rsi(src, period):
        delta = src.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    rsi14 = calc_rsi(close, 14)
    rsi13 = calc_rsi(close, 13)
    indicators["rsi_14"] = round(rsi14.iloc[-1], 2)
    indicators["rsi_13"] = round(rsi13.iloc[-1], 2)

    # ─── MACD (12/26/9 ve 13/21/8) ───
    ema12      = close.ewm(span=12, adjust=False).mean()
    ema26      = close.ewm(span=26, adjust=False).mean()
    macd_line  = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram  = macd_line - signal_line
    indicators["macd"]           = round(macd_line.iloc[-1], 5)
    indicators["macd_signal"]    = round(signal_line.iloc[-1], 5)
    indicators["macd_histogram"] = round(histogram.iloc[-1], 5)
    indicators["macd_crossover"] = "bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "bearish"

    # MACD 13/21/8 (strateji parametrelerine göre)
    ema13s      = close.ewm(span=13, adjust=False).mean()
    ema21s      = close.ewm(span=21, adjust=False).mean()
    macd13      = ema13s - ema21s
    signal13    = macd13.ewm(span=8, adjust=False).mean()
    hist13      = macd13 - signal13
    indicators["macd13"]           = round(macd13.iloc[-1], 5)
    indicators["macd13_signal"]    = round(signal13.iloc[-1], 5)
    indicators["macd13_histogram"] = round(hist13.iloc[-1], 5)
    indicators["macd13_crossover"] = "bullish" if macd13.iloc[-1] > signal13.iloc[-1] else "bearish"

    # MACD crossover (önceki bar ile karşılaştır — crossover tespiti)
    indicators["macd13_cross_up"]   = bool(macd13.iloc[-1] > signal13.iloc[-1] and macd13.iloc[-2] <= signal13.iloc[-2])
    indicators["macd13_cross_down"] = bool(macd13.iloc[-1] < signal13.iloc[-1] and macd13.iloc[-2] >= signal13.iloc[-2])

    # ─── Bollinger Bands ───
    sma20   = close.rolling(20).mean()
    std20   = close.rolling(20).std()
    bb_upper = sma20 + (2 * std20)
    bb_lower = sma20 - (2 * std20)
    current_close = close.iloc[-1]
    bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
    bb_width = bb_range / sma20.iloc[-1] if sma20.iloc[-1] != 0 else 0
    indicators["bb_upper"]    = round(bb_upper.iloc[-1], 5)
    indicators["bb_middle"]   = round(sma20.iloc[-1], 5)
    indicators["bb_lower"]    = round(bb_lower.iloc[-1], 5)
    indicators["bb_width"]    = round(bb_width, 5)
    indicators["bb_position"] = round(
        (current_close - bb_lower.iloc[-1]) / bb_range, 3
    ) if bb_range > 0 else 0.5

    # ─── ATR ───
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    indicators["atr_14"] = round(tr.rolling(14).mean().iloc[-1], 5)
    indicators["atr_13"] = round(tr.rolling(13).mean().iloc[-1], 5)

    # ─── MFI (Money Flow Index) — Alpha Trend için ───
    typical_price = (high + low + close) / 3
    raw_mf        = typical_price * volume
    pos_mf = raw_mf.where(typical_price > typical_price.shift(1), 0)
    neg_mf = raw_mf.where(typical_price < typical_price.shift(1), 0)
    pos_mf_sum = pos_mf.rolling(13).sum()
    neg_mf_sum = neg_mf.rolling(13).sum()
    mfr = pos_mf_sum / neg_mf_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    indicators["mfi_13"] = round(mfi.iloc[-1], 2)

    # ─── Alpha Trend ───
    # ATR tabanlı dinamik trend çizgisi (MFI >= 40 ise bullish)
    atr13        = tr.rolling(13).mean()
    coeff        = 1.1
    up_t         = low  - atr13 * coeff
    down_t       = high + atr13 * coeff
    alpha_trend  = pd.Series(np.nan, index=close.index)
    mfi_series   = mfi

    for i in range(1, len(close)):
        if mfi_series.iloc[i] >= 40:
            val = up_t.iloc[i]
            prev = alpha_trend.iloc[i-1] if not np.isnan(alpha_trend.iloc[i-1]) else val
            alpha_trend.iloc[i] = max(val, prev)
        else:
            val = down_t.iloc[i]
            prev = alpha_trend.iloc[i-1] if not np.isnan(alpha_trend.iloc[i-1]) else val
            alpha_trend.iloc[i] = min(val, prev)

    indicators["alpha_trend"]       = round(alpha_trend.iloc[-1], 5)
    indicators["alpha_trend_prev"]  = round(alpha_trend.iloc[-3], 5)  # 2 bar öncesi (crossover için)
    indicators["alpha_trend_cross_up"]   = bool(alpha_trend.iloc[-1] > alpha_trend.iloc[-3] and alpha_trend.iloc[-2] <= alpha_trend.iloc[-4]) if len(alpha_trend) >= 4 else False
    indicators["alpha_trend_cross_down"] = bool(alpha_trend.iloc[-1] < alpha_trend.iloc[-3] and alpha_trend.iloc[-2] >= alpha_trend.iloc[-4]) if len(alpha_trend) >= 4 else False

    # ─── Stochastic ───
    low14   = low.rolling(14).min()
    high14  = high.rolling(14).max()
    stoch_range = high14 - low14
    stoch_k = 100 * (close - low14) / stoch_range.replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()
    indicators["stoch_k"] = round(stoch_k.iloc[-1], 2)
    indicators["stoch_d"] = round(stoch_d.iloc[-1], 2)

    # ─── ADX ───
    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm  = plus_dm.where((plus_dm > minus_dm)  & (plus_dm > 0),  0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    period   = 14
    tr_smooth  = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di    = 100 * (plus_dm.ewm(alpha=1/period,  adjust=False).mean() / tr_smooth.replace(0, np.nan))
    minus_di   = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / tr_smooth.replace(0, np.nan))
    di_sum     = plus_di + minus_di
    dx         = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    adx        = dx.ewm(alpha=1/period, adjust=False).mean()
    indicators["adx"]            = round(adx.iloc[-1], 2)
    indicators["plus_di"]        = round(plus_di.iloc[-1], 2)
    indicators["minus_di"]       = round(minus_di.iloc[-1], 2)
    indicators["trend_strength"] = "strong" if adx.iloc[-1] > 25 else "weak"

    # ─── Fibonacci Seviyeleri (144 periyot) ───
    fibo_period = 144 if len(df) >= 144 else len(df)
    fibo_high   = high.tail(fibo_period).max()
    fibo_low    = low.tail(fibo_period).min()
    fibo_range  = fibo_high - fibo_low
    if fibo_range > 0:
        indicators["fibo_0"]    = round(fibo_low, 5)
        indicators["fibo_236"]  = round(fibo_low + fibo_range * 0.236, 5)
        indicators["fibo_382"]  = round(fibo_low + fibo_range * 0.382, 5)
        indicators["fibo_500"]  = round(fibo_low + fibo_range * 0.500, 5)
        indicators["fibo_618"]  = round(fibo_low + fibo_range * 0.618, 5)
        indicators["fibo_786"]  = round(fibo_low + fibo_range * 0.786, 5)
        indicators["fibo_1000"] = round(fibo_high, 5)

        # Fiyatın hangi Fibonacci bölgesine yakın olduğunu tespit et
        price      = current_close
        fibo_levels = [
            ("fibo_0",   indicators["fibo_0"]),
            ("fibo_236", indicators["fibo_236"]),
            ("fibo_382", indicators["fibo_382"]),
            ("fibo_500", indicators["fibo_500"]),
            ("fibo_618", indicators["fibo_618"]),
            ("fibo_786", indicators["fibo_786"]),
            ("fibo_1000",indicators["fibo_1000"]),
        ]
        threshold = fibo_range * 0.02  # %2 yakınlık
        indicators["near_fibo_level"] = None
        for name, level in fibo_levels:
            if abs(price - level) <= threshold:
                indicators["near_fibo_level"] = name
                break

    # ─── RSI Diverjans (basit tespit) ───
    # Son 8 bar içinde fiyat yeni dip ama RSI daha yüksek → bullish diverjans
    # Son 8 bar içinde fiyat yeni zirve ama RSI daha düşük → bearish diverjans
    lookback = min(8, len(df) - 1)
    price_low_now  = close.iloc[-1]
    price_low_prev = close.iloc[-lookback-1:-1].min()
    rsi_now        = rsi13.iloc[-1]
    rsi_prev       = rsi13.iloc[-lookback-1:-1].min()

    price_high_now  = close.iloc[-1]
    price_high_prev = close.iloc[-lookback-1:-1].max()
    rsi_high_now    = rsi13.iloc[-1]
    rsi_high_prev   = rsi13.iloc[-lookback-1:-1].max()

    indicators["rsi_bullish_divergence"] = bool(
        price_low_now < price_low_prev and rsi_now > rsi_prev
    )
    indicators["rsi_bearish_divergence"] = bool(
        price_high_now > price_high_prev and rsi_high_now < rsi_high_prev
    )

    # MACD diverjans
    macd_now  = hist13.iloc[-1]
    macd_prev = hist13.iloc[-lookback-1:-1].min()
    indicators["macd_bullish_divergence"] = bool(
        price_low_now < price_low_prev and macd_now > macd_prev
    )
    indicators["macd_bearish_divergence"] = bool(
        price_high_now > price_high_prev and hist13.iloc[-1] < hist13.iloc[-lookback-1:-1].max()
    )

    # ─── Volume Profile / POC (basit yaklaşım) ───
    poc_period = min(89, len(df))
    poc_df     = df.tail(poc_period)
    price_min  = poc_df["low"].min()
    price_max  = poc_df["high"].max()
    if price_max > price_min:
        bins       = 21
        bin_size   = (price_max - price_min) / bins
        max_vol    = 0
        poc_level  = (price_min + price_max) / 2
        for i in range(bins):
            bin_low  = price_min + i * bin_size
            bin_high = bin_low + bin_size
            mask     = (poc_df["close"] >= bin_low) & (poc_df["close"] < bin_high)
            bin_vol  = poc_df.loc[mask, "volume"].sum()
            if bin_vol > max_vol:
                max_vol   = bin_vol
                poc_level = (bin_low + bin_high) / 2
        indicators["poc_level"]     = round(poc_level, 5)
        poc_proximity = abs(current_close - poc_level) / current_close * 100
        indicators["poc_proximity_pct"] = round(poc_proximity, 3)
        indicators["near_poc"] = bool(poc_proximity < 0.5)

    # ─── Order Block (basit tespit) ───
    ob_period = min(8, len(df) - 1)
    # Son ob_period içinde en güçlü bearish mum → bearish OB
    recent_df  = df.tail(ob_period + 1).iloc[:-1]
    bull_candles = recent_df[recent_df["close"] > recent_df["open"]]
    bear_candles = recent_df[recent_df["close"] < recent_df["open"]]

    if not bear_candles.empty:
        biggest_bear    = bear_candles.loc[bear_candles["high"] - bear_candles["low"] == (bear_candles["high"] - bear_candles["low"]).max()]
        ob_bear_high    = float(biggest_bear["high"].iloc[-1])
        ob_bear_low     = float(biggest_bear["low"].iloc[-1])
        indicators["bearish_ob_high"] = round(ob_bear_high, 5)
        indicators["bearish_ob_low"]  = round(ob_bear_low, 5)
        indicators["in_bearish_ob"]   = bool(ob_bear_low <= current_close <= ob_bear_high)
    else:
        indicators["in_bearish_ob"] = False

    if not bull_candles.empty:
        biggest_bull    = bull_candles.loc[bull_candles["high"] - bull_candles["low"] == (bull_candles["high"] - bull_candles["low"]).max()]
        ob_bull_high    = float(biggest_bull["high"].iloc[-1])
        ob_bull_low     = float(biggest_bull["low"].iloc[-1])
        indicators["bullish_ob_high"] = round(ob_bull_high, 5)
        indicators["bullish_ob_low"]  = round(ob_bull_low, 5)
        indicators["in_bullish_ob"]   = bool(ob_bull_low <= current_close <= ob_bull_high)
    else:
        indicators["in_bullish_ob"] = False

    # ─── Fair Value Gap (FVG) ───
    atr_val = tr.rolling(14).mean().iloc[-1]
    # Bullish FVG: low[i] > high[i-2] (boşluk)
    bull_fvg = bool(low.iloc[-1] > high.iloc[-3] and (low.iloc[-1] - high.iloc[-3]) > atr_val * 0.5) if len(df) >= 3 else False
    # Bearish FVG: high[i] < low[i-2]
    bear_fvg = bool(high.iloc[-1] < low.iloc[-3] and (low.iloc[-3] - high.iloc[-1]) > atr_val * 0.5) if len(df) >= 3 else False
    indicators["bull_fvg"] = bull_fvg
    indicators["bear_fvg"] = bear_fvg

    # ─── Volume analysis ───
    if volume.sum() > 0:
        vol_sma20     = volume.rolling(20).mean()
        vol_sma20_val = vol_sma20.iloc[-1]
        indicators["volume_ratio"] = round(volume.iloc[-1] / vol_sma20_val, 2) if vol_sma20_val > 0 else 1.0
        indicators["volume_trend"] = "above_avg" if volume.iloc[-1] > vol_sma20_val else "below_avg"

    # ─── Pivot Points ───
    recent   = df.tail(50)
    prev_high  = recent["high"].iloc[-2]
    prev_low   = recent["low"].iloc[-2]
    prev_close = recent["close"].iloc[-2]
    pivot      = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    indicators["pivot"]        = round(pivot, 5)
    indicators["resistance_1"] = round(r1, 5)
    indicators["resistance_2"] = round(r2, 5)
    indicators["support_1"]    = round(s1, 5)
    indicators["support_2"]    = round(s2, 5)

    # ─── Price Action ───
    indicators["current_price"]      = round(current_close, 5)
    indicators["price_change_1bar"]  = round(close.pct_change().iloc[-1] * 100, 4)
    indicators["price_change_5bar"]  = round(close.pct_change(5).iloc[-1] * 100, 4)
    indicators["price_change_20bar"] = round(close.pct_change(20).iloc[-1] * 100, 4)

    # ─── Trend direction ───
    above_ema50  = current_close > indicators["ema_50"]
    above_ema89  = current_close > indicators["ema_89"]
    above_ema200 = (current_close > indicators["ema_200"]) if indicators.get("ema_200") else None
    indicators["trend_direction"]  = "bullish" if above_ema50 else "bearish"
    indicators["ema13_above_ema21"] = bool(indicators["ema_13"] > indicators["ema_21"])
    indicators["ema13_above_ema89"] = bool(indicators["ema_13"] > indicators["ema_89"])
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
    risk_amount = account_balance * (risk_pct / 100)
    price_diff  = abs(entry_price - stop_loss)
    if price_diff == 0:
        return 0.01
    pips     = price_diff / 0.0001
    lot_size = risk_amount / (pips * pip_value)
    return round(max(0.01, min(lot_size, 100.0)), 2)


def calculate_lot_size_from_risk(
    account_balance: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    market_type: str = "forex",
) -> float:
    risk_amount = account_balance * (risk_pct / 100)
    price_diff  = abs(entry_price - stop_loss)
    if price_diff == 0 or entry_price == 0:
        return 0.01
    if market_type == "forex":
        lot_size = risk_amount / (price_diff * 100_000)
    elif market_type == "crypto":
        lot_size = risk_amount / price_diff
    elif market_type in ("stock", "index"):
        lot_size = risk_amount / price_diff
    elif market_type == "commodity":
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
    min_sl_distance = atr * 0.5
    max_sl_distance = atr * 10.0
    if signal == "buy":
        if stop_loss >= entry_price:
            stop_loss = entry_price - max(atr * 1.5, entry_price * 0.01)
        if take_profit <= entry_price:
            take_profit = entry_price + abs(entry_price - stop_loss) * 2.0
        sl_dist = entry_price - stop_loss
        if sl_dist < min_sl_distance:
            stop_loss = entry_price - min_sl_distance
        elif sl_dist > max_sl_distance:
            stop_loss = entry_price - max_sl_distance
    elif signal == "sell":
        if stop_loss <= entry_price:
            stop_loss = entry_price + max(atr * 1.5, entry_price * 0.01)
        if take_profit >= entry_price:
            take_profit = entry_price - abs(stop_loss - entry_price) * 2.0
        sl_dist = stop_loss - entry_price
        if sl_dist < min_sl_distance:
            stop_loss = entry_price + min_sl_distance
        elif sl_dist > max_sl_distance:
            stop_loss = entry_price + max_sl_distance
    return round(stop_loss, 5), round(take_profit, 5)


def detect_patterns(df: pd.DataFrame) -> list[str]:
    patterns = []
    if len(df) < 3:
        return patterns
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    body        = abs(c[-1] - o[-1])
    total_range = h[-1] - l[-1]
    if total_range > 0 and body / total_range < 0.1:
        patterns.append("doji")
    if (c[-2] < o[-2] and c[-1] > o[-1] and c[-1] > o[-2] and o[-1] < c[-2]):
        patterns.append("bullish_engulfing")
    if (c[-2] > o[-2] and c[-1] < o[-1] and c[-1] < o[-2] and o[-1] > c[-2]):
        patterns.append("bearish_engulfing")
    lower_wick = min(o[-1], c[-1]) - l[-1]
    upper_wick = h[-1] - max(o[-1], c[-1])
    if body > 0 and lower_wick > 2 * body and upper_wick < body:
        patterns.append("hammer")
    if body > 0 and upper_wick > 2 * body and lower_wick < body:
        patterns.append("shooting_star")
    return patterns


def apply_strategy_filters(indicators: dict, strategy_type: str, params: dict) -> dict:
    filtered = dict(indicators)
    if strategy_type == "trend_following":
        filtered["_filter_adx_threshold"]  = params.get("adx_threshold", 25)
        filtered["_filter_rsi_min"]         = params.get("rsi_min", 40)
        filtered["_filter_rsi_max"]         = params.get("rsi_max", 70)
        filtered["_filter_min_confidence"]  = params.get("min_confidence", 0.70)
        filtered["_strategy_adx_ok"]        = indicators.get("adx", 0) > params.get("adx_threshold", 25)
        filtered["_strategy_rsi_ok"]        = params.get("rsi_min", 40) < indicators.get("rsi_14", 50) < params.get("rsi_max", 70)
    elif strategy_type == "momentum":
        filtered["_filter_rsi_buy"]         = params.get("rsi_buy_threshold", 55)
        filtered["_filter_rsi_sell"]        = params.get("rsi_sell_threshold", 45)
        filtered["_filter_volume_ratio_min"] = params.get("volume_ratio_min", 1.5)
        filtered["_filter_min_confidence"]  = params.get("min_confidence", 0.75)
        filtered["_strategy_volume_ok"]     = indicators.get("volume_ratio", 0) >= params.get("volume_ratio_min", 1.5)
    elif strategy_type == "mean_reversion":
        filtered["_filter_bb_buy"]          = params.get("bb_position_buy", 0.15)
        filtered["_filter_bb_sell"]         = params.get("bb_position_sell", 0.85)
        filtered["_filter_rsi_oversold"]    = params.get("rsi_oversold", 30)
        filtered["_filter_rsi_overbought"]  = params.get("rsi_overbought", 70)
        filtered["_filter_min_confidence"]  = params.get("min_confidence", 0.68)
        filtered["_strategy_bb_oversold"]   = indicators.get("bb_position", 0.5) < params.get("bb_position_buy", 0.15)
        filtered["_strategy_bb_overbought"] = indicators.get("bb_position", 0.5) > params.get("bb_position_sell", 0.85)
    return filtered


def get_mtf_trend(htf_indicators: dict) -> str:
    ema_50  = htf_indicators.get("ema_50")
    ema_200 = htf_indicators.get("ema_200")
    adx     = htf_indicators.get("adx", 0)
    price   = htf_indicators.get("current_price", 0)
    if not ema_50 or not price:
        return "neutral"
    above_ema50  = price > ema_50
    above_ema200 = (price > ema_200) if ema_200 else None
    strong_trend = adx > 20
    if above_ema50 and (above_ema200 is None or above_ema200) and strong_trend:
        return "bullish"
    if not above_ema50 and (above_ema200 is None or not above_ema200) and strong_trend:
        return "bearish"
    return "neutral"