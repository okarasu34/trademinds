import pandas as pd
import numpy as np
from typing import Optional


def _find_pivots(series: pd.Series, period: int) -> tuple[list, list]:
    """Pivot high ve pivot low noktalarını bul."""
    pivot_highs = []  # (index, value)
    pivot_lows  = []  # (index, value)
    
    for i in range(period, len(series) - period):
        # Pivot High: ortadaki bar her iki taraftakinden yüksek
        is_ph = True
        for j in range(1, period + 1):
            if series.iloc[i] <= series.iloc[i - j] or series.iloc[i] <= series.iloc[i + j]:
                is_ph = False
                break
        if is_ph:
            pivot_highs.append((i, series.iloc[i]))
        
        # Pivot Low: ortadaki bar her iki taraftakinden düşük
        is_pl = True
        for j in range(1, period + 1):
            if series.iloc[i] >= series.iloc[i - j] or series.iloc[i] >= series.iloc[i + j]:
                is_pl = False
                break
        if is_pl:
            pivot_lows.append((i, series.iloc[i]))
    
    return pivot_highs, pivot_lows


def _check_divergence(
    price: pd.Series,
    indicator: pd.Series,
    pivot_period: int,
    max_bars: int,
    max_pivots: int,
    div_type: str = "regular",
) -> bool:
    """
    Diverjans kontrolü.
    
    regular bullish: fiyat düşük dip, indikatör yüksek dip
    regular bearish: fiyat yüksek zirve, indikatör düşük zirve
    """
    if len(price) < pivot_period * 2 + 5:
        return False
    
    try:
        if div_type == "regular":
            # Bullish: pivot lows'da kontrol
            _, price_lows = _find_pivots(price, pivot_period)
            _, ind_lows   = _find_pivots(indicator, pivot_period)
            
            if len(price_lows) < 2 or len(ind_lows) < 2:
                return False
            
            # Son pivot low
            curr_price_idx, curr_price_val = price_lows[-1]
            curr_ind_val = indicator.iloc[curr_price_idx] if curr_price_idx < len(indicator) else None
            if curr_ind_val is None or np.isnan(curr_ind_val):
                return False
            
            # Önceki pivot lows ile karşılaştır
            for i in range(len(price_lows) - 2, max(len(price_lows) - max_pivots - 1, -1), -1):
                prev_idx, prev_price_val = price_lows[i]
                if curr_price_idx - prev_idx > max_bars:
                    break
                if curr_price_idx - prev_idx < 5:
                    continue
                
                prev_ind_val = indicator.iloc[prev_idx] if prev_idx < len(indicator) else None
                if prev_ind_val is None or np.isnan(prev_ind_val):
                    continue
                
                # Bullish regular: fiyat düşük dip, indikatör yüksek dip
                if curr_price_val < prev_price_val and curr_ind_val > prev_ind_val:
                    return True
            
            return False
        
        elif div_type == "regular_bear":
            # Bearish: pivot highs'da kontrol
            price_highs, _ = _find_pivots(price, pivot_period)
            ind_highs, _   = _find_pivots(indicator, pivot_period)
            
            if len(price_highs) < 2 or len(ind_highs) < 2:
                return False
            
            curr_price_idx, curr_price_val = price_highs[-1]
            curr_ind_val = indicator.iloc[curr_price_idx] if curr_price_idx < len(indicator) else None
            if curr_ind_val is None or np.isnan(curr_ind_val):
                return False
            
            for i in range(len(price_highs) - 2, max(len(price_highs) - max_pivots - 1, -1), -1):
                prev_idx, prev_price_val = price_highs[i]
                if curr_price_idx - prev_idx > max_bars:
                    break
                if curr_price_idx - prev_idx < 5:
                    continue
                
                prev_ind_val = indicator.iloc[prev_idx] if prev_idx < len(indicator) else None
                if prev_ind_val is None or np.isnan(prev_ind_val):
                    continue
                
                # Bearish regular: fiyat yüksek zirve, indikatör düşük zirve
                if curr_price_val > prev_price_val and curr_ind_val < prev_ind_val:
                    return True
            
            return False
    
    except Exception:
        return False
    
    return False


def _detect_multi_divergence(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    rsi: pd.Series,
    macd: pd.Series,
    macd_hist: pd.Series,
    stoch: pd.Series,
    cci: pd.Series,
    momentum: pd.Series,
    obv: pd.Series,
    vwmacd: pd.Series,
    cmf: pd.Series,
    mfi: pd.Series,
    pivot_period: int = 8,
    max_bars: int = 144,
    max_pivots: int = 13,
) -> dict:
    """
    10 indikatörde diverjans tespit et.
    
    TradingView Pine Script'teki mantığı Python'a çevrilmiş hali.
    MACD, Histogram, RSI, Stochastic, CCI, Momentum, OBV, VWmacd, CMF, MFI
    """
    indicators_list = [
        ("MACD",  macd),
        ("Hist",  macd_hist),
        ("RSI",   rsi),
        ("Stoch", stoch),
        ("CCI",   cci),
        ("MOM",   momentum),
        ("OBV",   obv),
        ("VWMACD", vwmacd),
        ("CMF",   cmf),
        ("MFI",   mfi),
    ]
    
    bull_divs = []
    bear_divs = []
    
    for name, ind_series in indicators_list:
        try:
            if ind_series is None or len(ind_series) < pivot_period * 2 + 10:
                continue
            
            # NaN kontrolü
            clean = ind_series.dropna()
            if len(clean) < pivot_period * 2 + 10:
                continue
            
            # Bullish divergence (fiyat düşük dip, indikatör yüksek dip)
            if _check_divergence(close, ind_series, pivot_period, max_bars, max_pivots, "regular"):
                bull_divs.append(name)
            
            # Bearish divergence (fiyat yüksek zirve, indikatör düşük zirve)
            if _check_divergence(close, ind_series, pivot_period, max_bars, max_pivots, "regular_bear"):
                bear_divs.append(name)
        
        except Exception:
            continue
    
    return {
        "bull_count": len(bull_divs),
        "bear_count": len(bear_divs),
        "bull_names": ", ".join(bull_divs) if bull_divs else "",
        "bear_names": ", ".join(bear_divs) if bear_divs else "",
    }


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

    # ─── Ek İndikatörler (Multi-Divergence için) ───

    # Stochastic 13 (diverjans için ayrı — mevcut stoch 14 periyot)
    low13_stoch  = low.rolling(13).min()
    high13_stoch = high.rolling(13).max()
    stoch13_k    = 100 * (close - low13_stoch) / (high13_stoch - low13_stoch).replace(0, np.nan)
    stoch13      = stoch13_k.rolling(3).mean()
    indicators["stoch_13"] = round(stoch13.iloc[-1], 2) if not np.isnan(stoch13.iloc[-1]) else 50

    # CCI (Commodity Channel Index, 13 periyot)
    tp_cci   = (high + low + close) / 3
    sma_tp   = tp_cci.rolling(13).mean()
    mad_tp   = tp_cci.rolling(13).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci      = (tp_cci - sma_tp) / (0.015 * mad_tp.replace(0, np.nan))
    indicators["cci_13"] = round(cci.iloc[-1], 2) if not np.isnan(cci.iloc[-1]) else 0

    # Momentum (13 periyot)
    momentum = close - close.shift(13)
    indicators["momentum_13"] = round(momentum.iloc[-1], 5) if not np.isnan(momentum.iloc[-1]) else 0

    # OBV (On Balance Volume)
    obv = pd.Series(0.0, index=close.index)
    for idx in range(1, len(close)):
        if close.iloc[idx] > close.iloc[idx-1]:
            obv.iloc[idx] = obv.iloc[idx-1] + volume.iloc[idx]
        elif close.iloc[idx] < close.iloc[idx-1]:
            obv.iloc[idx] = obv.iloc[idx-1] - volume.iloc[idx]
        else:
            obv.iloc[idx] = obv.iloc[idx-1]
    indicators["obv"] = round(obv.iloc[-1], 2)

    # VWmacd (Volume Weighted MACD)
    def vwma(src, vol, period):
        return (src * vol).rolling(period).sum() / vol.rolling(period).sum().replace(0, np.nan)
    vwma_fast = vwma(close, volume, 13)
    vwma_slow = vwma(close, volume, 21)
    vwmacd    = vwma_fast - vwma_slow
    indicators["vwmacd"] = round(vwmacd.iloc[-1], 5) if not np.isnan(vwmacd.iloc[-1]) else 0

    # CMF (Chaikin Money Flow, 21 periyot)
    cmf_m = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    cmf_v = cmf_m * volume
    cmf   = cmf_v.rolling(21).sum() / volume.rolling(21).sum().replace(0, np.nan)
    indicators["cmf_21"] = round(cmf.iloc[-1], 4) if not np.isnan(cmf.iloc[-1]) else 0

    # ─── Multi-Indicator Divergence (10 indikatör) ───
    # Pine Script'teki pivot-tabanlı diverjans tespiti
    div_result = _detect_multi_divergence(
        close=close, high=high, low=low,
        rsi=rsi13, macd=macd13, macd_hist=hist13,
        stoch=stoch13, cci=cci, momentum=momentum,
        obv=obv, vwmacd=vwmacd, cmf=cmf, mfi=mfi,
        pivot_period=8, max_bars=144, max_pivots=13,
    )
    indicators["multi_div_bull_count"]   = div_result["bull_count"]
    indicators["multi_div_bear_count"]   = div_result["bear_count"]
    indicators["multi_div_bull_names"]   = div_result["bull_names"]
    indicators["multi_div_bear_names"]   = div_result["bear_names"]
    indicators["multi_div_bull_signal"]  = div_result["bull_count"] >= 2
    indicators["multi_div_bear_signal"]  = div_result["bear_count"] >= 2

    # Eski basit diverjans (geriye uyumluluk)
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

    # ─── RSI Midline (fiyat bazlı destek/direnç) ───
    # Pine Script: RSI overbought/oversold seviyelerini fiyat üzerinde gösterir
    ob_level = 70
    os_level = 30
    rsi_len  = 21
    ep_rsi   = 2 * rsi_len - 1
    rsi_src  = close
    auc = rsi_src.diff().clip(lower=0).ewm(span=ep_rsi, adjust=False).mean()
    adc = (-rsi_src.diff()).clip(lower=0).ewm(span=ep_rsi, adjust=False).mean()

    x1 = (rsi_len - 1) * (adc * ob_level / (100 - ob_level) - auc)
    ub_rsi = pd.Series(np.where(x1 >= 0, rsi_src + x1, rsi_src + x1 * (100 - ob_level) / ob_level), index=close.index)

    x2 = (rsi_len - 1) * (adc * os_level / (100 - os_level) - auc)
    lb_rsi = pd.Series(np.where(x2 >= 0, rsi_src + x2, rsi_src + x2 * (100 - os_level) / os_level), index=close.index)

    rsi_midline = (ub_rsi + lb_rsi) / 2
    indicators["rsi_midline"]    = round(rsi_midline.iloc[-1], 5) if not np.isnan(rsi_midline.iloc[-1]) else None
    indicators["rsi_resistance"] = round(ub_rsi.iloc[-1], 5) if not np.isnan(ub_rsi.iloc[-1]) else None
    indicators["rsi_support"]    = round(lb_rsi.iloc[-1], 5) if not np.isnan(lb_rsi.iloc[-1]) else None
    indicators["above_rsi_midline"] = bool(current_close > rsi_midline.iloc[-1]) if not np.isnan(rsi_midline.iloc[-1]) else None

    # ─── Tillson T3 ───
    # Üçlü üstel yumuşatma — gürültüyü filtreler
    t3_len = 8
    t3_vf  = 0.7
    t3_src = (high + low + open_ + close) / 4

    t3_e1 = t3_src.ewm(span=t3_len, adjust=False).mean()
    t3_e2 = t3_e1.ewm(span=t3_len, adjust=False).mean()
    t3_e3 = t3_e2.ewm(span=t3_len, adjust=False).mean()
    t3_e4 = t3_e3.ewm(span=t3_len, adjust=False).mean()
    t3_e5 = t3_e4.ewm(span=t3_len, adjust=False).mean()
    t3_e6 = t3_e5.ewm(span=t3_len, adjust=False).mean()

    t3_c1 = -t3_vf ** 3
    t3_c2 = 3 * t3_vf ** 2 + 3 * t3_vf ** 3
    t3_c3 = -6 * t3_vf ** 2 - 3 * t3_vf - 3 * t3_vf ** 3
    t3_c4 = 1 + 3 * t3_vf + t3_vf ** 3 + 3 * t3_vf ** 2

    t3 = t3_c1 * t3_e6 + t3_c2 * t3_e5 + t3_c3 * t3_e4 + t3_c4 * t3_e3
    indicators["tillson_t3"]       = round(t3.iloc[-1], 5) if not np.isnan(t3.iloc[-1]) else None
    indicators["t3_rising"]        = bool(t3.iloc[-1] > t3.iloc[-2]) if len(t3) >= 2 and not np.isnan(t3.iloc[-2]) else None
    indicators["above_t3"]         = bool(current_close > t3.iloc[-1]) if not np.isnan(t3.iloc[-1]) else None

    # T3 Fibonacci (length=5, vf=0.618)
    t3f_len = 5
    t3f_vf  = 0.618
    t3f_e1 = t3_src.ewm(span=t3f_len, adjust=False).mean()
    t3f_e2 = t3f_e1.ewm(span=t3f_len, adjust=False).mean()
    t3f_e3 = t3f_e2.ewm(span=t3f_len, adjust=False).mean()
    t3f_e4 = t3f_e3.ewm(span=t3f_len, adjust=False).mean()
    t3f_e5 = t3f_e4.ewm(span=t3f_len, adjust=False).mean()
    t3f_e6 = t3f_e5.ewm(span=t3f_len, adjust=False).mean()

    t3f_c1 = -t3f_vf ** 3
    t3f_c2 = 3 * t3f_vf ** 2 + 3 * t3f_vf ** 3
    t3f_c3 = -6 * t3f_vf ** 2 - 3 * t3f_vf - 3 * t3f_vf ** 3
    t3f_c4 = 1 + 3 * t3f_vf + t3f_vf ** 3 + 3 * t3f_vf ** 2

    t3_fibo = t3f_c1 * t3f_e6 + t3f_c2 * t3f_e5 + t3f_c3 * t3f_e4 + t3f_c4 * t3f_e3
    indicators["tillson_t3_fibo"]  = round(t3_fibo.iloc[-1], 5) if not np.isnan(t3_fibo.iloc[-1]) else None
    indicators["t3_fibo_rising"]   = bool(t3_fibo.iloc[-1] > t3_fibo.iloc[-2]) if len(t3_fibo) >= 2 and not np.isnan(t3_fibo.iloc[-2]) else None

    # ─── Sharpe Ratio (180 periyot) ───
    # Overvalued/Undervalued sınıflandırması
    sharpe_lookback = min(180, len(df) - 1)
    risk_free_rate  = 0.04  # yıllık
    daily_return    = close.pct_change()

    if sharpe_lookback >= 30:
        mean_ret   = daily_return.tail(sharpe_lookback).mean()
        std_ret    = daily_return.tail(sharpe_lookback).std() * np.sqrt(sharpe_lookback)
        if std_ret and std_ret > 0:
            sharpe = (mean_ret * 365 - risk_free_rate) / std_ret
        else:
            sharpe = 0.0

        indicators["sharpe_ratio"]   = round(sharpe, 3)
        indicators["sharpe_status"]  = (
            "overvalued" if sharpe > 5.0
            else "undervalued" if -3.0 < sharpe < -1.0
            else "critical_undervalued" if sharpe <= -3.0
            else "neutral"
        )
    else:
        indicators["sharpe_ratio"]  = 0.0
        indicators["sharpe_status"] = "neutral"

    # ─── Linear Regression Channel (144 periyot) ───
    linreg_len = min(144, len(df))
    if linreg_len >= 30:
        linreg_src = close.tail(linreg_len).values
        x_vals     = np.arange(1, linreg_len + 1, dtype=float)

        # Slope ve intercept hesapla
        sum_x    = x_vals.sum()
        sum_y    = linreg_src.sum()
        sum_xy   = (x_vals * linreg_src).sum()
        sum_x2   = (x_vals ** 2).sum()
        n        = float(linreg_len)

        slope     = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        avg_y     = sum_y / n
        intercept = avg_y - slope * sum_x / n + slope

        start_price = intercept + slope * (linreg_len - 1)
        end_price   = intercept

        # Standard deviation ve Pearson R
        predicted = intercept + slope * np.arange(0, linreg_len, dtype=float)
        residuals = linreg_src - predicted
        std_dev   = np.std(residuals)

        # Pearson R korelasyonu
        mean_x = x_vals.mean()
        mean_y = linreg_src.mean()
        dsxx   = ((x_vals - mean_x) ** 2).sum()
        dsyy   = ((predicted - predicted.mean()) ** 2).sum()
        dsxy   = ((x_vals - mean_x) * (predicted - predicted.mean())).sum()
        pearson_r = dsxy / np.sqrt(dsxx * dsyy) if dsxx > 0 and dsyy > 0 else 0.0

        upper_band = end_price + 2 * std_dev
        lower_band = end_price - 2 * std_dev

        # Trend yönü: start > end ise yükseliş (slope negatif = fiyat yükseliyor)
        linreg_trend = "bullish" if slope < 0 else "bearish"

        indicators["linreg_slope"]      = round(slope, 8)
        indicators["linreg_upper"]      = round(upper_band, 5)
        indicators["linreg_lower"]      = round(lower_band, 5)
        indicators["linreg_middle"]     = round(end_price, 5)
        indicators["linreg_pearson_r"]  = round(abs(pearson_r), 4)
        indicators["linreg_trend"]      = linreg_trend
        indicators["linreg_strong"]     = bool(abs(pearson_r) > 0.8)
        indicators["above_linreg"]      = bool(current_close > end_price)
        indicators["linreg_position"]   = (
            "above_upper" if current_close > upper_band
            else "below_lower" if current_close < lower_band
            else "inside"
        )
    else:
        indicators["linreg_slope"]     = 0.0
        indicators["linreg_pearson_r"] = 0.0
        indicators["linreg_trend"]     = "neutral"
        indicators["linreg_strong"]    = False
        indicators["linreg_position"]  = "inside"

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