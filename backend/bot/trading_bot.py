"""
TradeMinds bot core — pipeline architecture + 3 strateji.

Pipeline (her strateji için):
  1. Sinyal üret (AlphaTrend / RSIDivergence / SmartMoney)
  2. EMA200 + EMA50 trend filtresi
  3. RSI aralık filtresi (35-65)
  4. MACD teyidi
  5. ★ News Guard (high-impact haber varsa sembolü atla)
  6. ★ News Sentiment Boost (actual vs forecast → confidence ayarla)
  7. Signal Validator (idempotency)
  8. Position Guard
  9. ★ Margin Guard (toplam margin %10'u geçerse dur)
  10. Risk Manager (ATR bazlı SL/TP)
  11. Order Executor
"""
import asyncio
import hashlib
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import AsyncSessionLocal
from db.models import (
    BotConfig, BotStatus, BrokerAccount, Trade, Strategy,
    OrderSide, OrderStatus, TradeMode, MarketType, AISignalLog
)
from db.redis_client import cache_set, cache_get
from brokers.base_adapter import BrokerAdapter, get_broker_adapter
from bot.indicators import calculate_indicators
from data.calendar import calendar_client
from core.config import settings


# ─────────────────────── MARGIN LIMIT ───────────────────────

MAX_MARGIN_PCT = settings.BOT_MAX_MARGIN_PCT  # Default %10, .env ile değiştirilebilir

# ─────────────────────── SL/TP — ATR BAZLI ───────────────────────

ATR_STOP_MULTIPLIER = 2.0   # SL = ATR × 2
ATR_TP_MULTIPLIER   = 3.0   # TP = ATR × 3  → R/R = 1:1.5

# Minimum SL mesafesi (piyasa tipine göre fallback)
MARKET_SL_PCT = {
    "forex":     0.20,
    "crypto":    1.00,
    "index":     0.30,
    "commodity": 0.50,
    "stock":     0.70,
}


# ─────────────────────── DATA CLASSES ───────────────────────

@dataclass
class Signal:
    user_id: str
    broker_id: str
    symbol: str
    market_type: MarketType
    side: OrderSide
    confidence: float
    reasoning: str
    indicators: dict
    timestamp: datetime
    strategy_id: Optional[str] = None

    def idempotency_key(self) -> str:
        bucket = int(self.timestamp.timestamp() / 300)
        raw = f"{self.user_id}:{self.symbol}:{self.side.value}:{bucket}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class ScanContext:
    open_symbols: set = field(default_factory=set)
    open_count: int = 0
    market_counts: dict = field(default_factory=dict)


# ─────────────────────── STRATEJI SINIFLAR ───────────────────────

class AlphaTrendStrategy:
    """Alpha Trend crossover + EMA hizalaması."""

    @staticmethod
    def generate(ind: dict, params: dict) -> Optional[tuple[OrderSide, float, str]]:
        at_cross_up   = ind.get("alpha_trend_cross_up", False)
        at_cross_down = ind.get("alpha_trend_cross_down", False)
        ema13_gt_21   = ind.get("ema13_above_ema21", False)
        ema13_gt_89   = ind.get("ema13_above_ema89", False)
        rsi           = ind.get("rsi_13", 50)
        macd_cross    = ind.get("macd13_crossover", "bearish")

        if at_cross_up and ema13_gt_21 and ema13_gt_89:
            confidence = 0.65
            if macd_cross == "bullish":
                confidence += 0.10
            if rsi > 50:
                confidence += 0.05
            return (
                OrderSide.BUY,
                min(confidence, 0.95),
                f"AlphaTrend cross UP | EMA13>{ind.get('ema_13','?')} > EMA21 | RSI={rsi:.1f}"
            )

        if at_cross_down and not ema13_gt_21 and not ema13_gt_89:
            confidence = 0.65
            if macd_cross == "bearish":
                confidence += 0.10
            if rsi < 50:
                confidence += 0.05
            return (
                OrderSide.SELL,
                min(confidence, 0.95),
                f"AlphaTrend cross DOWN | EMA13 < EMA21 | RSI={rsi:.1f}"
            )

        return None


class MultiDivergenceStrategy:
    """10 İndikatör Multi-Divergence + Tillson T3 + Sharpe Ratio + LinReg + RSI Midline.
    
    Pine Script v10 stratejisinin tam Python karşılığı.
    
    Sinyal: 10 indikatörde diverjans taraması
    Filtreler: Tillson T3 trend, Sharpe Ratio, Linear Regression, RSI Midline
    """

    @staticmethod
    def generate(ind: dict, params: dict) -> Optional[tuple[OrderSide, float, str]]:
        bull_count = ind.get("multi_div_bull_count", 0)
        bear_count = ind.get("multi_div_bear_count", 0)
        bull_names = ind.get("multi_div_bull_names", "")
        bear_names = ind.get("multi_div_bear_names", "")
        near_fibo  = ind.get("near_fibo_level")
        rsi        = ind.get("rsi_13", 50)
        macd_cross = ind.get("macd13_crossover", "bearish")

        # Yeni indikatörler
        t3_rising      = ind.get("t3_rising")
        above_t3       = ind.get("above_t3")
        sharpe_status  = ind.get("sharpe_status", "neutral")
        sharpe_ratio   = ind.get("sharpe_ratio", 0)
        linreg_trend   = ind.get("linreg_trend", "neutral")
        linreg_strong  = ind.get("linreg_strong", False)
        linreg_pos     = ind.get("linreg_position", "inside")
        above_midline  = ind.get("above_rsi_midline")

        min_div = params.get("min_divergence_count", 2)

        # ── Bullish Divergence ──
        if bull_count >= min_div:
            confidence = 0.55 + bull_count * 0.05

            # Fibonacci desteği
            if near_fibo in ("fibo_382", "fibo_500", "fibo_618"):
                confidence += 0.08

            # RSI oversold bölgesi
            if rsi < 40:
                confidence += 0.05
            
            # MACD crossover teyidi
            if macd_cross == "bullish":
                confidence += 0.05

            # Tillson T3 yükseliyor
            if t3_rising is True:
                confidence += 0.03

            # RSI Midline altında (dip bölgesi)
            if above_midline is False:
                confidence += 0.03

            # Linear Regression trend teyidi
            if linreg_trend == "bullish" and linreg_strong:
                confidence += 0.05
            
            # Linear Regression alt bandında
            if linreg_pos == "below_lower":
                confidence += 0.03

            # Sharpe Ratio filtresi
            if sharpe_status == "overvalued":
                confidence -= 0.10  # aşırı değerli, BUY riskli
            elif sharpe_status == "critical_undervalued":
                confidence += 0.05  # çok düşük değerli, BUY fırsatı

            return (
                OrderSide.BUY,
                min(max(confidence, 0.40), 0.95),
                f"MultiDiv BULL ({bull_count}): {bull_names} | Fibo={near_fibo} | "
                f"RSI={rsi:.1f} | T3={'↑' if t3_rising else '↓'} | "
                f"Sharpe={sharpe_ratio:.1f} | LinReg={linreg_trend}"
            )

        # ── Bearish Divergence ──
        if bear_count >= min_div:
            confidence = 0.55 + bear_count * 0.05

            if near_fibo in ("fibo_618", "fibo_786", "fibo_1000"):
                confidence += 0.08

            if rsi > 60:
                confidence += 0.05
            
            if macd_cross == "bearish":
                confidence += 0.05

            # Tillson T3 düşüyor
            if t3_rising is False:
                confidence += 0.03

            # RSI Midline üstünde (tepe bölgesi)
            if above_midline is True:
                confidence += 0.03

            # Linear Regression trend teyidi
            if linreg_trend == "bearish" and linreg_strong:
                confidence += 0.05

            # Linear Regression üst bandında
            if linreg_pos == "above_upper":
                confidence += 0.03

            # Sharpe Ratio filtresi
            if sharpe_status == "critical_undervalued":
                confidence -= 0.10  # çok düşük değerli, SELL riskli
            elif sharpe_status == "overvalued":
                confidence += 0.05  # aşırı değerli, SELL fırsatı

            return (
                OrderSide.SELL,
                min(max(confidence, 0.40), 0.95),
                f"MultiDiv BEAR ({bear_count}): {bear_names} | Fibo={near_fibo} | "
                f"RSI={rsi:.1f} | T3={'↑' if t3_rising else '↓'} | "
                f"Sharpe={sharpe_ratio:.1f} | LinReg={linreg_trend}"
            )

        return None


class SmartMoneyStrategy:
    """Order Block + FVG + POC."""

    @staticmethod
    def generate(ind: dict, params: dict) -> Optional[tuple[OrderSide, float, str]]:
        in_bull_ob    = ind.get("in_bullish_ob", False)
        in_bear_ob    = ind.get("in_bearish_ob", False)
        near_poc      = ind.get("near_poc", False)
        bull_fvg      = ind.get("bull_fvg", False)
        bear_fvg      = ind.get("bear_fvg", False)
        macd_cross    = ind.get("macd13_crossover", "bearish")
        poc_prox      = ind.get("poc_proximity_pct", 999)
        rsi           = ind.get("rsi_13", 50)
        poc_threshold = params.get("poc_proximity_pct", 0.5)

        if in_bull_ob and poc_prox <= poc_threshold:
            confidence = 0.65
            if bull_fvg:
                confidence += 0.10
            if macd_cross == "bullish":
                confidence += 0.10
            if rsi < 50:
                confidence += 0.05
            return (
                OrderSide.BUY,
                min(confidence, 0.95),
                f"Bullish OB + POC ({poc_prox:.2f}%) | FVG={bull_fvg} | MACD={macd_cross} | RSI={rsi:.1f}"
            )

        if in_bear_ob and poc_prox <= poc_threshold:
            confidence = 0.65
            if bear_fvg:
                confidence += 0.10
            if macd_cross == "bearish":
                confidence += 0.10
            if rsi > 50:
                confidence += 0.05
            return (
                OrderSide.SELL,
                min(confidence, 0.95),
                f"Bearish OB + POC ({poc_prox:.2f}%) | FVG={bear_fvg} | MACD={macd_cross} | RSI={rsi:.1f}"
            )

        return None


class HybridStrategy:
    """Üç stratejinin güçlü yanlarını birleştiren hibrit strateji.
    
    Market tipine göre sinyal üretici seçer, Smart Money ile teyit eder.
    
    Yönlendirme:
      Stock  → Alpha Trend (trend takibi güçlü)
      Crypto → Multi Divergence (dönüş tespiti güçlü)
      Index  → Multi Divergence
      Commodity → Multi Divergence
      Forex  → Multi Divergence
    
    Smart Money teyidi (tüm marketler):
      - Order Block uyumu → confidence +0.08
      - FVG uyumu → confidence +0.05
      - POC yakınlığı → confidence +0.05
      - Smart Money ters yönde → confidence -0.08
    """

    # Market tipi → strateji eşleşmesi
    MARKET_STRATEGY = {
        "stock": "alpha_trend",
        "crypto": "multi_divergence",
        "index": "multi_divergence",
        "commodity": "multi_divergence",
        "forex": "multi_divergence",
    }

    @staticmethod
    def _infer_market(symbol: str) -> str:
        s = symbol.upper()
        if any(c in s for c in ("BTC","ETH","XRP","DOGE","SOL","ADA","AAVE","AVAX","LTC","SHIB","PEPE","TRX","HBAR","XLM","ALPHA","USDT")):
            return "crypto"
        if any(c in s for c in ("XAU","XAG","OIL","GAS","GOLD","SILVER","PLATINUM","PALLADIUM","COPPER","CORN","WHEAT","NATURALGAS","BRENT")):
            return "commodity"
        if any(c in s for c in ("SPX","NDX","DJI","DAX","FTSE","NKY","US100","US500","US30","DE40")):
            return "index"
        if len(s) == 6 and s.isalpha():
            return "forex"
        return "stock"

    @staticmethod
    def _smart_money_confirm(ind: dict, side: OrderSide, confidence: float) -> tuple[float, str]:
        """Smart Money teyidi — OB + FVG + POC ile confidence ayarla."""
        in_bull_ob = ind.get("in_bullish_ob", False)
        in_bear_ob = ind.get("in_bearish_ob", False)
        bull_fvg   = ind.get("bull_fvg", False)
        bear_fvg   = ind.get("bear_fvg", False)
        poc_prox   = ind.get("poc_proximity_pct", 999)
        sm_notes   = []

        if side == OrderSide.BUY:
            # Bullish OB'de → güçlü teyit
            if in_bull_ob:
                confidence += 0.08
                sm_notes.append("BullOB")
            # Bearish OB'de → ters sinyal, ceza
            elif in_bear_ob:
                confidence -= 0.08
                sm_notes.append("BearOB↓")
            # Bullish FVG
            if bull_fvg:
                confidence += 0.05
                sm_notes.append("BullFVG")
            # POC yakını
            if poc_prox <= 0.5:
                confidence += 0.05
                sm_notes.append(f"POC({poc_prox:.1f}%)")

        elif side == OrderSide.SELL:
            if in_bear_ob:
                confidence += 0.08
                sm_notes.append("BearOB")
            elif in_bull_ob:
                confidence -= 0.08
                sm_notes.append("BullOB↓")
            if bear_fvg:
                confidence += 0.05
                sm_notes.append("BearFVG")
            if poc_prox <= 0.5:
                confidence += 0.05
                sm_notes.append(f"POC({poc_prox:.1f}%)")

        return confidence, " ".join(sm_notes) if sm_notes else ""

    @staticmethod
    def generate(ind: dict, params: dict) -> Optional[tuple[OrderSide, float, str]]:
        # Sembol bilgisi indicators'dan gelmiyor, params'dan alıyoruz
        symbol = params.get("_symbol", "")
        market = HybridStrategy._infer_market(symbol)
        strategy_type = HybridStrategy.MARKET_STRATEGY.get(market, "multi_divergence")

        result = None
        source_name = ""

        # ── Market tipine göre sinyal üret ──
        if strategy_type == "alpha_trend":
            result = AlphaTrendStrategy.generate(ind, params)
            source_name = "AT"
        else:
            result = MultiDivergenceStrategy.generate(ind, params)
            source_name = "MD"

        if not result:
            return None

        side, confidence, reasoning = result

        # ── Smart Money teyidi ──
        sm_confidence, sm_notes = HybridStrategy._smart_money_confirm(ind, side, confidence)

        # Final confidence
        final_conf = min(max(sm_confidence, 0.40), 0.95)

        # Reasoning birleştir
        sm_text = f" | SM:[{sm_notes}]" if sm_notes else ""
        final_reasoning = f"[Hybrid:{source_name}/{market}] {reasoning}{sm_text}"

        return (side, final_conf, final_reasoning)


# ─────────────────────── STRATEJİ FACTORY ───────────────────────

STRATEGY_MAP = {
    "Alpha Trend":      AlphaTrendStrategy,
    "Multi Divergence": MultiDivergenceStrategy,
    "Smart Money":      SmartMoneyStrategy,
    "Hybrid":           HybridStrategy,
}


# ─────────────────────── PIPELINE GATES ───────────────────────

class SignalValidator:
    @staticmethod
    async def validate(signal: Signal) -> tuple[bool, str]:
        key = f"signal:processed:{signal.idempotency_key()}"
        existing = await cache_get(key)
        if existing:
            return False, f"Duplicate signal (key={signal.idempotency_key()})"
        await cache_set(key, {"processed_at": datetime.utcnow().isoformat()}, ttl=3600)
        return True, "OK"


class OrderedSymbolCache:
    """Başarılı emir açılan sembolleri Redis'te saklar."""

    @staticmethod
    async def add(user_id: str, symbol: str, ttl: int = 3600):
        key = f"ordered:{user_id}:{symbol}"
        await cache_set(key, {"ordered_at": datetime.utcnow().isoformat()}, ttl=ttl)

    @staticmethod
    async def exists(user_id: str, symbol: str) -> bool:
        key = f"ordered:{user_id}:{symbol}"
        return await cache_get(key) is not None

    @staticmethod
    async def get_all_ordered(user_id: str, symbols: list) -> set:
        ordered = set()
        for symbol in symbols:
            if await OrderedSymbolCache.exists(user_id, symbol):
                ordered.add(symbol)
        return ordered


class TrendFilter:
    """EMA200 + EMA50 trend filtresi (YUMUŞAK).
    
    Sert red yerine confidence düşürür.
    BUY + trend aşağı → confidence -0.12
    EMA50 teyidi yok → confidence -0.08
    """

    @staticmethod
    def check(signal: Signal, ind: dict) -> tuple[float, str]:
        """Returns (confidence_adjustment, message)"""
        ema_long  = ind.get("ema_89") or ind.get("ema_200")
        ema_mid   = ind.get("ema_50")
        price     = ind.get("current_price")

        if not ema_long or not price:
            return 0.0, ""

        trend_up = price > ema_long
        adjustment = 0.0
        notes = []

        # EMA50 teyidi
        if ema_mid:
            ema_confirms = (ema_mid > ema_long) if trend_up else (ema_mid < ema_long)
            if not ema_confirms:
                adjustment -= 0.08
                notes.append(f"EMA50 teyidi yok({adjustment:.2f})")

        # Trend yönü kontrolü
        if signal.side == OrderSide.BUY and not trend_up:
            adjustment -= 0.12
            notes.append(f"Trend ASAGI({adjustment:.2f})")
        elif signal.side == OrderSide.SELL and trend_up:
            adjustment -= 0.12
            notes.append(f"Trend YUKARI({adjustment:.2f})")

        return adjustment, " | ".join(notes)


class RSIFilter:
    """RSI aralık filtresi (YUMUŞAK).
    
    Sert red yerine confidence düşürür.
    RSI 65-75 → confidence -0.08
    RSI > 75 → confidence -0.15
    RSI 25-35 → confidence -0.08
    RSI < 25 → confidence -0.15
    """

    @staticmethod
    def check(signal: Signal, ind: dict) -> tuple[float, str]:
        """Returns (confidence_adjustment, message)"""
        rsi = ind.get("rsi_14") or ind.get("rsi_13")
        if not rsi:
            return 0.0, ""

        adjustment = 0.0
        note = ""

        if signal.side == OrderSide.BUY:
            if rsi > 75:
                adjustment = -0.15
                note = f"RSI çok yüksek({rsi:.1f})"
            elif rsi > 65:
                adjustment = -0.08
                note = f"RSI yüksek({rsi:.1f})"
        elif signal.side == OrderSide.SELL:
            if rsi < 25:
                adjustment = -0.15
                note = f"RSI çok düşük({rsi:.1f})"
            elif rsi < 35:
                adjustment = -0.08
                note = f"RSI düşük({rsi:.1f})"

        return adjustment, note


class MACDFilter:
    """MACD teyidi filtresi (YUMUŞAK).
    
    Sert red yerine confidence düşürür.
    Histogram ters yönde → confidence -0.08
    """

    @staticmethod
    def check(signal: Signal, ind: dict) -> tuple[float, str]:
        """Returns (confidence_adjustment, message)"""
        hist = ind.get("macd13_histogram") or ind.get("macd_histogram")
        if hist is None:
            return 0.0, ""

        adjustment = 0.0
        note = ""

        if signal.side == OrderSide.BUY and hist < 0:
            adjustment = -0.08
            note = f"MACD negatif({hist:.5f})"
        elif signal.side == OrderSide.SELL and hist > 0:
            adjustment = -0.08
            note = f"MACD pozitif({hist:.5f})"

        return adjustment, note


# ─────────────────────── ★ NEWS GUARD ───────────────────────

class NewsGuard:
    """High-impact haber filtresi.
    
    MyFXBook takviminden sembolün para birimini etkileyen
    yaklaşan high-impact eventleri kontrol eder.
    BOT_NEWS_PAUSE_MINUTES içinde high-impact event varsa
    o sembolü atlar.
    
    Örnek: EURUSD taranırken 15dk sonra ECB faiz kararı varsa → SKIP
    """

    # Sembolden para birimlerini çıkar
    SYMBOL_CURRENCIES = {
        "EURUSD": ["EUR", "USD"], "GBPUSD": ["GBP", "USD"], "USDJPY": ["USD", "JPY"],
        "USDCHF": ["USD", "CHF"], "USDCAD": ["USD", "CAD"], "AUDUSD": ["AUD", "USD"],
        "NZDUSD": ["NZD", "USD"], "EURGBP": ["EUR", "GBP"], "EURJPY": ["EUR", "JPY"],
        "GBPJPY": ["GBP", "JPY"], "EURCHF": ["EUR", "CHF"], "EURAUD": ["EUR", "AUD"],
        "EURCAD": ["EUR", "CAD"], "GBPCHF": ["GBP", "CHF"], "GBPAUD": ["GBP", "AUD"],
        "GBPCAD": ["GBP", "CAD"], "AUDJPY": ["AUD", "JPY"], "AUDCAD": ["AUD", "CAD"],
        "AUDNZD": ["AUD", "NZD"], "CADJPY": ["CAD", "JPY"], "NZDJPY": ["NZD", "JPY"],
        "CHFJPY": ["CHF", "JPY"], "USDCNH": ["USD", "CNY"],
    }

    # Emtia/endeks para birimi eşleşmesi
    COMMODITY_CURRENCIES = {
        "GOLD": ["USD"], "XAUUSD": ["USD"], "SILVER": ["USD"], "XAGUSD": ["USD"],
        "OIL": ["USD"], "BRENT": ["USD"], "NATURALGAS": ["USD"],
        "US500": ["USD"], "US30": ["USD"], "NAS100": ["USD"], "US100": ["USD"],
        "DE40": ["EUR"], "UK100": ["GBP"],
    }

    @staticmethod
    def _get_currencies(symbol: str) -> list[str]:
        """Sembolden etkilenen para birimlerini döndür."""
        s = symbol.upper()
        
        # Direkt eşleşme
        if s in NewsGuard.SYMBOL_CURRENCIES:
            return NewsGuard.SYMBOL_CURRENCIES[s]
        if s in NewsGuard.COMMODITY_CURRENCIES:
            return NewsGuard.COMMODITY_CURRENCIES[s]
        
        # 6 harfli forex çifti tahmini (XXXYYY)
        if len(s) == 6 and s.isalpha():
            return [s[:3], s[3:]]
        
        # Kripto → USD bağımlı
        if any(c in s for c in ("BTC", "ETH", "XRP", "SOL", "DOGE", "ADA")):
            return ["USD"]
        
        return []

    @staticmethod
    async def check(symbol: str, pause_minutes: int = None) -> tuple[bool, str, list]:
        """
        High-impact haber kontrolü.
        
        Returns:
            (ok, message, events)
            ok=False → sembol atlanmalı
            events → bulunan eventler (boost için de kullanılabilir)
        """
        if pause_minutes is None:
            pause_minutes = settings.BOT_NEWS_PAUSE_MINUTES or 30

        currencies = NewsGuard._get_currencies(symbol)
        if not currencies:
            return True, "OK", []

        try:
            # Takvimden yaklaşan eventleri çek
            events = await calendar_client.get_calendar(
                hours_ahead=2,
                impact_filter=["high"],
            )
        except Exception as e:
            logger.warning(f"[{symbol}] NewsGuard: Calendar fetch failed: {e}")
            return True, "OK (calendar unavailable)", []  # hata varsa geç, bloklamak istemiyoruz

        # Bu sembolün para birimlerini etkileyen eventleri filtrele
        relevant = []
        for event in events:
            if event.get("currency") in currencies:
                mins = event.get("minutes_until", 999)
                if 0 <= mins <= pause_minutes:
                    relevant.append(event)

        if relevant:
            event_names = ", ".join(
                f"{e['title']} ({e['currency']}, {int(e['minutes_until'])}dk)"
                for e in relevant[:3]
            )
            return False, f"News Guard: {event_names}", relevant

        return True, "OK", events


# ─────────────────────── ★ NEWS SENTIMENT BOOST ───────────────────────

class NewsSentimentBoost:
    """Haber sentiment'ına göre confidence ayarla.
    
    Yaklaşan medium/high impact haberlerin actual vs forecast
    karşılaştırmasını yapar:
    
    - actual > forecast + sinyal aynı yön → confidence +0.05 ~ +0.10
    - actual < forecast + sinyal ters yön → confidence -0.05 ~ -0.10
    - Henüz açıklanmamış ama yaklaşan medium event → confidence -0.03 (belirsizlik)
    
    Minimum confidence eşiği: 0.55 (altına düşerse sinyal iptal)
    """

    MIN_CONFIDENCE = 0.55  # Bu eşiğin altına düşerse sinyal iptal

    # Para birimi → pozitif veri = o para birimini güçlendirir
    # BUY EURUSD → EUR güçlenir → EUR verisi pozitif = iyi
    # SELL EURUSD → USD güçlenir → USD verisi pozitif = iyi

    @staticmethod
    def _parse_number(val: str) -> Optional[float]:
        """Haber verisindeki sayıyı parse et (%, K, M, B destekli)."""
        if not val or val.strip() == "":
            return None
        clean = val.strip().replace(",", "").replace("%", "").replace(" ", "")
        multiplier = 1.0
        if clean.endswith("K"):
            multiplier = 1_000
            clean = clean[:-1]
        elif clean.endswith("M"):
            multiplier = 1_000_000
            clean = clean[:-1]
        elif clean.endswith("B"):
            multiplier = 1_000_000_000
            clean = clean[:-1]
        try:
            return float(clean) * multiplier
        except ValueError:
            return None

    @staticmethod
    def _is_currency_base(symbol: str, currency: str) -> bool:
        """Para birimi bu semboldeki 'base' (ilk) mi yoksa 'quote' (ikinci) mi?
        
        EURUSD → EUR = base, USD = quote
        BUY EURUSD = EUR güçlenir
        Eğer EUR verisi iyi → BUY'ı destekler
        Eğer USD verisi iyi → SELL'i destekler
        """
        s = symbol.upper()
        c = currency.upper()
        
        # Forex çifti
        if len(s) == 6 and s.isalpha():
            return s[:3] == c
        
        # Emtia/endeks → USD genelde quote
        return False

    @staticmethod
    async def adjust(
        symbol: str,
        side: OrderSide,
        confidence: float,
    ) -> tuple[float, str]:
        """
        Confidence'ı haberlere göre ayarla.
        
        Returns:
            (new_confidence, reason)
        """
        currencies = NewsGuard._get_currencies(symbol)
        if not currencies:
            return confidence, ""

        try:
            # Son 4 saatteki medium + high eventleri çek
            events = await calendar_client.get_calendar(
                hours_ahead=4,
                impact_filter=["medium", "high"],
            )
        except Exception as e:
            logger.warning(f"[{symbol}] NewsSentimentBoost: Calendar error: {e}")
            return confidence, ""

        # Bu sembolle ilgili eventleri filtrele
        relevant = [
            e for e in events
            if e.get("currency") in currencies
        ]

        if not relevant:
            return confidence, ""

        adjustment = 0.0
        reasons = []

        for event in relevant:
            actual_str   = event.get("actual", "")
            forecast_str = event.get("forecast", "")
            impact       = event.get("impact", "medium")
            currency     = event.get("currency", "")
            title        = event.get("title", "")
            mins_until   = event.get("minutes_until", 0)

            actual   = NewsSentimentBoost._parse_number(actual_str)
            forecast = NewsSentimentBoost._parse_number(forecast_str)

            # ── DURUM 1: Veri açıklanmış (actual var) ──
            if actual is not None and forecast is not None and forecast != 0:
                surprise_pct = (actual - forecast) / abs(forecast) * 100
                is_base = NewsSentimentBoost._is_currency_base(symbol, currency)

                # Pozitif sürpriz → o para birimi güçlenir
                currency_bullish = surprise_pct > 2  # %2'den fazla sürpriz
                currency_bearish = surprise_pct < -2

                if currency_bullish:
                    if (is_base and side == OrderSide.BUY) or (not is_base and side == OrderSide.SELL):
                        # Sinyal ile uyumlu → boost
                        boost = 0.07 if impact == "high" else 0.04
                        adjustment += boost
                        reasons.append(f"{title}: {currency} beat ({actual_str}>{forecast_str}) → +{boost:.2f}")
                    else:
                        # Sinyal ile ters → penalty
                        penalty = -0.07 if impact == "high" else -0.04
                        adjustment += penalty
                        reasons.append(f"{title}: {currency} beat ({actual_str}>{forecast_str}) vs signal → {penalty:.2f}")

                elif currency_bearish:
                    if (is_base and side == OrderSide.SELL) or (not is_base and side == OrderSide.BUY):
                        boost = 0.07 if impact == "high" else 0.04
                        adjustment += boost
                        reasons.append(f"{title}: {currency} miss ({actual_str}<{forecast_str}) → +{boost:.2f}")
                    else:
                        penalty = -0.07 if impact == "high" else -0.04
                        adjustment += penalty
                        reasons.append(f"{title}: {currency} miss ({actual_str}<{forecast_str}) vs signal → {penalty:.2f}")

            # ── DURUM 2: Henüz açıklanmamış, yaklaşan event (belirsizlik) ──
            elif actual is None and 0 < mins_until <= 120:
                # Yaklaşan event belirsizlik yaratır → küçük penalty
                penalty = -0.03 if impact == "high" else -0.01
                adjustment += penalty
                reasons.append(f"{title} ({currency}, {int(mins_until)}dk) pending → {penalty:.2f}")

        if adjustment == 0:
            return confidence, ""

        # Adjustment'ı sınırla: max ±0.15
        adjustment = max(-0.15, min(0.15, adjustment))
        new_confidence = round(confidence + adjustment, 2)

        reason_str = " | ".join(reasons)

        if new_confidence < NewsSentimentBoost.MIN_CONFIDENCE:
            reason_str = f"BELOW MIN ({new_confidence:.2f} < {NewsSentimentBoost.MIN_CONFIDENCE}) | {reason_str}"

        return new_confidence, reason_str


# ─────────────────────── POSITION / DAILY LOSS GUARDS ───────────────────────

class PositionGuard:
    @staticmethod
    def check(signal: Signal, ctx: ScanContext, config: BotConfig) -> tuple[bool, str]:
        if signal.symbol in ctx.open_symbols:
            return False, f"Position already open: {signal.symbol}"
        if ctx.open_count >= config.max_positions:
            return False, f"Max positions reached ({ctx.open_count}/{config.max_positions})"
        market_limits = config.market_limits or {}
        market_key    = signal.market_type.value
        market_limit  = market_limits.get(market_key)
        if market_limit:
            current = ctx.market_counts.get(market_key, 0)
            if current >= market_limit:
                return False, f"Market limit reached: {market_key} {current}/{market_limit}"
        return True, "OK"

    @staticmethod
    def register_open(signal: Signal, ctx: ScanContext):
        ctx.open_symbols.add(signal.symbol)
        ctx.open_count += 1
        market_key = signal.market_type.value
        ctx.market_counts[market_key] = ctx.market_counts.get(market_key, 0) + 1

    @staticmethod
    def _infer_market(symbol: str) -> str:
        s = symbol.upper()
        if any(c in s for c in ("BTC","ETH","XRP","DOGE","SOL","ADA","AAVE","AVAX","LTC","SHIB","PEPE","TRX","HBAR","XLM","ALPHA","USDT")):
            return "crypto"
        if any(c in s for c in ("XAU","XAG","OIL","GAS","GOLD","SILVER","PLATINUM","PALLADIUM","COPPER","CORN","WHEAT","NATURALGAS","BRENT")):
            return "commodity"
        if any(c in s for c in ("SPX","NDX","DJI","DAX","FTSE","NKY","US100","US500","US30","DE40")):
            return "index"
        if len(s) == 6 and s.isalpha():
            return "forex"
        return "stock"


class DailyLossGuard:
    @staticmethod
    async def check(config: BotConfig, db: AsyncSession, broker: BrokerAccount, adapter: BrokerAdapter) -> tuple[bool, str]:
        if not config.max_daily_loss_pct or config.max_daily_loss_pct <= 0:
            return True, "OK"
        try:
            info    = await adapter.get_account_info()
            balance = info.balance
        except Exception as e:
            return False, f"Balance fetch failed: {e}"
        if balance <= 0:
            return False, "Invalid balance"
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        result = await db.execute(
            select(func.coalesce(func.sum(Trade.pnl), 0)).where(
                Trade.user_id == config.user_id,
                Trade.status == OrderStatus.CLOSED,
                Trade.closed_at >= today_start,
                Trade.pnl.isnot(None),
            )
        )
        daily_pnl = float(result.scalar() or 0)
        max_loss  = balance * (config.max_daily_loss_pct / 100.0)
        if daily_pnl < 0 and abs(daily_pnl) >= max_loss:
            return False, f"Daily loss limit reached: {daily_pnl:.2f} / -{max_loss:.2f}"
        return True, "OK"


class MarginGuard:
    """Toplam margin kullanım limiti.
    
    Mevcut margin / hesap değeri oranı MAX_MARGIN_PCT'yi geçerse
    yeni pozisyon açılmaz.
    
    Örnek: Hesap €39,000, margin €3,500 → %8.97 → OK
           Hesap €39,000, margin €4,100 → %10.5 → BLOCKED
    """

    @staticmethod
    async def check(adapter: BrokerAdapter) -> tuple[bool, str]:
        try:
            info = await adapter.get_account_info()
        except Exception as e:
            return False, f"MarginGuard: Account info failed: {e}"

        equity = info.equity or info.balance
        if equity <= 0:
            return False, "MarginGuard: Invalid equity"

        margin_used = info.margin_used or 0
        margin_pct  = (margin_used / equity) * 100

        if margin_pct >= MAX_MARGIN_PCT:
            return False, (
                f"MarginGuard: Margin limit reached "
                f"({margin_used:.2f}/{equity:.2f} = {margin_pct:.1f}% >= {MAX_MARGIN_PCT}%)"
            )

        logger.info(
            f"MarginGuard: OK — margin {margin_used:.2f}/{equity:.2f} = {margin_pct:.1f}% "
            f"(limit {MAX_MARGIN_PCT}%)"
        )
        return True, "OK"


class RiskManager:
    """ATR bazlı SL/TP hesabı.
    
    SL = ATR × 2.0
    TP = ATR × 3.0  → R/R = 1:1.5
    """

    @staticmethod
    async def calculate(
        signal: Signal,
        adapter: BrokerAdapter,
        config: BotConfig,
        ind: dict,
    ) -> tuple[bool, str, Optional[dict]]:
        try:
            rules = await adapter.get_market_rules(signal.symbol)
        except Exception as e:
            return False, f"Market rules fetch failed: {e}", None

        min_size     = rules.get("min_size", 1.0)
        min_stop_pct = rules.get("min_stop_pct", 0.1)
        bid          = rules.get("bid", 0)
        ask          = rules.get("ask", 0)

        if not bid or not ask:
            return False, f"Invalid price for {signal.symbol}", None

        price = ask if signal.side == OrderSide.BUY else bid

        # ATR bazlı SL/TP
        atr = ind.get("atr_14") or ind.get("atr_13")
        if atr and atr > 0:
            sl_distance = atr * ATR_STOP_MULTIPLIER
            tp_distance = atr * ATR_TP_MULTIPLIER
        else:
            # ATR yoksa sabit % fallback
            market_key  = PositionGuard._infer_market(signal.symbol)
            sl_pct      = MARKET_SL_PCT.get(market_key, 0.30) / 100.0
            sl_distance = price * sl_pct
            tp_distance = sl_distance * 1.5

        # Capital.com minimum stop distance kontrolü
        min_sl_distance = price * (min_stop_pct / 100.0)
        if sl_distance < min_sl_distance:
            sl_distance = min_sl_distance * 1.5
            tp_distance = sl_distance * 1.5

        if signal.side == OrderSide.BUY:
            stop_loss   = round(price - sl_distance, 5)
            take_profit = round(price + tp_distance, 5)
        else:
            stop_loss   = round(price + sl_distance, 5)
            take_profit = round(price - tp_distance, 5)

        # Lot size hesabı
        risk_pct = (config.max_risk_per_trade_pct or 1.0) / 100.0
        try:
            info    = await adapter.get_account_info()
            balance = info.balance if info.balance > 0 else 10000
        except Exception:
            balance = 10000

        risk_amount    = balance * risk_pct
        calculated_lot = risk_amount / (sl_distance * 100) if sl_distance > 0 else float(min_size)
        lot_size = max(float(min_size), min(round(calculated_lot, 2), 100.0))

        logger.info(
            f"[{signal.symbol}] ATR={atr:.5f} | "
            f"sl={stop_loss} tp={take_profit} | "
            f"risk={config.max_risk_per_trade_pct}% | lot={lot_size}"
        )

        return True, "OK", {
            "lot_size":    lot_size,
            "entry_price": price,
            "stop_loss":   stop_loss,
            "take_profit": take_profit,
        }


class OrderExecutor:
    @staticmethod
    async def execute(signal: Signal, adapter: BrokerAdapter, risk_params: dict, config: BotConfig, db: AsyncSession) -> tuple[bool, str, Optional[Trade]]:
        deal_ref = await adapter.place_order(
            symbol      = signal.symbol,
            side        = signal.side.value,
            lot_size    = risk_params["lot_size"],
            stop_loss   = risk_params["stop_loss"],
            take_profit = risk_params["take_profit"],
            comment     = f"TradeMinds:{signal.idempotency_key()}",
        )
        if not deal_ref:
            return False, "Capital.com returned no dealReference", None

        trade = Trade(
            user_id         = signal.user_id,
            broker_id       = signal.broker_id,
            strategy_id     = signal.strategy_id,
            symbol          = signal.symbol,
            market_type     = signal.market_type,
            side            = signal.side,
            status          = OrderStatus.OPEN,
            trade_mode      = TradeMode(config.trade_mode.value),
            entry_price     = risk_params["entry_price"],
            lot_size        = risk_params["lot_size"],
            stop_loss       = risk_params["stop_loss"],
            take_profit     = risk_params["take_profit"],
            ai_reasoning    = signal.reasoning,
            ai_confidence   = signal.confidence,
            signals_used    = signal.indicators,
            broker_order_id = deal_ref,
            opened_at       = datetime.utcnow(),
        )
        db.add(trade)
        await db.commit()
        await db.refresh(trade)
        return True, f"Order placed: {deal_ref}", trade


# ─────────────────────── MAIN BOT ───────────────────────

class TradingBot:
    def __init__(self):
        self._adapter_cache: dict[str, BrokerAdapter] = {}

    async def _get_adapter(self, broker: BrokerAccount) -> Optional[BrokerAdapter]:
        if broker.id in self._adapter_cache:
            adapter = self._adapter_cache[broker.id]
            if await adapter.is_connected():
                return adapter
            await adapter.disconnect()
            del self._adapter_cache[broker.id]
        adapter = get_broker_adapter(broker)
        if not await adapter.connect():
            logger.error(f"Failed to connect to broker {broker.name}")
            return None
        self._adapter_cache[broker.id] = adapter
        return adapter

    async def _get_active_strategy(self, db: AsyncSession, user_id: str) -> Optional[Strategy]:
        result = await db.execute(
            select(Strategy).where(
                Strategy.user_id == user_id,
                Strategy.is_active == True,
            )
        )
        return result.scalars().first()

    async def scan(self):
        logger.info(">>> Bot scan started")
        async with AsyncSessionLocal() as db:
            try:
                cfg_result = await db.execute(
                    select(BotConfig).where(BotConfig.status == BotStatus.RUNNING)
                )
                configs = cfg_result.scalars().all()
                if not configs:
                    logger.info("No running bots")
                    return
                for config in configs:
                    await self._scan_for_user(db, config)
            except Exception as e:
                logger.exception(f"Scan failed: {e}")
            finally:
                logger.info("<<< Bot scan finished")

    async def _scan_for_user(self, db: AsyncSession, config: BotConfig):
        br_result = await db.execute(
            select(BrokerAccount).where(
                BrokerAccount.user_id == config.user_id,
                BrokerAccount.is_active == True,
            )
        )
        broker = br_result.scalars().first()
        if not broker:
            logger.warning(f"User {config.user_id} has no active broker")
            return

        adapter = await self._get_adapter(broker)
        if not adapter:
            return

        strategy = await self._get_active_strategy(db, config.user_id)
        if not strategy:
            logger.warning("No active strategy found")
            return

        strategy_class = STRATEGY_MAP.get(strategy.name)
        if not strategy_class:
            logger.warning(f"Unknown strategy: {strategy.name}")
            return

        logger.info(f"Active strategy: {strategy.name}")

        symbols = adapter.get_cached_watchlist_symbols()
        if not symbols:
            logger.warning("Watchlist empty, nothing to scan")
            return

        ok, msg = await DailyLossGuard.check(config, db, broker, adapter)
        if not ok:
            logger.warning(f"Daily loss limit hit: {msg}")
            return

        try:
            live_positions = await adapter.get_open_orders()
        except Exception as e:
            logger.error(f"Failed to fetch live positions: {e}")
            return

        ctx = ScanContext(
            open_symbols  = {p.symbol for p in live_positions},
            open_count    = len(live_positions),
            market_counts = {},
        )
        for p in live_positions:
            mk = PositionGuard._infer_market(p.symbol)
            ctx.market_counts[mk] = ctx.market_counts.get(mk, 0) + 1

        try:
            redis_ordered = await OrderedSymbolCache.get_all_ordered(config.user_id, symbols)
            ctx.open_symbols.update(redis_ordered)
            if redis_ordered:
                logger.info(f"Redis cache: {len(redis_ordered)} symbols blocked")
        except Exception as e:
            logger.warning(f"Redis cache read failed: {e}")

        logger.info(
            f"Scanning {len(symbols)} symbols | open={ctx.open_count} | "
            f"strategy={strategy.name} | risk={config.max_risk_per_trade_pct}%"
        )

        processed = 0
        for symbol in symbols:
            try:
                await self._process_symbol(db, config, broker, adapter, symbol, ctx, strategy, strategy_class)
                processed += 1
            except Exception as e:
                logger.exception(f"[{symbol}] Pipeline error: {e}")

        logger.info(f"Scan complete: {processed}/{len(symbols)} symbols processed")

    async def _process_symbol(
        self,
        db: AsyncSession,
        config: BotConfig,
        broker: BrokerAccount,
        adapter: BrokerAdapter,
        symbol: str,
        ctx: ScanContext,
        strategy: Strategy,
        strategy_class,
    ):
        try:
            df = await adapter.get_candles(symbol, "1h", limit=200)
        except Exception as e:
            logger.warning(f"[{symbol}] Candle fetch failed: {e}")
            return

        if df is None or df.empty or len(df) < 50:
            return

        try:
            ind = calculate_indicators(df)
        except Exception as e:
            logger.warning(f"[{symbol}] Indicator error: {e}")
            return

        if not ind:
            return

        # Strateji sinyali üret
        strat_params = dict(strategy.parameters or {})
        strat_params["_symbol"] = symbol  # HybridStrategy market tipi için
        result = strategy_class.generate(ind, strat_params)
        if not result:
            return

        side, confidence, reasoning = result

        # Market type
        inferred   = PositionGuard._infer_market(symbol)
        market_map = {
            "crypto":    MarketType.CRYPTO,
            "commodity": MarketType.COMMODITY,
            "index":     MarketType.INDEX,
            "stock":     MarketType.STOCK,
            "forex":     MarketType.FOREX,
        }
        market_type = market_map.get(inferred, MarketType.FOREX)

        signal = Signal(
            user_id     = config.user_id,
            broker_id   = broker.id,
            symbol      = symbol,
            market_type = market_type,
            side        = side,
            confidence  = confidence,
            reasoning   = reasoning,
            indicators  = {
                "rsi":         ind.get("rsi_13"),
                "macd":        ind.get("macd13"),
                "alpha_trend": ind.get("alpha_trend"),
                "ema_13":      ind.get("ema_13"),
                "ema_21":      ind.get("ema_21"),
                "atr":         ind.get("atr_14"),
            },
            timestamp   = datetime.utcnow(),
            strategy_id = strategy.id,
        )

        logger.info(f"[{symbol}] [{strategy.name}] Signal: {side.value} conf={confidence:.2f} | {reasoning}")

        # ── YUMUŞAK FILTRELER (confidence ayarlama) ──
        MIN_CONFIDENCE = 0.50  # Bu eşiğin altına düşerse sinyal iptal

        filter_notes = []

        # 1. Trend filtresi (yumuşak)
        trend_adj, trend_note = TrendFilter.check(signal, ind)
        if trend_adj != 0:
            signal.confidence += trend_adj
            filter_notes.append(f"Trend:{trend_adj:+.2f}")
            if trend_note:
                logger.info(f"[{symbol}] Soft Trend: {trend_note} → conf={signal.confidence:.2f}")

        # 2. RSI filtresi (yumuşak)
        rsi_adj, rsi_note = RSIFilter.check(signal, ind)
        if rsi_adj != 0:
            signal.confidence += rsi_adj
            filter_notes.append(f"RSI:{rsi_adj:+.2f}")
            if rsi_note:
                logger.info(f"[{symbol}] Soft RSI: {rsi_note} → conf={signal.confidence:.2f}")

        # 3. MACD filtresi (yumuşak)
        macd_adj, macd_note = MACDFilter.check(signal, ind)
        if macd_adj != 0:
            signal.confidence += macd_adj
            filter_notes.append(f"MACD:{macd_adj:+.2f}")
            if macd_note:
                logger.info(f"[{symbol}] Soft MACD: {macd_note} → conf={signal.confidence:.2f}")

        # Minimum confidence kontrolü
        if signal.confidence < MIN_CONFIDENCE:
            all_notes = " | ".join(filter_notes)
            logger.info(
                f"[{symbol}] REJECTED by Soft Filters: "
                f"conf={signal.confidence:.2f} < {MIN_CONFIDENCE} | {all_notes}"
            )
            return

        # Filtre notlarını reasoning'e ekle
        if filter_notes:
            signal.reasoning += f" | Filters:[{' '.join(filter_notes)}]"

        # ★ 4. NEWS GUARD — High-impact haber filtresi
        ok, msg, news_events = await NewsGuard.check(symbol)
        if not ok:
            logger.warning(f"[{symbol}] SKIPPED by {msg}")
            return

        # ★ 5. NEWS SENTIMENT BOOST — Confidence ayarlama
        new_confidence, news_reason = await NewsSentimentBoost.adjust(
            symbol, signal.side, signal.confidence
        )
        if news_reason:
            old_conf = signal.confidence
            signal.confidence = new_confidence
            signal.reasoning += f" | News: {news_reason}"
            logger.info(
                f"[{symbol}] News Boost: conf {old_conf:.2f} → {new_confidence:.2f} | {news_reason}"
            )
            # Minimum confidence kontrolü
            if new_confidence < NewsSentimentBoost.MIN_CONFIDENCE:
                logger.warning(
                    f"[{symbol}] REJECTED by News Boost: "
                    f"conf {new_confidence:.2f} < min {NewsSentimentBoost.MIN_CONFIDENCE}"
                )
                return

        # 6. Signal Validator (idempotency)
        ok, msg = await SignalValidator.validate(signal)
        if not ok:
            logger.info(f"[{symbol}] REJECTED at Validator: {msg}")
            return

        # 7. Position Guard
        ok, msg = PositionGuard.check(signal, ctx, config)
        if not ok:
            logger.info(f"[{symbol}] REJECTED at PositionGuard: {msg}")
            return

        # 8. ★ Margin Guard — toplam margin %10'u geçerse dur
        ok, msg = await MarginGuard.check(adapter)
        if not ok:
            logger.warning(f"[{symbol}] REJECTED at {msg}")
            return

        # 9. Risk Manager (ATR bazlı)
        ok, msg, risk_params = await RiskManager.calculate(signal, adapter, config, ind)
        if not ok:
            logger.info(f"[{symbol}] REJECTED at RiskManager: {msg}")
            return

        # 10. Order Executor
        if config.trade_mode != TradeMode.LIVE:
            logger.info(f"[{symbol}] PAPER mode — order not sent")
            return

        ok, msg, trade = await OrderExecutor.execute(signal, adapter, risk_params, config, db)
        if ok:
            logger.success(f"[{symbol}] ORDER PLACED: {msg}")
            PositionGuard.register_open(signal, ctx)
            await OrderedSymbolCache.add(signal.user_id, symbol, ttl=3600)
            log = AISignalLog(
                user_id     = signal.user_id,
                symbol      = signal.symbol,
                market_type = signal.market_type,
                signal      = signal.side.value,
                confidence  = signal.confidence,
                reasoning   = signal.reasoning,
                indicators  = signal.indicators,
                acted_on    = True,
                trade_id    = trade.id,
            )
            db.add(log)
            await db.commit()
        else:
            logger.error(f"[{symbol}] ORDER FAILED: {msg}")


# Global bot instance
bot_instance = TradingBot()