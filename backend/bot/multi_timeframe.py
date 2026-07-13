"""
TradeMinds - Multi-Timeframe Analiz Modülü
============================================
Mantık:
  1H'de sinyal geldiğinde 4H ve 1D trend yönüne bakar.
  - 3 timeframe aynı yönde  → confidence YÜKSELT (güçlü sinyal)
  - 2 timeframe aynı yönde  → confidence'ı olduğu gibi bırak (nötr)
  - Ters yöndeyse           → confidence DÜŞÜR veya trade'i ATLA (mod'a göre)

Trend tespiti EMA20 / EMA50 ilişkisiyle yapılır (bağımsız, indikatör
kütüphanenize ihtiyaç duymaz). Kendi trend fonksiyonunuz varsa
`trend_detector` parametresiyle değiştirebilirsiniz.

Kurulum:
  pip install pandas numpy   (muhtemelen zaten kurulu)

Kullanım (Hybrid stratejinizin sinyal ürettiği yerde):

  from multi_timeframe import MultiTimeframeAnalyzer, MTFMode

  mtf = MultiTimeframeAnalyzer(mode=MTFMode.SOFT)  # veya MTFMode.HARD

  result = await mtf.confirm(
      symbol="EUR/USD",
      signal_side="BUY",              # 1H'den gelen ham sinyal
      base_confidence=0.62,           # stratejinizin ürettiği confidence
      fetch_candles=broker.get_candles  # sizin OHLCV fetch fonksiyonunuz
  )

  if result.action == MTFAction.SKIP:
      continue  # trade açma
  else:
      final_confidence = result.adjusted_confidence
      # ... trade'i final_confidence ile aç ...
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, List, Optional

logger = logging.getLogger("multi_timeframe")

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    raise ImportError(
        "multi_timeframe.py için pandas ve numpy gerekli. "
        "Kurulum: pip install pandas numpy"
    ) from e


# ------------------------------------------------------------------
# Ayarlanabilir sabitler
# ------------------------------------------------------------------

EMA_FAST = 20
EMA_SLOW = 50

# 3/3 uyum → bu kadar confidence ekle (üst sınır 1.0'da kırpılır)
BOOST_FULL_ALIGN = 0.15
# 2/3 uyum (sadece bir timeframe destekliyor) → küçük ekleme
BOOST_PARTIAL_ALIGN = 0.05
# Ters yön (4H ve 1D ikisi de karşı yönde) → SOFT modda bu kadar düş
PENALTY_FULL_OPPOSITE = 0.25
# Karışık (biri destekliyor biri karşı) → hafif düşür
PENALTY_MIXED = 0.08


class Trend(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"


class MTFMode(str, Enum):
    SOFT = "soft"   # Ters yönde confidence düşür ama trade'e izin ver
    HARD = "hard"   # Ters yönde trade'i tamamen atla


class MTFAction(str, Enum):
    PROCEED = "PROCEED"   # trade'e devam et (confidence ayarlanmış olabilir)
    SKIP = "SKIP"         # trade'i açma


@dataclass
class MTFResult:
    action: MTFAction
    adjusted_confidence: float
    trend_1h: Trend
    trend_4h: Trend
    trend_1d: Trend
    reason: str
    alignment_score: int = 0  # -2..+2 arası, kaç timeframe destekliyor/karşı


# Broker'ınızdan mum verisi çeken fonksiyonun beklenen imzası:
#   async def get_candles(symbol: str, timeframe: str, count: int) -> pd.DataFrame
# DataFrame en az "close" kolonunu içermeli (open/high/low/volume opsiyonel).
CandleFetcher = Callable[[str, str, int], Awaitable["pd.DataFrame"]]


class MultiTimeframeAnalyzer:
    def __init__(
        self,
        mode: MTFMode = MTFMode.SOFT,
        ema_fast: int = EMA_FAST,
        ema_slow: int = EMA_SLOW,
        trend_detector: Optional[Callable[["pd.DataFrame"], Trend]] = None,
    ):
        self.mode = mode
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        # Kendi trend fonksiyonunuzu enjekte edebilirsiniz (opsiyonel)
        self._trend_detector = trend_detector or self._default_trend

    # ---------- Trend tespiti ----------

    def _default_trend(self, df: "pd.DataFrame") -> Trend:
        """
        EMA20 / EMA50 ilişkisi + fiyatın EMA20'ye göre konumu ile basit,
        bağımsız trend tespiti. Kendi indikatör setiniz varsa
        trend_detector parametresiyle bunu değiştirin.
        """
        if df is None or len(df) < self.ema_slow + 5:
            return Trend.FLAT

        closes = df["close"].astype(float)
        ema_fast = closes.ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = closes.ewm(span=self.ema_slow, adjust=False).mean()

        last_fast = ema_fast.iloc[-1]
        last_slow = ema_slow.iloc[-1]
        last_close = closes.iloc[-1]

        # EMA'lar arası fark, slow EMA'ya oranla (yüzde) — "flat" eşiği için
        spread_pct = abs(last_fast - last_slow) / last_slow * 100 if last_slow else 0

        if spread_pct < 0.02:
            return Trend.FLAT

        if last_fast > last_slow and last_close > last_fast:
            return Trend.UP
        if last_fast < last_slow and last_close < last_fast:
            return Trend.DOWN
        return Trend.FLAT

    # ---------- Ana giriş noktası ----------

    async def confirm(
        self,
        symbol: str,
        signal_side: str,
        base_confidence: float,
        fetch_candles: CandleFetcher,
        candle_count: int = 100,
    ) -> MTFResult:
        """
        1H sinyalini 4H ve 1D trend ile teyit eder, confidence'ı ayarlar
        veya trade'i atlamayı önerir.
        """
        signal_side = signal_side.upper()
        signal_trend = Trend.UP if signal_side == "BUY" else Trend.DOWN

        try:
            df_1h = await fetch_candles(symbol, "1H", candle_count)
            df_4h = await fetch_candles(symbol, "4H", candle_count)
            df_1d = await fetch_candles(symbol, "1D", candle_count)
        except Exception as e:
            logger.error(f"MTF candle fetch hatası ({symbol}): {e}")
            # Veri alınamazsa güvenli taraf: confidence'a dokunma, uyar
            return MTFResult(
                action=MTFAction.PROCEED,
                adjusted_confidence=base_confidence,
                trend_1h=Trend.FLAT,
                trend_4h=Trend.FLAT,
                trend_1d=Trend.FLAT,
                reason=f"MTF verisi alınamadı, filtre atlandı: {e}",
            )

        trend_1h = self._trend_detector(df_1h)
        trend_4h = self._trend_detector(df_4h)
        trend_1d = self._trend_detector(df_1d)

        alignment = self._score_alignment(signal_trend, trend_4h, trend_1d)
        adjusted, action, reason = self._apply_adjustment(
            base_confidence, alignment, trend_4h, trend_1d, signal_side
        )

        result = MTFResult(
            action=action,
            adjusted_confidence=round(min(max(adjusted, 0.0), 1.0), 4),
            trend_1h=trend_1h,
            trend_4h=trend_4h,
            trend_1d=trend_1d,
            reason=reason,
            alignment_score=alignment,
        )

        logger.info(
            f"[MTF] {symbol} {signal_side} | 1H={trend_1h} 4H={trend_4h} 1D={trend_1d} "
            f"| base={base_confidence:.2f} -> adj={result.adjusted_confidence:.2f} "
            f"| action={action.value} | {reason}"
        )
        return result

    # ---------- Yardımcı ----------

    def _score_alignment(self, signal_trend: Trend, trend_4h: Trend, trend_1d: Trend) -> int:
        """
        +1: 4H sinyal yönüyle aynı, -1: ters, 0: flat/belirsiz.
        Aynısı 1D için. Toplam -2..+2 arası döner.
        """
        score = 0
        for tf_trend in (trend_4h, trend_1d):
            if tf_trend == signal_trend:
                score += 1
            elif tf_trend == Trend.FLAT:
                score += 0
            else:
                score -= 1
        return score

    def _apply_adjustment(
        self,
        base_confidence: float,
        alignment: int,
        trend_4h: Trend,
        trend_1d: Trend,
        signal_side: str,
    ):
        if alignment == 2:
            return (
                base_confidence + BOOST_FULL_ALIGN,
                MTFAction.PROCEED,
                "4H ve 1D aynı yönde — güçlü sinyal, confidence artırıldı",
            )
        if alignment == 1:
            return (
                base_confidence + BOOST_PARTIAL_ALIGN,
                MTFAction.PROCEED,
                "Bir üst timeframe destekliyor — confidence hafif artırıldı",
            )
        if alignment == 0:
            return (
                base_confidence,
                MTFAction.PROCEED,
                "Üst timeframe'ler nötr/flat — confidence değiştirilmedi",
            )
        if alignment == -1:
            new_conf = base_confidence - PENALTY_MIXED
            return (
                new_conf,
                MTFAction.PROCEED,
                "Üst timeframe'ler karışık — confidence hafif düşürüldü",
            )
        # alignment == -2: 4H ve 1D ikisi de ters yönde
        if self.mode == MTFMode.HARD:
            return (
                base_confidence,
                MTFAction.SKIP,
                "4H ve 1D ikisi de ters yönde (HARD mod) — trade atlandı",
            )
        return (
            base_confidence - PENALTY_FULL_OPPOSITE,
            MTFAction.PROCEED,
            "4H ve 1D ikisi de ters yönde (SOFT mod) — confidence düşürüldü",
        )
