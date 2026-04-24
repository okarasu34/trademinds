import asyncio
from datetime import datetime
from loguru import logger
from sqlalchemy import update

# Varsayılan kütüphanelerini ve modellerini kullandığını varsayıyorum
from db.database import AsyncSessionLocal
from db.models import Trade, OrderStatus, OrderSide, TradeMode

class TradingBot:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.is_running = False
        self.adapters = {}
        # KRİTİK: Aynı anda aynı sembolün işlenmesini engelleyen kilit seti
        self.processing_symbols = set() 
        self.active_trades_cache = {} # Sembol bazlı hızlı kontrol için

    async def _scan_and_execute(self):
        async with AsyncSessionLocal() as db:
            strategies = await self._get_active_strategies(db)
            if not strategies: return

            # Mevcut açık pozisyonları veritabanından çek ve cache'e at
            open_trades = await self._get_open_positions(db)
            self.active_trades_cache = {t.symbol: t for t in open_trades}
            
            account_info = await self._get_consolidated_account()
            risk_manager = RiskManager(self.config)

            # Acil durum stop kontrolü...
            
            for strategy in strategies:
                for market_type in (strategy.markets or []):
                    symbols = self._get_symbols(strategy, market_type)
                    for symbol in symbols:
                        # 1. KONTROL: Sembol şu an analiz ediliyor mu veya zaten açık pozisyon var mı?
                        if symbol in self.processing_symbols or symbol in self.active_trades_cache:
                            continue
                        
                        try:
                            # 2. KONTROL: Broker tarafında (DB'ye yansımamış) pozisyon var mı? (Duplicate Önleyici)
                            adapter = self.adapters.get(market_type)
                            if adapter:
                                broker_orders = await adapter.get_open_orders()
                                if any(o.symbol == symbol for o in broker_orders):
                                    logger.warning(f"Duplicate detected on broker for {symbol}, skipping.")
                                    continue

                            # Analiz sürecini başlat ve sembolü kilitle
                            asyncio.create_task(self._safe_analyze_and_trade(
                                db, symbol, market_type, strategy, account_info, risk_manager
                            ))
                        except Exception as e:
                            logger.error(f"Error initiating analysis for {symbol}: {e}")

    async def _safe_analyze_and_trade(self, db, symbol, market_type, strategy, account_info, risk_manager):
        """Kilitleme mekanizmalı analiz fonksiyonu"""
        if symbol in self.processing_symbols: return
        
        self.processing_symbols.add(symbol)
        try:
            # Strateji Analizi (MTF + EMA + RSI)
            signal = await self._enhanced_strategy_logic(symbol, market_type, strategy)
            
            if signal and signal["signal"] != "hold" and signal.get("confidence", 0) >= 0.70:
                # İşlem Açma Mantığı (Burada mevcut place_order ve _save_trade kodların çalışacak)
                await self._execute_trade(db, symbol, market_type, strategy, signal, account_info, risk_manager)
                
        finally:
            self.processing_symbols.remove(symbol)

    async def _enhanced_strategy_logic(self, symbol, market_type, strategy):
        """Geliştirilmiş Strateji: 4H Trend + 1H Momentum (RSI/ATR)"""
        adapter = self.adapters.get(market_type)
        
        # Veri çekme
        df_1h = await adapter.get_candles(symbol, "1h", 100)
        df_4h = await adapter.get_candles(symbol, "4h", 50)
        
        if df_1h is None or df_4h is None: return None

        # İndikatör hesaplamaları (pandas_ta vb. kullanarak)
        # 4H EMA 200 (Ana Trend)
        ema_200_4h = ta.ema(df_4h['close'], length=50).iloc[-1] # 4H'de 50 periyot ~200H'ye yakındır
        current_price = df_1h['close'].iloc[-1]
        
        # 1H RSI ve ATR
        rsi_1h = ta.rsi(df_1h['close'], length=14).iloc[-1]
        atr_1h = ta.atr(df_1h['high'], df_1h['low'], df_1h['close'], length=14).iloc[-1]

        signal = {"signal": "hold", "confidence": 0, "reasoning": ""}

        # BOĞA STRATEJİSİ (Long)
        # 1. Ana trend yukarı (Fiyat 4H EMA üzerinde)
        # 2. RSI aşırı satımdan dönüyor (Örn: 35)
        if current_price > ema_200_4h and 30 < rsi_1h < 45:
            signal.update({
                "signal": "buy",
                "confidence": 0.85,
                "stop_loss": current_price - (atr_1h * 2),
                "take_profit": current_price + (atr_1h * 3),
                "reasoning": "4H Trend Bullish + 1H RSI Rebound"
            })

        # AYI STRATEJİSİ (Short)
        elif current_price < ema_200_4h and 55 < rsi_1h < 70:
             signal.update({
                "signal": "sell",
                "confidence": 0.85,
                "stop_loss": current_price + (atr_1h * 2),
                "take_profit": current_price - (atr_1h * 3),
                "reasoning": "4H Trend Bearish + 1H RSI Pullback"
            })

        return signal

    async def _execute_trade(self, db, symbol, market_type, strategy, signal, account_info, risk_manager):
        # İşlem açma kodun buraya gelir. 
        # ÖNEMLİ: İşlem açıldıktan sonra self.active_trades_cache[symbol] = trade 
        # diyerek cache'i manuel güncelle ki loop henüz bitmeden tekrar açmasın.
        pass