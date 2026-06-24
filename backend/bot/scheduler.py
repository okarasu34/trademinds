"""
Scheduler — bot job'larını zamanlar.

Job'lar:
  1. bot_scan: Her 5 dakikada bir sinyal + emir pipeline'ı
  2. balance_sync: Her 5 dakikada bir balance DB'ye yaz
  3. trade_sync: Her 5 dakikada bir Capital.com ↔ DB tam senkron
     - Capital.com'da açık ama DB'de olmayan → DB'ye ekle
     - DB'de OPEN ama Capital.com'da olmayan → CLOSED yap + PnL yaz
     - DB'deki açık pozisyonların PnL'ini güncelle
  4. profit_guard: Her 1 dakikada bir kâr koruma
     - Kâra geçen pozisyonların zirvesini takip et
     - Zirveden belirli oranda düşerse pozisyonu kapat
"""
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from sqlalchemy import select

from db.database import AsyncSessionLocal
from db.models import BrokerAccount, Trade, OrderStatus, MarketType, OrderSide, TradeMode
from db.redis_client import cache_set, cache_get
from brokers.capital_adapter import CapitalAdapter
from bot.trading_bot import bot_instance
from core.config import settings


scheduler = AsyncIOScheduler(timezone="UTC")


def _infer_market(symbol: str) -> MarketType:
    s = symbol.upper()
    if any(c in s for c in ("BTC","ETH","XRP","DOGE","SOL","ADA","AAVE","AVAX","LTC","SHIB","PEPE","TRX","HBAR","XLM","ALPHA","USDT")):
        return MarketType.CRYPTO
    if any(c in s for c in ("XAU","XAG","OIL","GAS","GOLD","SILVER","PLATINUM","PALLADIUM","COPPER","CORN","WHEAT","NATURALGAS","BRENT")):
        return MarketType.COMMODITY
    if any(c in s for c in ("SPX","NDX","DJI","DAX","FTSE","NKY","US100","US500","US30","DE40")):
        return MarketType.INDEX
    if len(s) == 6 and s.isalpha():
        return MarketType.FOREX
    return MarketType.STOCK


async def bot_scan_job():
    try:
        await bot_instance.scan()
    except Exception as e:
        logger.exception(f"bot_scan_job failed: {e}")


async def balance_sync_job():
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(
                select(BrokerAccount).where(BrokerAccount.is_active == True)
            )
            brokers = result.scalars().all()

            for broker in brokers:
                if broker.broker_type != "capitalcom":
                    continue
                try:
                    adapter = CapitalAdapter(broker)
                    if not await adapter.connect():
                        logger.warning(f"[balance_sync] {broker.name}: connect failed")
                        continue

                    info = await adapter.get_account_info()
                    broker.balance     = info.balance
                    broker.equity      = info.equity
                    broker.margin_used = info.margin_used
                    broker.currency    = info.currency
                    broker.last_sync   = datetime.utcnow()
                    broker.is_connected = True

                    await db.commit()
                    logger.info(f"[balance_sync] {broker.name}: {info.balance:.2f} {info.currency}")
                    await adapter.disconnect()

                except Exception as e:
                    logger.exception(f"[balance_sync] {broker.name} error: {e}")
                    await db.rollback()

        except Exception as e:
            logger.exception(f"balance_sync_job failed: {e}")


async def trade_sync_job():
    """Capital.com ↔ DB tam senkron:
    1. Capital.com'da açık ama DB'de olmayan → ekle
    2. DB'de OPEN ama Capital.com'da olmayan → CLOSED yap
    3. DB'deki OPEN trade'lerin PnL'ini güncelle
    """
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(
                select(BrokerAccount).where(BrokerAccount.is_active == True)
            )
            brokers = result.scalars().all()

            for broker in brokers:
                if broker.broker_type != "capitalcom":
                    continue

                try:
                    adapter = CapitalAdapter(broker)
                    if not await adapter.connect():
                        logger.warning(f"[trade_sync] {broker.name}: connect failed")
                        continue

                    # Capital.com'dan canlı pozisyonları al
                    live_positions = await adapter.get_open_orders()
                    live_map = {p.order_id: p for p in live_positions}

                    # Broker'ın user_id'sini al
                    user_result = await db.execute(
                        select(BrokerAccount.user_id).where(BrokerAccount.id == broker.id)
                    )
                    user_id = user_result.scalar()

                    # DB'deki OPEN trade'leri al
                    open_trades_result = await db.execute(
                        select(Trade).where(
                            Trade.broker_id == broker.id,
                            Trade.status == OrderStatus.OPEN,
                        )
                    )
                    open_trades = open_trades_result.scalars().all()
                    db_order_ids = {t.broker_order_id for t in open_trades if t.broker_order_id}

                    # dealReference → dealId mapping (adapter'dan)
                    ref_map = getattr(adapter, '_deal_ref_map', {})
                    
                    # DB'deki dealReference'ları dealId'ye çevir
                    for trade in open_trades:
                        if trade.broker_order_id in ref_map:
                            mapped_id = ref_map[trade.broker_order_id]
                            if mapped_id in live_map and trade.broker_order_id not in live_map:
                                # dealReference → dealId güncelle
                                old_ref = trade.broker_order_id
                                trade.broker_order_id = mapped_id
                                db_order_ids.discard(old_ref)
                                db_order_ids.add(mapped_id)
                                logger.info(f"[trade_sync] ID fix: {trade.symbol} {old_ref[:20]}→{mapped_id[:20]}")

                    closed_count = 0
                    updated_count = 0

                    # Kapanan trade'ler varsa PnL'lerini transactions'dan çek
                    # Sembol + ID bazlı eşleşme (dealRef/dealId farkını yönetir)
                    live_symbols = {p.symbol: p for p in live_positions}
                    live_ids = set(live_map.keys())
                    
                    closing_trades = []
                    for t in open_trades:
                        # Önce ID ile eşleştir
                        if t.broker_order_id in live_ids:
                            continue  # eşleşti, açık
                        # ID eşleşmediyse sembol ile kontrol et
                        if t.symbol in live_symbols:
                            # Aynı sembol açık — ID'yi güncelle
                            live_pos = live_symbols[t.symbol]
                            old_id = t.broker_order_id
                            t.broker_order_id = live_pos.order_id
                            logger.info(f"[trade_sync] ID fix: {t.symbol} {old_id[:16]}→{live_pos.order_id[:16]}")
                            continue  # eşleşti, açık
                        # Ne ID ne sembol eşleşti — kapanmış
                        closing_trades.append(t)
                    
                    closed_txns = {}
                    if closing_trades:
                        closed_txns = await adapter.get_closed_transactions(hours_back=6)

                    # 1. DB'de OPEN ama Capital.com'da olmayan → CLOSED + PnL
                    for trade in closing_trades:
                        trade.status    = OrderStatus.CLOSED
                        trade.closed_at = datetime.utcnow()
                        trade.closed_by = "bot"
                        
                        # Transactions'dan PnL al
                        # Önce dealId ile dene, bulamazsa sembol ile eşleştir
                        txn = closed_txns.get(trade.broker_order_id)
                        if not txn:
                            # dealId eşleşmedi — sembol ile ara
                            for txn_id, txn_data in closed_txns.items():
                                if txn_data["symbol"] == trade.symbol:
                                    txn = txn_data
                                    break
                        
                        if txn:
                            trade.pnl = txn["pnl"]
                            logger.info(
                                f"[trade_sync] Closed: {trade.symbol} "
                                f"pnl={txn['pnl']}"
                            )
                        elif trade.pnl is not None:
                            # Son bilinen PnL'i koru (önceki sync'ten)
                            logger.info(
                                f"[trade_sync] Closed: {trade.symbol} "
                                f"pnl={trade.pnl} (last known)"
                            )
                        else:
                            logger.warning(
                                f"[trade_sync] Closed: {trade.symbol} "
                                f"pnl=UNKNOWN (no transaction found)"
                            )
                        closed_count += 1

                    # 2. DB'deki OPEN trade'lerin PnL'ini güncelle
                    for trade in open_trades:
                        matched = live_map.get(trade.broker_order_id)
                        if not matched and trade.symbol in live_symbols:
                            matched = live_symbols[trade.symbol]
                        if matched:
                            trade.pnl        = matched.pnl
                            trade.exit_price = matched.current_price
                            updated_count += 1

                    # 3. Capital.com'da açık ama DB'de olmayan → ekle
                    added_count = 0
                    db_symbols = {t.symbol for t in open_trades if t.status.value == "OPEN"}
                    for order_id, pos in live_map.items():
                        if order_id not in db_order_ids and pos.symbol not in db_symbols:
                            new_trade = Trade(
                                user_id         = user_id,
                                broker_id       = broker.id,
                                symbol          = pos.symbol,
                                market_type     = _infer_market(pos.symbol),
                                side            = OrderSide.BUY if pos.side == "buy" else OrderSide.SELL,
                                status          = OrderStatus.OPEN,
                                trade_mode      = TradeMode.LIVE,
                                entry_price     = pos.entry_price,
                                lot_size        = pos.lot_size,
                                pnl             = pos.pnl,
                                exit_price      = pos.current_price,
                                broker_order_id = order_id,
                                opened_at       = datetime.utcnow(),
                            )
                            db.add(new_trade)
                            added_count += 1
                            logger.info(f"[trade_sync] Added: {pos.symbol} {pos.side} pnl={pos.pnl}")

                    await db.commit()

                    if closed_count or added_count or updated_count:
                        logger.info(
                            f"[trade_sync] closed={closed_count} "
                            f"added={added_count} "
                            f"pnl_updated={updated_count}"
                        )

                    await adapter.disconnect()

                except Exception as e:
                    logger.exception(f"[trade_sync] {broker.name} error: {e}")
                    await db.rollback()

        except Exception as e:
            logger.exception(f"trade_sync_job failed: {e}")


async def profit_guard_job():
    """Kâr koruma — açık pozisyonları izle, kârı koru.
    
    1. Her açık pozisyonun anlık PnL'ini çek
    2. PnL >= kâr kilidi eşiği → koruma aktif
    3. Zirve PnL'i Redis'te tut
    4. PnL zirveden belirli oranda düştüyse → pozisyonu kapat
    """
    if not settings.PROFIT_GUARD_ENABLED:
        return

    lock_eur  = settings.PROFIT_GUARD_LOCK_EUR   # default: 10 EUR
    drop_pct  = settings.PROFIT_GUARD_DROP_PCT    # default: 40%

    try:
        async with AsyncSessionLocal() as db:
            # Aktif broker'ları al
            brokers_result = await db.execute(
                select(BrokerAccount).where(BrokerAccount.is_active == True)
            )
            brokers = brokers_result.scalars().all()

            for broker in brokers:
                try:
                    adapter = CapitalAdapter(broker)
                    if not await adapter.connect():
                        continue

                    # Capital.com'dan açık pozisyonları çek
                    live_positions = await adapter.get_open_orders()
                    if not live_positions:
                        await adapter.disconnect()
                        continue

                    closed_count = 0

                    for pos in live_positions:
                        pnl = pos.pnl
                        if pnl is None:
                            continue

                        redis_key = f"profit_guard:{broker.id}:{pos.order_id}"

                        # ── PnL kâr eşiğinin altında → henüz koruma yok ──
                        if pnl < lock_eur:
                            # Daha önce koruma aktifti ama PnL düştüyse
                            # (bu SL'e yaklaşıyor demek, müdahale etme)
                            continue

                        # ── Kâr kilidi aktif — zirve takibi ──
                        cached = await cache_get(redis_key)

                        if cached:
                            peak_pnl = cached.get("peak_pnl", pnl)
                        else:
                            peak_pnl = pnl
                            logger.info(
                                f"[profit_guard] {pos.symbol} kâr kilidi aktif! "
                                f"PnL={pnl:.2f} EUR"
                            )

                        # Yeni zirve mi?
                        if pnl > peak_pnl:
                            peak_pnl = pnl
                            logger.info(
                                f"[profit_guard] {pos.symbol} yeni zirve: {peak_pnl:.2f} EUR"
                            )

                        # Zirveyi Redis'e kaydet (24 saat TTL)
                        await cache_set(redis_key, {
                            "peak_pnl": peak_pnl,
                            "lock_activated": True,
                            "symbol": pos.symbol,
                        }, ttl=86400)

                        # ── Düşüş kontrolü ──
                        if peak_pnl > 0:
                            drop_from_peak = ((peak_pnl - pnl) / peak_pnl) * 100

                            if drop_from_peak >= drop_pct:
                                # KAPAT!
                                logger.warning(
                                    f"[profit_guard] {pos.symbol} KAPATILIYOR! "
                                    f"Zirve={peak_pnl:.2f} → Şimdi={pnl:.2f} "
                                    f"(düşüş: %{drop_from_peak:.1f})"
                                )

                                success = await adapter.close_order(pos.order_id, pos.symbol)

                                if success:
                                    closed_count += 1
                                    logger.success(
                                        f"[profit_guard] {pos.symbol} KAPATILDI! "
                                        f"Kâr korundu: {pnl:.2f} EUR "
                                        f"(zirve: {peak_pnl:.2f})"
                                    )

                                    # DB'de trade'i güncelle
                                    trade_result = await db.execute(
                                        select(Trade).where(
                                            Trade.broker_order_id == pos.order_id,
                                            Trade.status == OrderStatus.OPEN,
                                        )
                                    )
                                    trade = trade_result.scalars().first()
                                    if trade:
                                        trade.status    = OrderStatus.CLOSED
                                        trade.pnl       = pnl
                                        trade.closed_at  = datetime.utcnow()
                                        trade.closed_by  = "profit_guard"
                                        trade.ai_reasoning = (trade.ai_reasoning or "") + f" | PG: peak={peak_pnl:.2f} close={pnl:.2f}"

                                    # Redis'ten sil
                                    await cache_set(redis_key, None, ttl=1)
                                else:
                                    logger.error(
                                        f"[profit_guard] {pos.symbol} kapatma BAŞARISIZ!"
                                    )

                    if closed_count:
                        await db.commit()
                        logger.info(f"[profit_guard] {closed_count} pozisyon kapatıldı")

                    await adapter.disconnect()

                except Exception as e:
                    logger.exception(f"[profit_guard] {broker.name} error: {e}")

    except Exception as e:
        logger.exception(f"profit_guard_job failed: {e}")


def setup_scheduler():
    scheduler.add_job(
        bot_scan_job,
        IntervalTrigger(minutes=5),
        id="bot_scan",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        balance_sync_job,
        IntervalTrigger(minutes=5, start_date="2000-01-01 00:01:00"),
        id="balance_sync",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        trade_sync_job,
        IntervalTrigger(minutes=5, start_date="2000-01-01 00:02:00"),
        id="trade_sync",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    # Profit Guard — her 1 dakikada bir kâr koruma
    if settings.PROFIT_GUARD_ENABLED:
        scheduler.add_job(
            profit_guard_job,
            IntervalTrigger(seconds=settings.PROFIT_GUARD_INTERVAL, start_date="2000-01-01 00:00:30"),
            id="profit_guard",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

    scheduler.start()
    pg_status = "ON" if settings.PROFIT_GUARD_ENABLED else "OFF"
    logger.info(
        f">>> Scheduler started: bot_scan + balance_sync + trade_sync (every 5 min) "
        f"+ profit_guard ({pg_status}, every {settings.PROFIT_GUARD_INTERVAL}s)"
    )