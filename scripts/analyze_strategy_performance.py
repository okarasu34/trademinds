"""
Strateji performans analizi
Sunucuda çalıştır: cd /root/trademinds && venv/bin/python3 scripts/analyze_strategy_performance.py
"""
import asyncio
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from sqlalchemy import select, func
from db.database import AsyncSessionLocal
from db.models import Trade, Strategy, BotConfig, AISignalLog, OrderStatus
from loguru import logger


async def analyze():
    print("\n" + "="*80)
    print("STRATEJI PERFORMANS ANALİZİ")
    print("="*80 + "\n")
    
    async with AsyncSessionLocal() as db:
        # 1. Bot Config
        print("1. BOT CONFIGURATION")
        print("-" * 80)
        result = await db.execute(select(BotConfig).limit(1))
        config = result.scalar_one_or_none()
        
        if config:
            print(f"Trade Mode: {config.trade_mode}")
            print(f"Max Positions: {config.max_positions}")
            print(f"Max Daily Loss: {config.max_daily_loss_pct}%")
            print(f"Max Risk Per Trade: {config.max_risk_per_trade_pct}%")
            print(f"Market Limits: {config.market_limits}")
            print(f"Daily Loss: {config.daily_loss:.2f}")
            print(f"Daily Trades: {config.daily_trades}")
        print()
        
        # 2. Active Strategies
        print("2. ACTIVE STRATEGIES")
        print("-" * 80)
        result = await db.execute(
            select(Strategy).where(Strategy.is_active == True)
        )
        strategies = result.scalars().all()
        
        for s in strategies:
            print(f"\nStrategy: {s.name}")
            print(f"  Type: {s.strategy_type}")
            print(f"  Markets: {s.markets}")
            print(f"  Symbols: {len(s.symbols) if s.symbols else 'Default'}")
            print(f"  Parameters: {s.parameters}")
            print(f"  Total Trades: {s.total_trades}")
            print(f"  Win Rate: {s.win_rate:.1f}%" if s.win_rate else "  Win Rate: N/A")
            print(f"  Total PnL: {s.total_pnl:.2f}" if s.total_pnl else "  Total PnL: 0.00")
        print()
        
        # 3. Last 7 Days Performance
        print("3. LAST 7 DAYS PERFORMANCE")
        print("-" * 80)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        
        result = await db.execute(
            select(Trade).where(
                Trade.opened_at >= seven_days_ago
            ).order_by(Trade.opened_at.desc())
        )
        trades = result.scalars().all()
        
        total_trades = len(trades)
        closed_trades = [t for t in trades if t.status == OrderStatus.CLOSED]
        open_trades = [t for t in trades if t.status == OrderStatus.OPEN]
        
        wins = [t for t in closed_trades if t.pnl and t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl and t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in closed_trades if t.pnl)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0
        
        print(f"Total Trades: {total_trades}")
        print(f"  Closed: {len(closed_trades)}")
        print(f"  Open: {len(open_trades)}")
        print(f"\nWins: {len(wins)}")
        print(f"Losses: {len(losses)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"\nTotal PnL: {total_pnl:.2f} EUR")
        print(f"Avg Win: {avg_win:.2f} EUR")
        print(f"Avg Loss: {avg_loss:.2f} EUR")
        print(f"Risk/Reward: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "Risk/Reward: N/A")
        print()
        
        # 4. By Market Type
        print("4. PERFORMANCE BY MARKET TYPE")
        print("-" * 80)
        for market in ["forex", "crypto", "commodity", "index", "stock"]:
            market_trades = [t for t in closed_trades if t.market_type.value == market]
            if market_trades:
                market_pnl = sum(t.pnl for t in market_trades if t.pnl)
                market_wins = len([t for t in market_trades if t.pnl and t.pnl > 0])
                market_wr = (market_wins / len(market_trades) * 100) if market_trades else 0
                print(f"{market.upper():12s}: {len(market_trades):3d} trades | PnL: {market_pnl:8.2f} EUR | WR: {market_wr:5.1f}%")
        print()
        
        # 5. Top Symbols
        print("5. TOP 10 SYMBOLS (by trade count)")
        print("-" * 80)
        symbol_stats = {}
        for t in closed_trades:
            if t.symbol not in symbol_stats:
                symbol_stats[t.symbol] = {"count": 0, "pnl": 0, "wins": 0}
            symbol_stats[t.symbol]["count"] += 1
            if t.pnl:
                symbol_stats[t.symbol]["pnl"] += t.pnl
                if t.pnl > 0:
                    symbol_stats[t.symbol]["wins"] += 1
        
        sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
        for symbol, stats in sorted_symbols:
            wr = (stats["wins"] / stats["count"] * 100) if stats["count"] > 0 else 0
            print(f"{symbol:12s}: {stats['count']:3d} trades | PnL: {stats['pnl']:8.2f} EUR | WR: {wr:5.1f}%")
        print()
        
        # 6. Signal Analysis
        print("6. SIGNAL ANALYSIS (Last 24 hours)")
        print("-" * 80)
        yesterday = datetime.utcnow() - timedelta(hours=24)
        
        result = await db.execute(
            select(AISignalLog).where(
                AISignalLog.created_at >= yesterday
            )
        )
        signals = result.scalars().all()
        
        buy_signals = [s for s in signals if s.signal == "buy"]
        sell_signals = [s for s in signals if s.signal == "sell"]
        hold_signals = [s for s in signals if s.signal == "hold"]
        acted_signals = [s for s in signals if s.acted_on]
        
        print(f"Total Signals: {len(signals)}")
        print(f"  Buy: {len(buy_signals)}")
        print(f"  Sell: {len(sell_signals)}")
        print(f"  Hold: {len(hold_signals)}")
        print(f"  Acted On: {len(acted_signals)}")
        print(f"\nSignal to Action Rate: {(len(acted_signals)/len(signals)*100):.1f}%" if signals else "N/A")
        
        if acted_signals:
            avg_confidence = sum(s.confidence for s in acted_signals) / len(acted_signals)
            print(f"Avg Confidence (acted): {avg_confidence:.2f}")
        print()
        
        # 7. Recent Trades Detail
        print("7. LAST 10 CLOSED TRADES")
        print("-" * 80)
        recent_closed = [t for t in trades if t.status == OrderStatus.CLOSED][:10]
        
        for t in recent_closed:
            duration = (t.closed_at - t.opened_at).total_seconds() / 3600 if t.closed_at else 0
            result_emoji = "✅" if t.pnl and t.pnl > 0 else "❌"
            print(f"{result_emoji} {t.symbol:12s} | {t.side.value:4s} | "
                  f"PnL: {t.pnl:7.2f} EUR | "
                  f"Duration: {duration:5.1f}h | "
                  f"Confidence: {t.ai_confidence:.2f}")
        print()
        
        # 8. Recommendations
        print("8. RECOMMENDATIONS")
        print("-" * 80)
        
        if win_rate < 45:
            print("⚠️  Win rate is low (<45%). Consider:")
            print("   - Increasing confidence threshold (min 0.75)")
            print("   - Tightening entry conditions (higher ADX, stronger trend)")
            print("   - Adding HTF trend filter requirement")
        
        if total_trades > 30:
            print("⚠️  Too many trades (>30 in 7 days). Consider:")
            print("   - Reducing max positions (current: {})".format(config.max_positions if config else "?"))
            print("   - Increasing minimum confidence threshold")
            print("   - Adding cooldown period between trades")
        
        if abs(avg_loss) > abs(avg_win):
            print("⚠️  Average loss > average win. Consider:")
            print("   - Widening take profit (3x ATR instead of 2x)")
            print("   - Tightening stop loss (1x ATR instead of 1.5x)")
            print("   - Better entry timing")
        
        if total_pnl < 0:
            print("⚠️  Negative PnL. Consider:")
            print("   - Switching to paper trading mode")
            print("   - Reviewing strategy parameters")
            print("   - Analyzing losing trades for patterns")
        
        print("\n" + "="*80)
        print("Analysis complete!")
        print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(analyze())
