from dataclasses import dataclass
from typing import Optional
from loguru import logger
from db.models import BotConfig, Trade, OrderStatus
from bot.indicators import calculate_lot_size_from_risk


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str
    adjusted_lot_size: Optional[float] = None


class RiskManager:
    """
    Enforces all user-defined risk limits before any trade is executed.
    This is the gatekeeper — no trade passes without its approval.
    """

    def __init__(self, config: BotConfig):
        self.config = config

    async def check_new_trade(
        self,
        user_id: str,
        market_type: str,
        symbol: str,
        lot_size: float,
        entry_price: float,
        stop_loss: float,
        account_balance: float,
        open_positions: list[Trade],
        upcoming_news: list[dict],
    ) -> RiskCheckResult:

        # 1. Global position limit
        total_open = len([t for t in open_positions if t.status == OrderStatus.OPEN])
        if total_open >= self.config.max_positions:
            return RiskCheckResult(
                allowed=False,
                reason=f"Max positions reached ({total_open}/{self.config.max_positions})"
            )

        # 2. Per-market position limit
        market_limits = self.config.market_limits or {}
        market_limit = market_limits.get(market_type, 999)
        market_open = len([
            t for t in open_positions
            if t.status == OrderStatus.OPEN and t.market_type.value == market_type
        ])
        if market_open >= market_limit:
            return RiskCheckResult(
                allowed=False,
                reason=f"Market limit reached for {market_type} ({market_open}/{market_limit})"
            )

        # 3. Daily loss limit
        if account_balance > 0:
            daily_loss_pct = (self.config.daily_loss / account_balance) * 100
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Daily loss limit hit ({daily_loss_pct:.1f}% >= {self.config.max_daily_loss_pct}%)"
                )

        # 4. Risk per trade limit — recalculate using correct formula
        price_diff = abs(entry_price - stop_loss)
        if price_diff > 0 and account_balance > 0:
            # Recalculate actual risk based on market type
            if market_type == "forex":
                risk_amount = lot_size * price_diff * 100_000
            elif market_type == "crypto":
                risk_amount = lot_size * price_diff
            elif market_type in ("stock", "index"):
                risk_amount = lot_size * price_diff
            elif market_type == "commodity":
                risk_amount = lot_size * price_diff * 100
            else:
                risk_amount = lot_size * price_diff * 100_000

            risk_pct = (risk_amount / account_balance) * 100

            if risk_pct > self.config.max_risk_per_trade_pct:
                # Auto-adjust lot size to fit within limit using canonical function
                adjusted_lot = calculate_lot_size_from_risk(
                    account_balance=account_balance,
                    risk_pct=self.config.max_risk_per_trade_pct,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    market_type=market_type,
                )
                logger.warning(
                    f"Risk {risk_pct:.2f}% > limit {self.config.max_risk_per_trade_pct}%. "
                    f"Adjusting lot from {lot_size} to {adjusted_lot}"
                )
                lot_size = adjusted_lot

        # 5. High-impact news pause
        if self.config.pause_on_high_impact_news and upcoming_news:
            high_impact = [e for e in upcoming_news if e.get("impact") == "high"]
            if high_impact:
                next_event = high_impact[0]
                minutes_until = next_event.get("minutes_until", 999)
                if minutes_until <= self.config.news_pause_minutes:
                    return RiskCheckResult(
                        allowed=False,
                        reason=f"High-impact news in {minutes_until}min: {next_event.get('title', 'Unknown')}"
                    )

        # 6. Duplicate symbol check — don't open same symbol twice
        same_symbol_open = [
            t for t in open_positions
            if t.status == OrderStatus.OPEN and t.symbol == symbol
        ]
        if same_symbol_open:
            return RiskCheckResult(
                allowed=False,
                reason=f"Already have open position on {symbol}"
            )

        return RiskCheckResult(allowed=True, reason="All checks passed", adjusted_lot_size=lot_size)

    def check_daily_limit_warning(self, account_balance: float) -> Optional[str]:
        """Returns warning message if approaching daily limit."""
        if account_balance <= 0:
            return None
        daily_loss_pct = (self.config.daily_loss / account_balance) * 100
        warning_threshold = self.config.max_daily_loss_pct * 0.8  # 80% of limit
        if daily_loss_pct >= warning_threshold:
            return (
                f"WARNING: Daily loss at {daily_loss_pct:.1f}% "
                f"(limit: {self.config.max_daily_loss_pct}%)"
            )
        return None

    def should_emergency_stop(self, account_balance: float) -> bool:
        """Returns True if bot should stop all activity immediately."""
        if account_balance <= 0:
            return True
        daily_loss_pct = (self.config.daily_loss / account_balance) * 100
        return daily_loss_pct >= self.config.max_daily_loss_pct
