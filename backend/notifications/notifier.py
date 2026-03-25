import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from loguru import logger
from core.config import settings


class Notifier:

    def __init__(self, user_id: str):
        self.user_id = user_id

    async def send(self, message: str, level: str = "info"):
        await self._send_telegram(message)
        if level in ("error", "critical"):
            await self._send_email("TradeMinds Alert", message)

    async def send_trade_opened(self, trade, signal: dict):
        emoji = "🟢" if trade.side.value == "buy" else "🔴"
        msg = (
            f"{emoji} TRADE OPENED\n"
            f"Symbol: {trade.symbol}\n"
            f"Side: {trade.side.value.upper()}\n"
            f"Entry: {trade.entry_price}\n"
            f"SL: {trade.stop_loss} | TP: {trade.take_profit}\n"
            f"Lot: {trade.lot_size}\n"
            f"Confidence: {signal.get('confidence', 0):.0%}\n"
            f"Reason: {signal.get('reasoning', '')[:100]}"
        )
        await self.send(msg)

    async def send_trade_closed(self, trade):
        emoji = "💰" if (trade.pnl or 0) >= 0 else "📉"
        msg = (
            f"{emoji} TRADE CLOSED\n"
            f"Symbol: {trade.symbol}\n"
            f"P&L: {trade.pnl:+.2f} {trade.currency}\n"
            f"Closed by: {trade.closed_by}\n"
            f"Duration: {self._calc_duration(trade)}"
        )
        await self.send(msg)

    async def send_daily_limit_warning(self, current_pct: float, limit_pct: float):
        msg = (
            f"⚠️ DAILY LOSS WARNING\n"
            f"Current: {current_pct:.1f}% | Limit: {limit_pct:.1f}%\n"
            f"Approaching daily loss limit!"
        )
        await self.send(msg, level="error")

    async def send_high_impact_news(self, event: dict):
        msg = (
            f"📰 HIGH IMPACT NEWS IN {event.get('minutes_until', '?')} MIN\n"
            f"{event.get('title')}\n"
            f"Currency: {event.get('currency')} | "
            f"Forecast: {event.get('forecast', 'N/A')} | "
            f"Previous: {event.get('previous', 'N/A')}\n"
            f"Bot paused until news passes."
        )
        await self.send(msg, level="error")

    async def _send_telegram(self, message: str):
        if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
            return
        try:
            url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
            async with aiohttp.ClientSession() as session:
                await session.post(url, json={
                    "chat_id": settings.TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "HTML",
                }, timeout=aiohttp.ClientTimeout(total=5))
        except Exception as e:
            logger.warning(f"Telegram notification failed: {e}")

    async def _send_email(self, subject: str, body: str):
        if not settings.SMTP_USER or not settings.NOTIFICATION_EMAIL:
            return
        try:
            msg = MIMEMultipart()
            msg["From"] = settings.SMTP_USER
            msg["To"] = settings.NOTIFICATION_EMAIL
            msg["Subject"] = f"[TradeMinds] {subject}"
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                server.starttls()
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.send_message(msg)
        except Exception as e:
            logger.warning(f"Email notification failed: {e}")

    def _calc_duration(self, trade) -> str:
        if trade.opened_at and trade.closed_at:
            delta = trade.closed_at - trade.opened_at
            hours = delta.seconds // 3600
            minutes = (delta.seconds % 3600) // 60
            return f"{hours}h {minutes}m"
        return "N/A"
