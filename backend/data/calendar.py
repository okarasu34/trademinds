"""
MyFXBook Economic Calendar — XML Feed
Ücretsiz, login gerektirmez.
https://www.myfxbook.com/calendar_statement.xml

Impact: 1=Low, 2=Medium, 3=High
"""
import aiohttp
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger
from db.redis_client import cache_set, cache_get

MYFXBOOK_XML = "https://www.myfxbook.com/calendar_statement.xml"

CURRENCY_SYMBOLS = {
    "USD": ["EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","XAUUSD","US500","NAS100","US30"],
    "EUR": ["EURUSD","EURGBP","EURJPY","EURCHF","EURAUD","EURCAD"],
    "GBP": ["GBPUSD","EURGBP","GBPJPY","GBPCHF","GBPAUD","GBPCAD"],
    "JPY": ["USDJPY","EURJPY","GBPJPY","AUDJPY","CADJPY"],
    "AUD": ["AUDUSD","EURAUD","GBPAUD","AUDJPY","AUDCAD","AUDNZD"],
    "CAD": ["USDCAD","EURCAD","GBPCAD","CADJPY","AUDCAD"],
    "CHF": ["USDCHF","EURCHF","GBPCHF"],
    "NZD": ["NZDUSD","NZDJPY","AUDNZD"],
    "CNY": ["USDCNH"],
}

IMPACT_MAP = {"1": "low", "2": "medium", "3": "high", "": "low"}


class MyFXBookCalendar:

    async def get_calendar(
        self,
        hours_ahead: int = 24,
        impact_filter: Optional[list] = None,
        currency_filter: Optional[list] = None,
    ) -> list:
        cache_key = f"myfxbook:xml:{hours_ahead}"
        cached = await cache_get(cache_key)
        if cached:
            return self._filter(cached, impact_filter, currency_filter)

        events = await self._fetch_xml(hours_ahead)
        if events:
            await cache_set(cache_key, events, ttl=300)

        return self._filter(events, impact_filter, currency_filter)

    async def _fetch_xml(self, hours_ahead: int) -> list:
        now = datetime.utcnow()
        end = now + timedelta(hours=hours_ahead)
        params = {
            "start": now.strftime("%Y-%m-%d %H:%M"),
            "end": end.strftime("%Y-%m-%d %H:%M"),
        }
        headers = {"User-Agent": "Mozilla/5.0 (compatible; TradeMinds/1.0)"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    MYFXBOOK_XML, params=params, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15), ssl=False,
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"MyFXBook XML {resp.status}")
                        return []
                    text = await resp.text()
            return self._parse_xml(text, now)
        except Exception as e:
            logger.error(f"MyFXBook XML error: {e}")
            return []

    def _parse_xml(self, xml_text: str, now: datetime) -> list:
        events = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            return []

        for item in root.findall(".//event"):
            try:
                date_str = item.findtext("date", "").strip()
                if not date_str:
                    continue
                try:
                    scheduled = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    scheduled = datetime.strptime(date_str, "%Y-%m-%d %H:%M")

                currency = item.findtext("currency", "").strip().upper()
                impact_raw = item.findtext("impact", "1").strip()

                events.append({
                    "id": item.findtext("id", ""),
                    "title": item.findtext("name", "").strip(),
                    "country": item.findtext("country", "").strip(),
                    "currency": currency,
                    "impact": IMPACT_MAP.get(impact_raw, "low"),
                    "scheduled_at": scheduled.isoformat(),
                    "minutes_until": round((scheduled - now).total_seconds() / 60, 0),
                    "previous": item.findtext("previous", "").strip(),
                    "forecast": item.findtext("forecast", "").strip(),
                    "actual": item.findtext("actual", "").strip(),
                    "affected_symbols": CURRENCY_SYMBOLS.get(currency, []),
                })
            except Exception:
                continue

        events.sort(key=lambda x: x["minutes_until"])
        logger.info(f"MyFXBook: {len(events)} events")
        return events

    def _filter(self, events, impact_filter, currency_filter):
        result = events
        if impact_filter:
            result = [e for e in result if e["impact"] in impact_filter]
        if currency_filter:
            result = [e for e in result if e["currency"] in currency_filter]
        return result

    async def get_upcoming_high_impact(self, minutes_ahead: int = 60) -> list:
        events = await self.get_calendar(hours_ahead=2, impact_filter=["high"])
        return [e for e in events if 0 <= e.get("minutes_until", 999) <= minutes_ahead]

    async def get_events_for_symbol(self, symbol: str, hours_ahead: int = 4) -> list:
        all_events = await self.get_calendar(hours_ahead=hours_ahead)
        return [e for e in all_events if symbol in e.get("affected_symbols", [])]

    async def is_available(self) -> bool:
        try:
            await self._fetch_xml(1)
            return True
        except Exception:
            return False


calendar_client = MyFXBookCalendar()
