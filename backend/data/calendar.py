"""
Finnhub Economic Calendar — JSON API
Ücretsiz, API key gerektirir (free tier: 60 req/min).
https://finnhub.io/api/v1/calendar/economic

Eski MyFXBook XML feed Cloudflare 403 veriyordu, Finnhub'a geçildi.
Interface aynı kaldı → calendar_client.get_calendar() vb. değişmedi.
"""
import aiohttp
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger
from db.redis_client import cache_set, cache_get
from core.config import settings

FINNHUB_URL = "https://finnhub.io/api/v1/calendar/economic"

# Finnhub ülke kodu → para birimi eşleşmesi
COUNTRY_CURRENCY = {
    "US": "USD", "EU": "EUR", "GB": "GBP", "JP": "JPY",
    "AU": "AUD", "CA": "CAD", "CH": "CHF", "NZ": "NZD",
    "CN": "CNY", "DE": "EUR", "FR": "EUR", "IT": "EUR",
    "ES": "EUR", "NL": "EUR", "BE": "EUR", "AT": "EUR",
    "FI": "EUR", "IE": "EUR", "PT": "EUR", "GR": "EUR",
    "NO": "NOK", "SE": "SEK", "DK": "DKK", "PL": "PLN",
    "TR": "TRY", "MX": "MXN", "BR": "BRL", "ZA": "ZAR",
    "IN": "INR", "KR": "KRW", "SG": "SGD", "HK": "HKD",
}

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

# Finnhub impact: low, medium, high
IMPACT_MAP = {"low": "low", "medium": "medium", "high": "high"}


class FinnhubCalendar:

    async def get_calendar(
        self,
        hours_ahead: int = 24,
        impact_filter: Optional[list] = None,
        currency_filter: Optional[list] = None,
    ) -> list:
        cache_key = f"finnhub:calendar:{hours_ahead}"
        cached = await cache_get(cache_key)
        if cached:
            return self._filter(cached, impact_filter, currency_filter)

        events = await self._fetch_api(hours_ahead)
        if events:
            await cache_set(cache_key, events, ttl=300)

        return self._filter(events, impact_filter, currency_filter)

    async def _fetch_api(self, hours_ahead: int) -> list:
        now = datetime.utcnow()
        start = now.strftime("%Y-%m-%d")
        end = (now + timedelta(hours=hours_ahead)).strftime("%Y-%m-%d")

        params = {
            "from": start,
            "to": end,
            "token": settings.FINNHUB_API_KEY,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    FINNHUB_URL, params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Finnhub API {resp.status}")
                        return []
                    data = await resp.json()
            return self._parse_response(data, now, hours_ahead)
        except Exception as e:
            logger.error(f"Finnhub API error: {e}")
            return []

    def _parse_response(self, data: dict, now: datetime, hours_ahead: int) -> list:
        events = []
        raw_events = data.get("economicCalendar", [])

        for item in raw_events:
            try:
                time_str = item.get("time", "")
                if not time_str:
                    continue

                try:
                    scheduled = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    scheduled = datetime.strptime(time_str, "%Y-%m-%d")

                minutes_until = round((scheduled - now).total_seconds() / 60, 0)

                # Sadece hours_ahead içindeki eventleri al
                if minutes_until > hours_ahead * 60:
                    continue

                country = item.get("country", "").upper()
                currency = COUNTRY_CURRENCY.get(country, "")
                impact = item.get("impact", "low") or "low"

                # actual ve estimate değerlerini string'e çevir (uyumluluk için)
                actual_val = item.get("actual")
                estimate_val = item.get("estimate")
                prev_val = item.get("prev")
                unit = item.get("unit", "")

                actual_str = f"{actual_val}{unit}" if actual_val is not None else ""
                forecast_str = f"{estimate_val}{unit}" if estimate_val is not None else ""
                previous_str = f"{prev_val}{unit}" if prev_val is not None else ""

                events.append({
                    "id": f"{country}:{item.get('event', '')[:30]}:{time_str}",
                    "title": item.get("event", "").strip(),
                    "country": country,
                    "currency": currency,
                    "impact": IMPACT_MAP.get(impact, "low"),
                    "scheduled_at": scheduled.isoformat(),
                    "minutes_until": minutes_until,
                    "previous": previous_str,
                    "forecast": forecast_str,
                    "actual": actual_str,
                    "affected_symbols": CURRENCY_SYMBOLS.get(currency, []),
                })
            except Exception:
                continue

        events.sort(key=lambda x: x["minutes_until"])
        logger.info(f"Finnhub: {len(events)} events")
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
            events = await self._fetch_api(1)
            return True
        except Exception:
            return False


# Aynı isim — import eden dosyalar değişmeyecek
calendar_client = FinnhubCalendar()