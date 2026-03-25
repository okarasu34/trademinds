import redis.asyncio as aioredis
from core.config import settings
from loguru import logger
import json
from typing import Any, Optional

redis_client: Optional[aioredis.Redis] = None


async def init_redis():
    global redis_client
    redis_client = aioredis.from_url(
        settings.REDIS_URL,
        max_connections=settings.REDIS_POOL_SIZE,
        decode_responses=True,
    )
    await redis_client.ping()
    logger.info("Redis connected")


async def close_redis():
    global redis_client
    if redis_client:
        await redis_client.close()


def get_redis() -> aioredis.Redis:
    return redis_client


# ─────────────────────── Cache helpers ───────────────────────

async def cache_set(key: str, value: Any, ttl: int = 60):
    await redis_client.setex(key, ttl, json.dumps(value))


async def cache_get(key: str) -> Optional[Any]:
    data = await redis_client.get(key)
    return json.loads(data) if data else None


async def cache_delete(key: str):
    await redis_client.delete(key)


async def cache_delete_pattern(pattern: str):
    keys = await redis_client.keys(pattern)
    if keys:
        await redis_client.delete(*keys)


# ─────────────────────── Pub/Sub for WebSocket ───────────────────────

async def publish(channel: str, message: Any):
    await redis_client.publish(channel, json.dumps(message))


async def subscribe(channel: str):
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(channel)
    return pubsub


# ─────────────────────── Rate limiting ───────────────────────

async def check_rate_limit(key: str, limit: int, window: int) -> bool:
    """Returns True if under limit, False if exceeded."""
    current = await redis_client.incr(key)
    if current == 1:
        await redis_client.expire(key, window)
    return current <= limit


# ─────────────────────── Bot state ───────────────────────

async def set_bot_state(user_id: str, state: dict):
    await cache_set(f"bot:state:{user_id}", state, ttl=3600)


async def get_bot_state(user_id: str) -> Optional[dict]:
    return await cache_get(f"bot:state:{user_id}")


async def set_live_price(symbol: str, price: float, bid: float, ask: float):
    await cache_set(f"price:{symbol}", {
        "price": price, "bid": bid, "ask": ask
    }, ttl=10)


async def get_live_price(symbol: str) -> Optional[dict]:
    return await cache_get(f"price:{symbol}")


async def set_open_positions_count(user_id: str, count: int):
    await redis_client.setex(f"positions:count:{user_id}", 3600, str(count))


async def get_open_positions_count(user_id: str) -> int:
    val = await redis_client.get(f"positions:count:{user_id}")
    return int(val) if val else 0
