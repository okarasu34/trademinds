from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
from loguru import logger
import asyncio, json, sys, os

from core.config import settings
from core.middleware import RequestLoggingMiddleware, RateLimitMiddleware, BruteForceProtectionMiddleware, SecurityHeadersMiddleware
from db.database import init_db
from db.redis_client import init_redis, close_redis, get_redis
from bot.scheduler import setup_scheduler

os.makedirs("/var/log/trademinds", exist_ok=True)
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")
logger.add("/var/log/trademinds/backend.log", rotation="50 MB", retention="30 days", level="DEBUG", catch=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TradeMinds starting...")
    await init_db()
    await init_redis()
    setup_scheduler()
    logger.info(f"TradeMinds v{settings.APP_VERSION} ready")
    yield
    await close_redis()
    logger.info("TradeMinds shutdown")

app = FastAPI(title="TradeMinds API", version=settings.APP_VERSION,
    docs_url="/api/docs" if settings.DEBUG else None, redoc_url=None, lifespan=lifespan)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(BruteForceProtectionMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

from api.auth import router as auth_router
from api.bot import router as bot_router
from api.trades import router as trades_router
from api.strategies import router as strategies_router
from api.backtests import router as backtests_router
from api.reports import router as reports_router
from api.brokers import router as brokers_router
from api.calendar import router as calendar_router
from api.dashboard import router as dashboard_router

P = settings.API_PREFIX
app.include_router(auth_router,       prefix=f"{P}/auth",       tags=["Auth"])
app.include_router(bot_router,        prefix=f"{P}/bot",        tags=["Bot"])
app.include_router(trades_router,     prefix=f"{P}/trades",     tags=["Trades"])
app.include_router(strategies_router, prefix=f"{P}/strategies", tags=["Strategies"])
app.include_router(backtests_router,  prefix=f"{P}/backtests",  tags=["Backtests"])
app.include_router(reports_router,    prefix=f"{P}/reports",    tags=["Reports"])
app.include_router(brokers_router,    prefix=f"{P}/brokers",    tags=["Brokers"])
app.include_router(calendar_router,   prefix=f"{P}/calendar",   tags=["Calendar"])
app.include_router(dashboard_router,  prefix=f"{P}/dashboard",  tags=["Dashboard"])

class WSManager:
    def __init__(self): self.connections: dict[str, list[WebSocket]] = {}
    async def connect(self, ws, uid): await ws.accept(); self.connections.setdefault(uid, []).append(ws)
    def disconnect(self, ws, uid):
        s = self.connections.get(uid, [])
        if ws in s: s.remove(ws)
    async def broadcast(self, uid, msg):
        for ws in list(self.connections.get(uid, [])):
            try: await ws.send_json(msg)
            except: self.disconnect(ws, uid)

manager = WSManager()

@app.websocket("/ws/{user_id}")
async def ws_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    redis = get_redis()
    channels = [f"{ch}:{user_id}" for ch in ["trades","prices","health","signals"]]
    async def listener():
        pub = redis.pubsub(); await pub.subscribe(*channels)
        async for msg in pub.listen():
            if msg["type"] == "message":
                try:
                    ch = msg["channel"].decode() if isinstance(msg["channel"], bytes) else msg["channel"]
                    await manager.broadcast(user_id, {"channel": ch, "data": json.loads(msg["data"])})
                except: pass
    task = asyncio.create_task(listener())
    try:
        while True:
            t = await websocket.receive_text()
            if t == "ping": await websocket.send_text("pong")
    except WebSocketDisconnect: pass
    finally: task.cancel(); manager.disconnect(websocket, user_id)

@app.get("/health", include_in_schema=False)
async def health(): return {"status": "ok", "version": settings.APP_VERSION}
