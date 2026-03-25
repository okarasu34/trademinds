from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger
import time
import uuid

from db.redis_client import get_redis


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with timing."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.time()

        response = await call_next(request)

        duration = round((time.time() - start) * 1000, 1)
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"→ {response.status_code} ({duration}ms)"
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration}ms"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Global rate limiting per IP.
    - 200 req/min for general endpoints
    - 10 req/min for auth endpoints (extra tight)
    """

    LIMITS = {
        "/api/v1/auth/login": (10, 60),
        "/api/v1/auth/register": (5, 60),
        "default": (200, 60),
    }

    async def dispatch(self, request: Request, call_next):
        ip = request.client.host
        path = request.url.path

        limit, window = self.LIMITS.get(path, self.LIMITS["default"])
        key = f"ratelimit:{ip}:{path}"

        redis = get_redis()
        if redis:
            try:
                count = await redis.incr(key)
                if count == 1:
                    await redis.expire(key, window)
                if count > limit:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Too many requests. Please slow down."},
                        headers={"Retry-After": str(window)},
                    )
            except Exception:
                pass  # Redis down → don't block traffic

        return await call_next(request)


class BruteForceProtectionMiddleware(BaseHTTPMiddleware):
    """
    Track failed login attempts per IP.
    After 5 failures → 15 min lockout.
    """

    MAX_FAILURES = 5
    LOCKOUT_SECONDS = 900  # 15 min

    async def dispatch(self, request: Request, call_next):
        if request.url.path != "/api/v1/auth/login" or request.method != "POST":
            return await call_next(request)

        ip = request.client.host
        lockout_key = f"lockout:{ip}"
        fail_key = f"loginfail:{ip}"

        redis = get_redis()
        if redis:
            # Check if locked out
            locked = await redis.get(lockout_key)
            if locked:
                ttl = await redis.ttl(lockout_key)
                return JSONResponse(
                    status_code=429,
                    content={"detail": f"Account locked. Try again in {ttl // 60} minutes."},
                )

        response = await call_next(request)

        if redis and response.status_code == 401:
            failures = await redis.incr(fail_key)
            if failures == 1:
                await redis.expire(fail_key, 300)  # reset counter after 5 min
            if failures >= self.MAX_FAILURES:
                await redis.setex(lockout_key, self.LOCKOUT_SECONDS, "1")
                await redis.delete(fail_key)
                logger.warning(f"IP {ip} locked out after {failures} failed login attempts")

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response
