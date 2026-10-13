"""API key authentication, role-based authorisation and per-key rate limiting."""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from dataclasses import dataclass

from fastapi import Request

from idengine.config import ApiSettings, Role
from idengine.errors import AuthenticationError, AuthorizationError, RateLimitError


@dataclass(frozen=True, slots=True)
class Principal:
    key_id: str  # short, non-reversible fingerprint used in logs and rate limiting
    role: Role


def fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


class TokenBucket:
    """Classic token bucket: ``rate`` tokens per second, bursts up to ``capacity``."""

    def __init__(self, rate: float, capacity: int, *, clock: object = time.monotonic) -> None:
        self.rate = rate
        self.capacity = capacity
        self._clock = clock
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def take(self, key: str) -> float:
        """Consume one token; returns 0 on success or seconds until a token is available."""
        now = self._clock()  # type: ignore[operator]
        with self._lock:
            tokens, updated = self._state.get(key, (float(self.capacity), now))
            tokens = min(float(self.capacity), tokens + (now - updated) * self.rate)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, now)
                return 0.0
            self._state[key] = (tokens, now)
            return (1.0 - tokens) / self.rate if self.rate > 0 else float("inf")


class Authenticator:
    def __init__(self, settings: ApiSettings) -> None:
        self.enabled = settings.auth_enabled
        # Store only digests; compare digests in constant time.
        self._keys = {
            hashlib.sha256(key.encode("utf-8")).digest(): (fingerprint(key), role)
            for key, role in settings.key_roles().items()
        }
        rate = settings.rate_limit_per_minute / 60.0
        self.limiter = (
            TokenBucket(rate, settings.rate_limit_burst) if settings.rate_limit_per_minute else None
        )

    @staticmethod
    def _extract_key(request: Request) -> str | None:
        key = request.headers.get("x-api-key")
        if key:
            return key.strip()
        authorization = request.headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
        return None

    def authenticate(self, request: Request) -> Principal:
        if not self.enabled:
            principal = Principal("anonymous", Role.ADMIN)
        else:
            key = self._extract_key(request)
            if not key:
                raise AuthenticationError("missing API key (X-API-Key header or Bearer token)")
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            match: tuple[str, Role] | None = None
            for known, value in self._keys.items():
                if hmac.compare_digest(known, digest):
                    match = value
            if match is None:
                raise AuthenticationError("invalid API key")
            principal = Principal(*match)
        if self.limiter is not None:
            client = request.client.host if request.client else "unknown"
            bucket = principal.key_id if self.enabled else f"ip:{client}"
            wait = self.limiter.take(bucket)
            if wait > 0:
                raise RateLimitError("rate limit exceeded", retry_after=wait)
        return principal


def authorize(principal: Principal, required: Role) -> None:
    if not principal.role.allows(required):
        raise AuthorizationError(
            f"this operation requires the '{required.value}' role",
            details={"role": principal.role.value, "required": required.value},
        )
