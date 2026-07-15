"""
Short-term Memory
- Redis mode: Redis List with TTL sliding window (distributed, survives restarts)
- Fallback: in-memory Python list (zero-dependency, always available)
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

DEFAULT_REDIS_TTL_SEC = 3600       # 1 hour
DEFAULT_MAX_TURNS = 10


class ShortTermMemory:

    def __init__(
        self,
        user_id: str = "default_user",
        session_id: str = "default",
        max_turns: int = DEFAULT_MAX_TURNS,
        redis_url: Optional[str] = None,
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.max_turns = max_turns
        self._redis = None
        self._fallback: List[Dict[str, Any]] = []

        if redis_url:
            self._try_connect_redis(redis_url)

    # ------------------------------------------------------------------
    # Redis connection
    # ------------------------------------------------------------------

    def _try_connect_redis(self, url: str) -> None:
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(url, decode_responses=False)
            logger.info("ShortTermMemory: connected to Redis")
        except Exception as exc:
            logger.warning("ShortTermMemory: Redis unavailable (%s), using in-memory fallback", exc)
            self._redis = None

    async def _redis_available(self) -> bool:
        if self._redis is None:
            return False
        try:
            await self._redis.ping()
            return True
        except Exception:
            await self.close()
            return False

    def _redis_key(self) -> str:
        return f"stm:{self.user_id}:{self.session_id}"

    # ------------------------------------------------------------------
    # Public API (all async)
    # ------------------------------------------------------------------

    async def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        if await self._redis_available():
            payload = json.dumps(message, ensure_ascii=False)
            key = self._redis_key()
            await self._redis.rpush(key, payload)
            max_msgs = self.max_turns * 2
            await self._redis.ltrim(key, -max_msgs, -1)
            await self._redis.expire(key, DEFAULT_REDIS_TTL_SEC)
        else:
            self._fallback.append(message)
            max_msgs = self.max_turns * 2
            if len(self._fallback) > max_msgs:
                self._fallback = self._fallback[-max_msgs:]

    async def get_recent_context(self, n_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        if await self._redis_available():
            key = self._redis_key()
            raw = await self._redis.lrange(key, 0, -1)
            messages = [json.loads(m) for m in raw]
        else:
            messages = list(self._fallback)

        if n_turns is None:
            return messages

        n_msgs = n_turns * 2
        return messages[-n_msgs:] if len(messages) > n_msgs else messages

    async def get_context_string(self, n_turns: int = 5) -> str:
        messages = await self.get_recent_context(n_turns)
        if not messages:
            return "无历史对话"

        lines = []
        for msg in messages:
            role_name = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role_name}: {msg['content']}")
        return "\n".join(lines)

    async def clear(self) -> None:
        if await self._redis_available():
            await self._redis.delete(self._redis_key())
        else:
            self._fallback.clear()
        logger.info("Short-term memory cleared")

    async def close(self) -> None:
        """Release Redis connection pool."""
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:
                pass
            self._redis = None

    async def get_statistics(self) -> Dict[str, Any]:
        if await self._redis_available():
            key = self._redis_key()
            total = await self._redis.llen(key)
            return {
                "total_messages": total,
                "max_turns": self.max_turns,
                "backend": "redis",
            }
        return {
            "total_messages": len(self._fallback),
            "max_turns": self.max_turns,
            "backend": "memory",
        }
