"""
Memory Manager — unified async API over two-tier memory + Redis caching.
- Short-term: Redis List (with in-memory fallback)
- Long-term:  PostgreSQL (with JSON-file fallback)
- Cache:      Redis Hash (preferences) + Redis String (LLM summary)
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional
import asyncio
import hashlib
import json
import logging

from .short_term_memory import ShortTermMemory
from .long_term_memory import LongTermMemory, close_pool, configure_pool

logger = logging.getLogger(__name__)

PREF_CACHE_TTL_SEC = 86400     # preference cache: 24 hours
SUMMARY_CACHE_TTL_SEC = 1800    # LLM summary cache: 30 minutes


class MemoryManager:

    def __init__(
        self,
        user_id: str,
        session_id: str,
        storage_path: str = "data/memory",
        llm_model=None,
        redis_url: Optional[str] = None,
        db_enabled: bool = False,
        **db_kwargs: Any,
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.llm_model = llm_model
        self._redis = None

        self.short_term = ShortTermMemory(
            user_id=user_id,
            session_id=session_id,
            redis_url=redis_url,
        )
        self.long_term = LongTermMemory(
            user_id=user_id,
            storage_path=storage_path,
            db_enabled=db_enabled,
            **db_kwargs,
        )

        if redis_url:
            self._try_connect_redis(redis_url)

        if db_kwargs:
            configure_pool(**db_kwargs)

        logger.info("MemoryManager initialized user=%s session=%s db=%s", user_id, session_id, db_enabled)

    # ------------------------------------------------------------------
    # Redis connection
    # ------------------------------------------------------------------

    def _try_connect_redis(self, url: str) -> None:
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(url, decode_responses=False)
            logger.info("MemoryManager: connected to Redis for caching")
        except Exception as exc:
            logger.warning("MemoryManager: Redis unavailable (%s), caching disabled", exc)
            self._redis = None

    async def _redis_ok(self) -> bool:
        if self._redis is None:
            return False
        try:
            await self._redis.ping()
            return True
        except Exception:
            self._redis = None
            return False

    # ------------------------------------------------------------------
    # Preference cache (Cache-Aside on Redis Hash)
    # ------------------------------------------------------------------

    async def _get_cached_preferences(self) -> Dict[str, Any]:
        """Read preferences: Redis Hash first, fallback to LongTermMemory."""
        cache_key = f"prefs:{self.user_id}"

        if await self._redis_ok():
            raw = await self._redis.hgetall(cache_key)
            if raw:
                return {
                    k.decode() if isinstance(k, bytes) else k:
                    json.loads(v.decode() if isinstance(v, bytes) else v)
                    for k, v in raw.items()
                }

        # cache miss — read from storage
        prefs = await self.long_term.get_preference()

        # write back to cache
        if await self._redis_ok() and prefs:
            pipe = self._redis.pipeline()
            pipe.delete(cache_key)
            for k, v in prefs.items():
                if v:
                    pipe.hset(cache_key, k, json.dumps(v, ensure_ascii=False))
            pipe.expire(cache_key, PREF_CACHE_TTL_SEC)
            await pipe.execute()

        return prefs

    async def invalidate_preference_cache(self) -> None:
        """Delete cached preferences (call after any preference write)."""
        if await self._redis_ok():
            await self._redis.delete(f"prefs:{self.user_id}")

    # ------------------------------------------------------------------
    # LLM summary cache (Redis String + content-hash key)
    # ------------------------------------------------------------------

    def _summary_cache_key(self, history_str: str, trip_str: str) -> str:
        h = hashlib.md5(history_str.encode()).hexdigest()
        t = hashlib.md5(trip_str.encode()).hexdigest()
        return f"summary:{self.user_id}:{h}:{t}"

    async def _get_cached_summary(self, cache_key: str) -> Optional[str]:
        if not await self._redis_ok():
            return None
        raw = await self._redis.get(cache_key)
        return raw.decode() if raw else None

    async def _set_cached_summary(self, cache_key: str, summary: str) -> None:
        if await self._redis_ok():
            await self._redis.setex(cache_key, SUMMARY_CACHE_TTL_SEC, summary)

    # ------------------------------------------------------------------
    # Short-term ops
    # ------------------------------------------------------------------

    async def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        await self.short_term.add_message(role, content, metadata)
        await self.long_term.add_chat_message(role, content, self.session_id)

    # ------------------------------------------------------------------
    # Composite queries
    # ------------------------------------------------------------------

    async def get_full_context(self) -> Dict[str, Any]:
        return {
            "short_term": {
                "recent_dialogue": await self.short_term.get_recent_context(5),
                "context_string": await self.short_term.get_context_string(5),
                "statistics": await self.short_term.get_statistics(),
            },
            "long_term": {
                "preferences": await self._get_cached_preferences(),
                "chat_history": await self.long_term.get_chat_history(10),
                "trip_history": await self.long_term.get_trip_history(5),
                "frequent_destinations": await self.long_term.get_frequent_destinations(3),
                "statistics": await self.long_term.get_statistics(),
            },
        }

    async def get_context_for_agent(self, long_term_summary: Optional[str] = None) -> str:
        lines: List[str] = []

        if long_term_summary:
            lines.append("【历史会话总结】")
            lines.append(long_term_summary)
            lines.append("")

        prefs = await self._get_cached_preferences()
        has_prefs = any(v for v in prefs.values() if v)
        if has_prefs:
            lines.append("【用户偏好】")
            for key, value in prefs.items():
                if value:
                    lines.append(f"- {key}: {value}")
            lines.append("")

        ctx = await self.short_term.get_context_string(3)
        if ctx != "无历史对话":
            lines.append("【当前会话对话】")
            lines.append(ctx)
            lines.append("")

        return "\n".join(lines) if lines else "无上下文信息"

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    async def end_session(self) -> None:
        await self.short_term.clear()
        await self.short_term.close()
        logger.info("Session ended: %s", self.session_id)

    # ------------------------------------------------------------------
    # LLM summary (long-term) — with Redis content-hash cache
    # ------------------------------------------------------------------

    async def get_long_term_summary_async(self, max_messages: int = 50) -> str:
        if not self.llm_model:
            return ""

        all_history = await self.long_term.get_chat_history(limit=max_messages)
        history_from_other = [m for m in all_history if m.get("session_id") != self.session_id]
        trip_history = await self.long_term.get_trip_history(limit=20)

        if not history_from_other and not trip_history:
            return ""

        history_str = "\n".join(
            f"[{m.get('timestamp', '')}] {m['role']}: {m['content']}" for m in history_from_other[-max_messages:]
        ) or "（无聊天记录）"

        trip_lines = []
        for t in trip_history:
            origin = t.get("origin", "未知")
            dest = t.get("destination", "未知")
            sd = t.get("start_date", "")
            ed = t.get("end_date", "")
            purpose = t.get("purpose", "旅游")
            ts = t.get("timestamp", "")
            if sd and ed:
                trip_lines.append(f"[{ts}] {origin} -> {dest} ({sd} 至 {ed}) - {purpose}")
            elif sd:
                trip_lines.append(f"[{ts}] {origin} -> {dest} ({sd}) - {purpose}")
            else:
                trip_lines.append(f"[{ts}] {origin} -> {dest} - {purpose}")
        trip_str = "\n".join(trip_lines) if trip_lines else "（无行程记录）"

        # ---- Redis content-hash cache ----
        cache_key = self._summary_cache_key(history_str, trip_str)
        cached = await self._get_cached_summary(cache_key)
        if cached:
            logger.info("Summary cache hit for user=%s", self.user_id)
            return cached

        prompt = (
            "请总结以下历史信息中的关键内容，包括：\n"
            "1. 用户的旅行偏好和习惯\n2. 用户询问过的重要问题\n"
            "3. 用户的出行历史和目的地\n4. 其他重要的上下文信息\n\n"
            f"【历史聊天记录】\n{history_str}\n\n【历史行程记录】\n{trip_str}\n\n"
            "请用简洁的语言总结（不超过200字）："
        )

        try:
            response = await self.llm_model([{"role": "user", "content": prompt}])
            summary = ""
            if hasattr(response, '__aiter__'):
                async for chunk in response:
                    if isinstance(chunk, str):
                        summary = chunk
                    elif hasattr(chunk, 'content'):
                        c = chunk.content
                        if isinstance(c, str):
                            summary = c
                        elif isinstance(c, list):
                            for item in c:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    summary = item.get('text', '')
            elif hasattr(response, 'content'):
                summary = str(response.content)
            else:
                summary = str(response)

            summary = summary.strip()
            if summary:
                await self._set_cached_summary(cache_key, summary)

            logger.info("Long-term summary generated (%d chars)", len(summary))
            return summary
        except Exception as exc:
            logger.error("Failed to generate long-term summary: %s", exc)
            return ""

    def get_long_term_summary(self, max_messages: int = 50) -> str:
        """Sync wrapper — called only from sync contexts."""
        try:
            loop = asyncio.get_running_loop()
            logger.warning("get_long_term_summary called from async context, use _async version")
            return ""
        except RuntimeError:
            return asyncio.run(self.get_long_term_summary_async(max_messages))
