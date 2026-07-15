"""
Memory Manager — unified async API over two-tier memory.
- Short-term: Redis (with in-memory fallback)
- Long-term:  PostgreSQL (with JSON-file fallback)
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional
import asyncio
import logging

from .short_term_memory import ShortTermMemory
from .long_term_memory import LongTermMemory, close_pool, configure_pool

logger = logging.getLogger(__name__)


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

        if db_kwargs:
            configure_pool(**db_kwargs)

        logger.info("MemoryManager initialized user=%s session=%s db=%s", user_id, session_id, db_enabled)

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
                "statistics": self.short_term.get_statistics(),
            },
            "long_term": {
                "preferences": await self.long_term.get_preference(),
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

        prefs = await self.long_term.get_preference()
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
        logger.info("Session ended: %s", self.session_id)

    # ------------------------------------------------------------------
    # LLM summary (long-term)
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
            logger.info("Long-term summary generated (%d chars)", len(summary))
            return summary.strip()
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
