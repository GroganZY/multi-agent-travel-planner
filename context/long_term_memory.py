"""
Long-term Memory
- PostgreSQL mode: structured persistence with JSONB, async queries
- Fallback: local JSON file (zero-dependency, same behavior as legacy)
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PostgreSQL connection pool (module-level singleton, lazy-init)
# ---------------------------------------------------------------------------

_pool = None
_pool_kwargs: Dict[str, Any] = {}


async def _get_pool() -> Any:
    """Return the module-level asyncpg connection pool, creating it on first call."""
    global _pool
    if _pool is not None:
        return _pool
    try:
        import asyncpg
        _pool = await asyncpg.create_pool(
            host=_pool_kwargs.get("host", "localhost"),
            port=_pool_kwargs.get("port", 5432),
            database=_pool_kwargs.get("database", "travel_planner"),
            user=_pool_kwargs.get("user", "travel"),
            password=_pool_kwargs.get("password", "travel123"),
            min_size=_pool_kwargs.get("min_size", 2),
            max_size=_pool_kwargs.get("max_size", 10),
        )
        logger.info("LongTermMemory: PostgreSQL connection pool created")
        return _pool
    except Exception as exc:
        logger.warning("LongTermMemory: PostgreSQL unavailable (%s)", exc)
        return None


def configure_pool(**kwargs: Any) -> None:
    """Set PostgreSQL connection parameters before first use."""
    _pool_kwargs.update(kwargs)


async def close_pool() -> None:
    """Close the connection pool (call on shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("LongTermMemory: connection pool closed")


# ---------------------------------------------------------------------------
# SQL templates
# ---------------------------------------------------------------------------

_INSERT_USER = """
    INSERT INTO users (user_id) VALUES ($1) ON CONFLICT DO NOTHING
"""

_UPSERT_PREFERENCE = """
    INSERT INTO preferences (user_id, pref_type, pref_value, updated_at)
    VALUES ($1, $2, $3::jsonb, NOW())
    ON CONFLICT (user_id, pref_type)
    DO UPDATE SET pref_value = $3::jsonb, updated_at = NOW()
"""

_GET_PREFERENCES = """
    SELECT pref_type, pref_value FROM preferences WHERE user_id = $1
"""

_GET_PREFERENCE_BY_TYPE = """
    SELECT pref_value FROM preferences WHERE user_id = $1 AND pref_type = $2
"""

_INSERT_CHAT = """
    INSERT INTO chat_history (user_id, session_id, role, content)
    VALUES ($1, $2, $3, $4)
"""

_GET_CHAT_HISTORY = """
    SELECT role, content, session_id, created_at AS timestamp
    FROM chat_history
    WHERE user_id = $1
    ORDER BY created_at DESC
    LIMIT $2
"""

_INSERT_TRIP = """
    INSERT INTO trip_history (user_id, trip_id, origin, destination, start_date, end_date, purpose)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
"""

_GET_TRIP_HISTORY = """
    SELECT trip_id, origin, destination, start_date, end_date, purpose
    FROM trip_history
    WHERE user_id = $1
    ORDER BY created_at DESC
    LIMIT $2
"""

_GET_FREQUENT_DESTINATIONS = """
    SELECT destination, COUNT(*) AS cnt
    FROM trip_history
    WHERE user_id = $1 AND destination IS NOT NULL
    GROUP BY destination
    ORDER BY cnt DESC
    LIMIT $2
"""

_COUNT_TRIPS = """
    SELECT COUNT(*) FROM trip_history WHERE user_id = $1
"""

_COUNT_MESSAGES = """
    SELECT COUNT(*) FROM chat_history WHERE user_id = $1
"""

_DELETE_HISTORY = """
    DELETE FROM chat_history WHERE user_id = $1;
    DELETE FROM trip_history WHERE user_id = $1;
"""


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------

class LongTermMemory:

    def __init__(
        self,
        user_id: str,
        storage_path: str = "data/memory",
        db_enabled: bool = False,
        **db_kwargs: Any,
    ):
        self.user_id = user_id
        self._db_enabled = db_enabled
        self._pool: Any = None

        # file fallback path
        self._file_path = Path(storage_path) / f"{user_id}.json"
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_data: Optional[Dict[str, Any]] = None

        if db_kwargs:
            configure_pool(**db_kwargs)

    async def _ensure_pool(self) -> Any:
        """Lazily connect to PostgreSQL. Returns None if unavailable."""
        if not self._db_enabled:
            return None
        if self._pool is not None:
            return self._pool
        self._pool = await _get_pool()
        if self._pool is not None:
            await self._ensure_tables()
            await self._ensure_user()
        return self._pool

    async def _ensure_tables(self) -> None:
        """Create tables if they don't exist (idempotent, safe to call every init)."""
        pool = self._pool
        if pool is None:
            return
        try:
            sql_path = Path(__file__).parent.parent / "migrations" / "init.sql"
            if sql_path.exists():
                ddl = sql_path.read_text(encoding="utf-8")
                await pool.execute(ddl)
        except Exception as exc:
            logger.warning("LongTermMemory: table init failed (%s)", exc)

    async def _ensure_user(self) -> None:
        if self._pool is None:
            return
        try:
            await self._pool.execute(_INSERT_USER, self.user_id)
        except Exception as exc:
            logger.warning("LongTermMemory: ensure user failed (%s)", exc)

    @property
    def _is_db(self) -> bool:
        return self._pool is not None

    # ------------------------------------------------------------------
    # File helpers (fallback)
    # ------------------------------------------------------------------

    def _load_file(self) -> Dict[str, Any]:
        if self._file_data is not None:
            return self._file_data
        if self._file_path.exists():
            try:
                self._file_data = json.loads(self._file_path.read_text(encoding="utf-8"))
                return self._file_data
            except Exception:
                pass
        self._file_data = self._init_file_data()
        return self._file_data

    def _save_file(self) -> None:
        if self._file_data is not None:
            self._file_data["updated_at"] = datetime.now().isoformat()
            self._file_path.write_text(json.dumps(self._file_data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _init_file_data(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "preferences": [],
            "chat_history": [],
            "trip_history": [],
            "statistics": {"total_trips": 0, "total_messages": 0, "frequent_destinations": {}},
        }

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    async def save_preference(self, pref_type: str, value: Any) -> None:
        pool = await self._ensure_pool()
        if pool is not None:
            await pool.execute(_UPSERT_PREFERENCE, self.user_id, pref_type, json.dumps(value, ensure_ascii=False))
            return

        data = self._load_file()
        prefs: list = data["preferences"]
        for p in prefs:
            if p.get("type") == pref_type:
                p["value"] = value
                self._save_file()
                return
        prefs.append({"type": pref_type, "value": value})
        self._save_file()

    async def get_preference(self, pref_type: Optional[str] = None) -> Any:
        pool = await self._ensure_pool()
        if pool is not None:
            if pref_type is not None:
                row = await pool.fetchrow(_GET_PREFERENCE_BY_TYPE, self.user_id, pref_type)
                return json.loads(row["pref_value"]) if row else None
            rows = await pool.fetch(_GET_PREFERENCES, self.user_id)
            return {r["pref_type"]: json.loads(r["pref_value"]) for r in rows}

        data = self._load_file()
        prefs: list = data["preferences"]
        if pref_type is not None:
            for p in prefs:
                if p.get("type") == pref_type:
                    return p.get("value")
            return None
        return {p.get("type"): p.get("value") for p in prefs}

    async def add_hotel_brand(self, brand: str) -> None:
        cur = await self.get_preference("hotel_brands") or []
        if not isinstance(cur, list):
            cur = [cur] if cur else []
        if brand not in cur:
            cur.append(brand)
        await self.save_preference("hotel_brands", cur)

    async def add_airline(self, airline: str) -> None:
        cur = await self.get_preference("airlines") or []
        if not isinstance(cur, list):
            cur = [cur] if cur else []
        if airline not in cur:
            cur.append(airline)
        await self.save_preference("airlines", cur)

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------

    async def add_chat_message(self, role: str, content: str, session_id: Optional[str] = None) -> None:
        pool = await self._ensure_pool()
        if pool is not None:
            await pool.execute(_INSERT_CHAT, self.user_id, session_id, role, content)
            return

        data = self._load_file()
        data["chat_history"].append({
            "role": role, "content": content,
            "timestamp": datetime.now().isoformat(), "session_id": session_id,
        })
        data["statistics"]["total_messages"] += 1
        self._save_file()

    async def get_chat_history(self, limit: Optional[int] = None, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        pool = await self._ensure_pool()
        if pool is not None:
            sql = """SELECT role, content, session_id, created_at AS timestamp
                     FROM chat_history WHERE user_id = $1"""
            params: list = [self.user_id]
            if session_id:
                sql += " AND session_id = $2"
                params.append(session_id)
            sql += " ORDER BY created_at DESC"
            if limit:
                sql += f" LIMIT {int(limit)}"
            rows = await pool.fetch(sql, *params)
            return [dict(r) for r in rows]

        data = self._load_file()
        msgs = data["chat_history"]
        if session_id:
            msgs = [m for m in msgs if m.get("session_id") == session_id]
        return msgs[-limit:] if limit else msgs

    # ------------------------------------------------------------------
    # Trip history
    # ------------------------------------------------------------------

    async def save_trip_history(self, trip_info: Dict[str, Any]) -> None:
        pool = await self._ensure_pool()
        trip_id = trip_info.get("trip_id") or f"trip_{int(datetime.now().timestamp())}"
        if pool is not None:
            await pool.execute(
                _INSERT_TRIP,
                self.user_id, trip_id,
                trip_info.get("origin"),
                trip_info.get("destination"),
                trip_info.get("start_date"),
                trip_info.get("end_date"),
                trip_info.get("purpose"),
            )
            return

        data = self._load_file()
        data["trip_history"].append({"trip_id": trip_id, "timestamp": datetime.now().isoformat(), **trip_info})
        data["statistics"]["total_trips"] += 1
        dest = trip_info.get("destination")
        if dest:
            freq = data["statistics"]["frequent_destinations"]
            freq[dest] = freq.get(dest, 0) + 1
        self._save_file()

    async def get_trip_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        pool = await self._ensure_pool()
        if pool is not None:
            rows = await pool.fetch(_GET_TRIP_HISTORY, self.user_id, limit)
            return [dict(r) for r in rows]

        data = self._load_file()
        trips = data["trip_history"]
        return trips[-limit:] if limit else trips

    async def get_frequent_destinations(self, top_n: int = 5) -> List[Tuple[str, int]]:
        pool = await self._ensure_pool()
        if pool is not None:
            rows = await pool.fetch(_GET_FREQUENT_DESTINATIONS, self.user_id, top_n)
            return [(r["destination"], r["cnt"]) for r in rows]

        data = self._load_file()
        freq: dict = data["statistics"].get("frequent_destinations", {})
        return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def get_statistics(self) -> Dict[str, Any]:
        pool = await self._ensure_pool()
        if pool is not None:
            trips = await pool.fetchval(_COUNT_TRIPS, self.user_id) or 0
            msgs = await pool.fetchval(_COUNT_MESSAGES, self.user_id) or 0
            return {"total_trips": trips, "total_messages": msgs}

        data = self._load_file()
        return data.get("statistics", {}).copy()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def clear_history(self) -> None:
        """Clear chat + trip history, keep preferences."""
        pool = await self._ensure_pool()
        if pool is not None:
            await pool.execute(_DELETE_HISTORY, self.user_id)
            return

        data = self._load_file()
        data["chat_history"] = []
        data["trip_history"] = []
        data["statistics"]["total_trips"] = 0
        data["statistics"]["total_messages"] = 0
        data["statistics"]["frequent_destinations"] = {}
        self._save_file()

    async def close(self) -> None:
        """Release resources (file mode is a no-op, pool is shared)."""
        pass  # pool is module-level singleton, closed via close_pool()
