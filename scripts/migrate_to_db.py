#!/usr/bin/env python
"""
Migrate existing JSON memory files → PostgreSQL.

Usage:
    python scripts/migrate_to_db.py

Before running: docker compose up -d
After running:  set DB_CONFIG["enabled"] = True in config.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from context.long_term_memory import _INSERT_USER, _UPSERT_PREFERENCE, _INSERT_CHAT, _INSERT_TRIP

STORAGE_DIR = project_root / "data" / "memory"
PREFERENCE_TYPES = {
    "home_location", "transportation_preference", "hotel_brands",
    "airlines", "seat_preference", "meal_preference", "budget_level",
}


async def migrate_user(pool, user_id: str, data: dict) -> dict:
    stats = {"preferences": 0, "chat": 0, "trips": 0}

    await pool.execute(_INSERT_USER, user_id)

    # Preferences (list format: [{type, value}])
    prefs = data.get("preferences", [])
    if isinstance(prefs, list):
        for item in prefs:
            ptype = item.get("type", "")
            pval = item.get("value")
            if ptype and pval is not None:
                await pool.execute(
                    _UPSERT_PREFERENCE,
                    user_id, ptype, json.dumps(pval, ensure_ascii=False),
                )
                stats["preferences"] += 1

    # Chat history
    for msg in data.get("chat_history", []):
        await pool.execute(
            _INSERT_CHAT,
            user_id,
            msg.get("session_id", "migrated"),
            msg.get("role", "user"),
            msg.get("content", ""),
        )
        stats["chat"] += 1

    # Trip history
    for trip in data.get("trip_history", []):
        await pool.execute(
            _INSERT_TRIP,
            user_id,
            trip.get("trip_id", f"trip_{stats['trips']}"),
            trip.get("origin"),
            trip.get("destination"),
            trip.get("start_date"),
            trip.get("end_date"),
            trip.get("purpose"),
        )
        stats["trips"] += 1

    return stats


async def main():
    print("=" * 60)
    print("JSON → PostgreSQL 数据迁移")
    print("=" * 60)
    print()

    if not STORAGE_DIR.exists():
        print("未找到 data/memory 目录，无需迁移。")
        return

    json_files = sorted(STORAGE_DIR.glob("*.json"))
    if not json_files:
        print("未找到 .json 文件，无需迁移。")
        return

    print(f"发现 {len(json_files)} 个 JSON 文件:")
    for f in json_files:
        print(f"  - {f.name}")
    print()

    # Connect to PostgreSQL
    try:
        import asyncpg
        pool = await asyncpg.create_pool(
            host="localhost",
            port=5432,
            database="travel_planner",
            user="travel",
            password="travel123",
            min_size=2,
            max_size=5,
        )
    except ImportError:
        print("请先安装 asyncpg: pip install asyncpg")
        return
    except Exception as e:
        print(f"无法连接 PostgreSQL: {e}")
        print("请先启动 Docker: docker compose up -d")
        return

    try:
        # Ensure tables exist
        init_sql = project_root / "migrations" / "init.sql"
        if init_sql.exists():
            await pool.execute(init_sql.read_text(encoding="utf-8"))
            print("✓ 数据库表已就绪\n")

        total = {"preferences": 0, "chat": 0, "trips": 0}

        for fp in json_files:
            user_id = fp.stem  # filename minus .json
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                print(f"  ⚠ 跳过损坏文件: {fp.name}")
                continue

            stats = await migrate_user(pool, user_id, data)
            for k in stats:
                total[k] += stats[k]
            print(f"  ✓ {fp.name}: {stats['preferences']} pref, {stats['chat']} chat, {stats['trips']} trip")

        print()
        print("=" * 60)
        print(f"迁移完成: {total['preferences']} 条偏好, {total['chat']} 条聊天, {total['trips']} 条行程")
        print("=" * 60)
        print()
        print("下一步: 将 config.py 中 DB_CONFIG['enabled'] 设为 True")

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
