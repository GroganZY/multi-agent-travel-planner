-- 长期记忆表结构
-- docker-compose 启动时自动执行

CREATE TABLE IF NOT EXISTS users (
    user_id       VARCHAR(64) PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS preferences (
    id            SERIAL PRIMARY KEY,
    user_id       VARCHAR(64) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    pref_type     VARCHAR(128) NOT NULL,
    pref_value    JSONB NOT NULL DEFAULT '{}',
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, pref_type)
);
CREATE INDEX IF NOT EXISTS idx_preferences_user ON preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_preferences_value ON preferences USING GIN (pref_value);

CREATE TABLE IF NOT EXISTS chat_history (
    id            SERIAL PRIMARY KEY,
    user_id       VARCHAR(64) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_id    VARCHAR(64),
    role          VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content       TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(user_id, session_id);

CREATE TABLE IF NOT EXISTS trip_history (
    id            SERIAL PRIMARY KEY,
    user_id       VARCHAR(64) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    trip_id       VARCHAR(64) NOT NULL,
    origin        VARCHAR(128),
    destination   VARCHAR(128),
    start_date    VARCHAR(32),
    end_date      VARCHAR(32),
    purpose       VARCHAR(256),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_trip_history_user ON trip_history(user_id);
CREATE INDEX IF NOT EXISTS idx_trip_history_dest ON trip_history(user_id, destination);
