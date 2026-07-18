"""
Configuration for the Multi-Agent Travel Planner
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# LLM Configuration
LLM_CONFIG = {
    "api_key": os.getenv("LLM_API_KEY", ""),
    "model_name": "doubao-seed-2-0-mini-260428",
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "temperature": 0.7,
    "max_tokens": 8192,
}

# System Configuration
SYSTEM_CONFIG = {
    "enable_llm": True,  # Set to True to use LLM (recommended), False for rule-based
    "log_level": "INFO",
    "max_retries": 3,
    "timeout": 60,  # Increased timeout for better stability
}

# RAG 知识库：嵌入模型（本地路径，无需连 HuggingFace）
RAG_CONFIG = {
    "embedding_model": "data/models/bge-small-zh-v1.5",
}

# 连接与可用性：重试、熔断、健康检查
RESILIENCE_CONFIG = {
    "max_retries": 3,              # 单次请求最大重试次数（与 SYSTEM_CONFIG 对齐）
    "retry_base_delay_sec": 1.0,   # 重试退避基数（秒）
    "retry_max_delay_sec": 30.0,   # 重试退避上限（秒）
    "circuit_failure_threshold": 5, # 连续失败多少次后熔断
    "circuit_recovery_timeout_sec": 60.0,  # 熔断后多少秒进入半开
    "circuit_half_open_successes": 2,      # 半开状态下连续成功多少次后关闭
    "health_check_timeout_sec": 10.0,      # 健康检查请求超时（秒）
}

# PostgreSQL — Long-term memory persistence
# 默认为 True，启动时自动探测：Docker 开着就用，没开自动降级为本地 JSON 文件
# 若想强制关闭（即使 Docker 开着也不走 DB），设为 False
DB_CONFIG = {
    "enabled": True,
    "host": "localhost",
    "port": 5432,
    "database": "travel_planner",
    "user": "travel",
    "password": "travel123",
    "min_size": 2,
    "max_size": 10,
}

# Redis — Short-term memory cache + preference/summary cache layer
# 默认为 True，启动时自动探测：Docker 开着就用，没开自动降级为 Python list
# 若想强制关闭（即使 Docker 开着也不走 Redis），设为 False
CACHE_CONFIG = {
    "enabled": True,
    "url": "redis://localhost:6379/0",
    "preferences_ttl_sec": 86400,     # 24 hours
    "summary_ttl_sec": 1800,           # 30 minutes
    "short_term_ttl_sec": 3600,        # 1 hour
}
