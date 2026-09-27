"""
Aegis – Core configuration (environment variables).
All settings are loaded from environment / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── App ────────────────────────────────────────────────────────────────
    APP_NAME: str = "Aegis Financial Risk Orchestrator"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"

    # ── Database ───────────────────────────────────────────────────────────
    # SQLite for local dev (zero-setup). Set to postgres:// in .env for production.
    DATABASE_URL: str = "sqlite:///./aegis_local.db"
    # psycopg3 DSN used by LangGraph PostgresSaver (only when using PostgreSQL)
    PSYCOPG_DSN: str = "postgresql://aegis:aegis_pw@localhost:5432/aegis_db"

    # ── LLM ────────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = "sk-placeholder"
    ANTHROPIC_API_KEY: str = ""
    LLM_MODEL: str = "gemini-1.5-pro"  # gemini-* | gpt-* | claude-*

    # ── Plaid ──────────────────────────────────────────────────────────────
    PLAID_CLIENT_ID: str = ""
    PLAID_SECRET: str = ""
    PLAID_ENV: str = "sandbox"  # sandbox | development | production

    # ── Vector DB (Qdrant) ─────────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "aegis_preferences"

    # ── GRPO ──────────────────────────────────────────────────────────────
    GRPO_GROUP_SIZE: int = 8       # Candidate policies per generation
    GRPO_LEARNING_RATE: float = 1e-4
    GRPO_EPOCHS: int = 3


settings = Settings()
