from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Slack
    slack_bot_token: str
    slack_app_token: str

    # Gemini
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    # ServiceNow
    sn_instance_url: str
    sn_username: str
    sn_password: str

    # Bot identity (for self-assignment)
    sn_bot_user_sys_id: str = ""  # env var: SN_BOT_USER_SYS_ID

    # Channel configuration
    help_channel_id: str = ""  # env var: HELP_CHANNEL_ID

    # Qdrant vector database
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "it_kb"
    qdrant_public_collection: str = "it_kb_public"

    # Google Custom Search Engine (for public article fallback)
    google_cse_api_key: str = ""
    google_cse_cx: str = ""

    # IT helpdesk channel for article approvals
    it_helpdesk_channel_id: str = ""

    # Debug channel for bot reasoning traces
    debug_channel_name: str = "servicedesk-bot-debug"

    # Public KB
    public_trust_threshold: int = 5

    # Recommendation approval gate
    recommendation_trust_threshold: int = 5

    # SMTP (used by dashboard for MFA)
    smtp_email: str = ""
    smtp_app_password: str = ""

    # Dashboard
    dashboard_url: str = "http://localhost:8050"

    # Database
    db_path: str = "interactions.db"

    # Misc
    log_level: str = "INFO"
    max_tool_loops: int = 10


def get_settings() -> Settings:
    return Settings()
