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

    # Channel configuration
    help_channel_id: str = ""  # env var: HELP_CHANNEL_ID

    # Qdrant vector database
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "it_kb"

    # Misc
    log_level: str = "INFO"
    max_tool_loops: int = 10


def get_settings() -> Settings:
    return Settings()
