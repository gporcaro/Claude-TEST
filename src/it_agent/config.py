from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Slack
    slack_bot_token: str
    slack_app_token: str

    # Anthropic
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-5-20250929"

    # Database
    db_path: Path = Path("tickets.db")

    # Knowledge base
    chroma_path: Path = Path("chroma_data")
    knowledge_docs_path: Path = Path("src/it_agent/docs")

    # Misc
    log_level: str = "INFO"
    max_tool_loops: int = 10


def get_settings() -> Settings:
    return Settings()
