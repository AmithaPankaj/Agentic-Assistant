"""Central configuration for the agent system.

All settings are pulled from environment variables (see .env.example).
Keeping this in one place makes it easy to swap models/providers later
without touching agent logic.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    model_name: str = os.getenv("AGENT_MODEL", "llama-3.3-70b-versatile")
    max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "6"))
    temperature: float = float(os.getenv("AGENT_TEMPERATURE", "0.2"))
    code_exec_timeout: int = int(os.getenv("CODE_EXEC_TIMEOUT", "10"))
    workspace_dir: str = os.getenv("AGENT_WORKSPACE", "./workspace")


settings = Settings()
