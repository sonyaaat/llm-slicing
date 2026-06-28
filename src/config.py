"""Central configuration: loads environment variables and exposes the OpenAI API key.

Import this module wherever the API key is needed:

    from config import OPENAI_API_KEY

It raises a clear error at import time if the key is missing or empty.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load the .env file from the project root (one level up from src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing or empty.\n"
        f"Open {ENV_PATH} and set:\n"
        "    OPENAI_API_KEY=sk-...your-real-key...\n"
        "Then re-run. (Do not commit .env — it is git-ignored.)"
    )
