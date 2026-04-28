import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_PROVIDER: str = "openai"
    MODEL_NAME: str = "gpt-5-nano"

    OPENAI_BASE_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")

    PAGEINDEX_API_KEY: str = os.getenv("PAGEINDEX_API_KEY", "")
