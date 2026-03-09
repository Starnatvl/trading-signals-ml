# integration/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print("✅ Загружены переменные из .env")
else:
    print("ℹ️ Файл .env не найден, используется демо-режим (мок-сервер)")

def is_production() -> bool:
    """Возвращает True, если не включён принудительный мок и заданы все production-переменные."""
    if os.getenv("USE_MOCK", "").lower() == "true":
        return False
    return bool(os.getenv("API_BASE_URL") and os.getenv("API_USERNAME") and os.getenv("API_PASSWORD"))

def get_mode_name() -> str:
    return "PRODUCTION" if is_production() else "DEMO (mock)"

if is_production():
    API_BASE = os.getenv("API_BASE_URL").rstrip('/')
    API_USERNAME = os.getenv("API_USERNAME")
    API_PASSWORD = os.getenv("API_PASSWORD")
else:
    API_BASE = os.getenv("API_BASE", "http://localhost:3000")
    API_USERNAME = None
    API_PASSWORD = None

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60" if is_production() else "5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))
SIGNAL_SOURCE = os.getenv("SIGNAL_SOURCE", "ml_shadow")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))