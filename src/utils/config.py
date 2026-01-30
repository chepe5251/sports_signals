import os
from dotenv import load_dotenv

load_dotenv()

# GEMINI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# API FOOTBALL
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")

# TELEGRAM
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# DATABASE (Windows Authentication)
DB_SERVER = os.getenv("DB_SERVER", "localhost")
DB_NAME = os.getenv("DB_NAME", "FootballAnalytics")
DB_TRUSTED = os.getenv("DB_TRUSTED", "true").lower() == "true"
