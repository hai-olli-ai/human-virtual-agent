"""Configuration for the Human Virtual Pipecat agent."""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# AI Service keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")

# Human Virtual API
HV_API_URL = os.getenv("HV_API_URL", "http://localhost:3001/api/v1")
HV_API_TOKEN = os.getenv("HV_API_TOKEN", "")

# Daily (managed by Pipecat Cloud, but useful for local Daily testing)
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")

# Default IDs for testing
DEFAULT_AVATAR_ID = os.getenv("DEFAULT_AVATAR_ID", "")
DEFAULT_SCENE_ID = os.getenv("DEFAULT_SCENE_ID", "")
DEFAULT_ROOM_ID = os.getenv("DEFAULT_ROOM_ID", "")

# TTS voice configuration
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121")
# Default: "British Reading Lady" — will be customizable per avatar later

# LLM model — must support vision for scene understanding (Session 46)
# gpt-4.1 and gpt-4o both support vision
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1")
