"""
Centralized settings. Loaded once at startup from .env.

Anywhere in the codebase you need a setting, do:

    from app.settings import settings
    print(settings.GROQ_API_KEY)
"""

from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root and storage paths. Paths are computed once here so the rest
# of the codebase never hardcodes paths.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent
STORAGE_ROOT = BACKEND_ROOT / "storage"
UPLOADS_DIR = STORAGE_ROOT / "uploads"
JOBS_DIR = STORAGE_ROOT / "jobs"
OUTPUT_DIR = STORAGE_ROOT / "output"
TEMPLATES_DIR = PROJECT_ROOT / "frontend" / "templates"
STATIC_DIR = PROJECT_ROOT / "frontend" / "static"

# Make sure all storage directories exist
for path in [UPLOADS_DIR, JOBS_DIR, OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """All settings loaded from .env file."""

    # LLM
    USE_OLLAMA: bool = False
    GROQ_API_KEY: str = ""
    GROQ_MODEL_FAST: str = "llama-3.1-8b-instant"
    GROQ_MODEL_STRONG: str = "llama-3.3-70b-versatile"
    OLLAMA_MODEL_FAST: str = "llama3.1:8b"
    OLLAMA_MODEL_STRONG: str = "llama3.1:8b"
    OLLAMA_HOST: str = "http://localhost:11434"
    # Ollama context window. The commentary writer uses small sequential
    # scene prompts, so 8192 is a better fit for i7/8GB local machines than
    # a large 16K context. Increase only if your model and RAM can handle it.
    OLLAMA_NUM_CTX: int = 8192

    # Images
    # USE_GEMINI_IMAGES = False (default): pick from curated SVG cliparts
    # in frontend/static/cliparts/. Zero API cost, instant, no failure
    # modes. Recommended for development and small-scale production.
    #
    # USE_GEMINI_IMAGES = True: try Gemini AI generation first, fall back
    # to clipart on failure. Set this only if you have paid Gemini access
    # or want to spend your free-tier budget on illustrations.
    USE_GEMINI_IMAGES: bool = False
    GEMINI_API_KEY: str = ""
    IMAGE_STYLE: str = "flat vector cartoon illustration, soft pastel colors"

    # Filesystem (exposed on settings so image_gen can find cliparts)
    PROJECT_ROOT: Path = PROJECT_ROOT
    STATIC_DIR: Path = STATIC_DIR

    # PDF Clipart (the pdf_clipart pipeline, exposed as a web product)
    # backend: "local" (Stable Diffusion via diffusers — needs torch on the
    # server), "api" (remote image service), or "stub" (offline placeholder).
    PDF_CLIPART_BACKEND: str = "local"
    PDF_CLIPART_MODEL: str = "sd-turbo"      # or "sd15-lcm"
    # Reuse a model already on disk instead of downloading: point at a
    # single-file .safetensors/.ckpt (A1111/ComfyUI) or a diffusers folder.
    PDF_CLIPART_MODEL_PATH: str = ""
    PDF_CLIPART_STEPS: int = 2
    PDF_CLIPART_MAX_PER_PAGE: int = 2

    # Gift Engine LLM personalization. Uses the same Ollama/Groq client as
    # ChatStory to infer each person's interests/skills and suggest creative,
    # personal gift ideas (grounded in real quotes). Falls back to the
    # deterministic engine if the model is unavailable.
    GIFT_ENGINE_USE_LLM: bool = True

    # Telegram payment-approval bot (optional). When both are set, payment
    # screenshots are pushed to the admin chat with Approve/Reject buttons and
    # the admin can message the user back. If unset, the /admin web flow is
    # used instead.
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_ADMIN_CHAT_ID: str = ""

    # Book config
    CHAPTERS_PER_BOOK: int = 8
    PREVIEW_CHAPTERS: int = 1
    CURRENCY_SYMBOL: str = "₹"
    FULL_BOOK_PRICE: int = 50
    SINGLE_EXPORT_PRICE: int = 50
    COMBINED_EXPORT_PRICE: int = 75
    PAYTM_UPI_ID: str = "your-paytm-upi@paytm"
    PAYTM_QR_IMAGE: str = "/static/paytm-qr.png"
    ADMIN_EMAIL: str = ""
    ADMIN_PASSWORD: str = ""

    # Storage
    AUTO_DELETE_AFTER_HOURS: int = 24
    MAX_UPLOAD_SIZE_MB: int = 200
    MAX_MESSAGES: int = 300_000

    # Queue (local hardware: keep concurrent pipelines low)
    QUEUE_MAX_CONCURRENT: int = 1
    QUEUE_JOB_TIMEOUT_SECONDS: int = 7200

    # Consumer / trust surface
    SUPPORT_EMAIL: str = "support@chatstory.app"
    # "How to export your chats" tutorial videos (paste full YouTube URLs).
    WHATSAPP_EXPORT_VIDEO: str = ""
    TELEGRAM_EXPORT_VIDEO: str = ""
    INSTAGRAM_EXPORT_VIDEO: str = ""
    REFUND_POLICY: str = "Refunds are considered case by case — email support within 7 days of payment."

    # App
    SECRET_KEY: str = "change-me"
    DEBUG: bool = True

    @field_validator("DEBUG", mode="before")
    @classmethod
    def _parse_debug(cls, value):
        """Accept deployment-style DEBUG values from .env.

        Pydantic already handles true/false, but real env files often use
        words like "release" or "production". Treat those as debug off
        instead of failing app startup.
        """
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production", "off", "no", "0"}:
                return False
            if normalized in {"debug", "dev", "development", "on", "yes", "1"}:
                return True
        return value

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Single shared instance
settings = Settings()
