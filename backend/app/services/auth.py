import base64
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime

from ..settings import JOBS_DIR, settings


USERS_FILE = JOBS_DIR.parent / "users.json"


def _load() -> list[dict]:
    if not USERS_FILE.exists():
        return []
    return json.loads(USERS_FILE.read_text(encoding="utf-8"))


def _save(users: list[dict]) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _hash(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()
    return f"{salt}:{digest}"


def _verify(password: str, stored: str) -> bool:
    salt, _ = stored.split(":", 1)
    return hmac.compare_digest(_hash(password, salt), stored)


def create_user(email: str, password: str) -> dict:
    email = email.strip().lower()
    if not email or "@" not in email:
        raise ValueError("Enter a valid email address")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters")
    users = _load()
    if any(u["email"] == email for u in users):
        raise ValueError("Email already registered")
    # Admin if this is the very first account OR the email matches the
    # configured ADMIN_EMAIL. Otherwise a normal user.
    configured_admin = (settings.ADMIN_EMAIL or "").strip().lower()
    role = "admin" if (not users or email == configured_admin) else "user"
    user = {
        "id": secrets.token_urlsafe(12),
        "email": email,
        "password_hash": _hash(password),
        "role": role,
        "created_at": datetime.now().isoformat(),
    }
    users.append(user)
    _save(users)
    return {k: v for k, v in user.items() if k != "password_hash"}


def seed_admin() -> dict | None:
    """Create or promote the configured admin account at startup.

    If ADMIN_EMAIL/ADMIN_PASSWORD are set in .env and that user does not
    exist yet, create it as an admin. If the user exists but isn't admin,
    promote it. This guarantees a known admin login regardless of who
    signed up first. No-op if the settings are blank.
    """
    email = (settings.ADMIN_EMAIL or "").strip().lower()
    password = settings.ADMIN_PASSWORD or ""
    if not email or not password:
        return None
    users = _load()
    existing = next((u for u in users if u["email"] == email), None)
    if existing:
        if existing.get("role") != "admin":
            existing["role"] = "admin"
            _save(users)
        return {k: v for k, v in existing.items() if k != "password_hash"}
    user = {
        "id": secrets.token_urlsafe(12),
        "email": email,
        "password_hash": _hash(password),
        "role": "admin",
        "created_at": datetime.now().isoformat(),
    }
    users.append(user)
    _save(users)
    return {k: v for k, v in user.items() if k != "password_hash"}


def authenticate(email: str, password: str) -> dict | None:
    for user in _load():
        if user["email"] == email.strip().lower() and _verify(password, user["password_hash"]):
            return {k: v for k, v in user.items() if k != "password_hash"}
    return None


def all_users() -> list[dict]:
    return [{k: v for k, v in user.items() if k != "password_hash"} for user in _load()]


def make_token(user: dict, ttl_seconds: int = 60 * 60 * 24 * 14) -> str:
    payload = {"email": user["email"], "role": user["role"], "exp": int(time.time()) + ttl_seconds}
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    sig = hmac.new(settings.SECRET_KEY.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def read_token(token: str | None) -> dict | None:
    if not token or "." not in token:
        return None
    body, sig = token.rsplit(".", 1)
    expected = hmac.new(settings.SECRET_KEY.encode(), body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(body.encode()))
    except (ValueError, json.JSONDecodeError):
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload
