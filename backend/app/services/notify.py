"""
Optional email notifications. No-op unless SMTP_HOST + SMTP_FROM are set, so
the app runs fine without any mail server. The dashboard remains the primary
way users track jobs — email is a convenience, never a dependency.
"""

from __future__ import annotations

import asyncio
import smtplib
from email.message import EmailMessage

from ..settings import settings


def enabled() -> bool:
    return bool(settings.SMTP_HOST and settings.SMTP_FROM)


def _send_sync(to: str, subject: str, body: str) -> None:
    if not enabled() or not to:
        return
    msg = EmailMessage()
    msg["From"] = settings.SMTP_FROM
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20) as s:
            s.starttls()
            if settings.SMTP_USER:
                s.login(settings.SMTP_USER, settings.SMTP_PASS)
            s.send_message(msg)
    except Exception as exc:
        print(f"[notify] email to {to} failed: {exc}")


async def send(to: str, subject: str, body: str) -> None:
    """Fire-and-forget email; never blocks the event loop or raises."""
    if not enabled() or not to:
        return
    try:
        await asyncio.to_thread(_send_sync, to, subject, body)
    except Exception:
        pass


_PRODUCT_NAME = {
    "chatstory": "ChatStory storybook",
    "chat-wrapped": "ChatWrapped",
    "gift-engine": "GiftBook",
    "pdf-clipart": "enhanced PDF",
}


def product_label(product: str | None) -> str:
    return _PRODUCT_NAME.get(product or "", "creation")
