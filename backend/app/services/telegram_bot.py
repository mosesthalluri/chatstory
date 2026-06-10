"""
Telegram payment-approval bot (optional).

When TELEGRAM_BOT_TOKEN + TELEGRAM_ADMIN_CHAT_ID are set, a payment screenshot
is pushed to the admin's Telegram with Approve / Reject buttons. The admin can
approve right from the phone, and message the user back with `/msg <job> ...`
(the note shows on the user's unlock/download page).

Uses long-polling (getUpdates) in a background task, so it needs NO public
webhook URL. Everything is wrapped so a Telegram outage never affects the app;
if the bot isn't configured, the /admin web flow is used instead.
"""

from __future__ import annotations

import asyncio
import json
import traceback

import httpx

from ..settings import settings
from . import payments, jobs

_task: asyncio.Task | None = None
_stop: asyncio.Event | None = None


def enabled() -> bool:
    return bool(settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_ADMIN_CHAT_ID)


def _url(method: str) -> str:
    return f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/{method}"


async def _call(method: str, *, data: dict | None = None, files: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(_url(method), data=data, files=files)
        return resp.json()


def _approve_keyboard(payment_id: str) -> str:
    return json.dumps({"inline_keyboard": [[
        {"text": "✅ Approve", "callback_data": f"approve:{payment_id}"},
        {"text": "❌ Reject", "callback_data": f"reject:{payment_id}"},
    ]]})


async def send_payment_for_review(payment: dict, screenshot: bytes | None = None,
                                  filename: str = "screenshot.jpg") -> None:
    """Push a submitted payment to the admin chat with action buttons."""
    if not enabled():
        return
    caption = (
        f"💸 Payment to review\n"
        f"Product: {payment.get('product')}\n"
        f"Email: {payment.get('email')}\n"
        f"Amount: ₹{payment.get('amount')}\n"
        f"Txn: {payment.get('transaction_id') or '—'}\n"
        f"Job: {payment.get('job_id')}\n\n"
        f"Reply to the user with:\n/msg {payment.get('job_id')} your message"
    )
    markup = _approve_keyboard(payment["id"])
    try:
        if screenshot:
            await _call("sendPhoto",
                        data={"chat_id": settings.TELEGRAM_ADMIN_CHAT_ID,
                              "caption": caption, "reply_markup": markup},
                        files={"photo": (filename or "screenshot.jpg", screenshot)})
        else:
            await _call("sendMessage",
                        data={"chat_id": settings.TELEGRAM_ADMIN_CHAT_ID,
                              "text": caption, "reply_markup": markup})
    except Exception as exc:
        print(f"[telegram] send failed: {exc}")


async def _send_text(text: str) -> None:
    try:
        await _call("sendMessage", data={"chat_id": settings.TELEGRAM_ADMIN_CHAT_ID, "text": text})
    except Exception:
        pass


def _decide(payment_id: str, approved: bool) -> str:
    record = payments.verify(payment_id, "telegram", approved)
    if record is None:
        return "Payment not found."
    if approved and record["status"] == "verified":
        jobs.update(record["job_id"], paid=True)
        code = record.get("access_code", "")
        return f"✅ Approved. The user can download now (access code {code})."
    return "❌ Marked as rejected."


async def _handle_update(update: dict) -> None:
    # Inline button taps
    cq = update.get("callback_query")
    if cq:
        data = cq.get("data", "")
        if ":" in data:
            action, pid = data.split(":", 1)
            msg = _decide(pid, action == "approve")
            try:
                await _call("answerCallbackQuery",
                            data={"callback_query_id": cq["id"], "text": msg})
            except Exception:
                pass
            await _send_text(msg)
        return

    # Text commands (admin chat only)
    message = update.get("message") or {}
    text = (message.get("text") or "").strip()
    chat_id = str((message.get("chat") or {}).get("id", ""))
    if not text or chat_id != str(settings.TELEGRAM_ADMIN_CHAT_ID):
        return

    if text.startswith("/approve "):
        await _send_text(_decide(text.split(maxsplit=1)[1].strip(), True))
    elif text.startswith("/reject "):
        await _send_text(_decide(text.split(maxsplit=1)[1].strip(), False))
    elif text.startswith("/msg "):
        parts = text.split(maxsplit=2)
        if len(parts) >= 3:
            rec = payments.set_admin_message_by_job(parts[1].strip(), parts[2].strip())
            await _send_text("✅ Message saved for the user." if rec else "Job not found.")
        else:
            await _send_text("Usage: /msg <job_id> your message")
    elif text in ("/start", "/help"):
        await _send_text("Send /approve <id>, /reject <id>, or /msg <job_id> <text>. "
                         "Payment screenshots arrive here with Approve/Reject buttons.")


async def _poll_loop() -> None:
    offset = 0
    # Drain any backlog so we don't reprocess old updates on restart.
    try:
        first = await _call("getUpdates", data={"timeout": 0, "offset": -1})
        for u in first.get("result", []):
            offset = max(offset, u["update_id"] + 1)
    except Exception:
        pass

    while _stop and not _stop.is_set():
        try:
            resp = await _call("getUpdates", data={"timeout": 25, "offset": offset})
            for update in resp.get("result", []):
                offset = max(offset, update["update_id"] + 1)
                try:
                    await _handle_update(update)
                except Exception:
                    print(f"[telegram] update error:\n{traceback.format_exc()}")
        except Exception:
            await asyncio.sleep(5)  # network blip — back off and retry


async def start() -> None:
    global _task, _stop
    if not enabled() or _task is not None:
        return
    _stop = asyncio.Event()
    _task = asyncio.create_task(_poll_loop())
    print("[telegram] approval bot started (long-polling)")


async def stop() -> None:
    global _task, _stop
    if _stop:
        _stop.set()
    if _task:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
    _task = None
