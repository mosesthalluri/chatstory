"""
Payment provider abstraction (V2).

Two providers share the same payment records (services/payments.py) and the
same unlock logic — only *how the user pays* differs:

  manual    — UPI/Paytm: user pays to a UPI id / scans a QR, uploads a
              screenshot, an admin (or the Telegram bot) verifies it.
  razorpay  — Razorpay Checkout. We create an order via Razorpay's REST API
              (httpx, no SDK dependency) and verify the payment signature
              client-side on return (HMAC-SHA256, stdlib) — so no public
              webhook endpoint is required, matching the same constraint the
              Telegram approval bot works under.

Switching providers is purely an .env change (PAYMENT_PROVIDER=...). Manual is
the default and is always available as a fallback.
"""

from __future__ import annotations

import hashlib
import hmac

from ..settings import settings


def active() -> str:
    return (settings.PAYMENT_PROVIDER or "manual").strip().lower()


def is_razorpay() -> bool:
    """Razorpay is only 'on' when selected AND keys are present — otherwise
    we transparently fall back to the manual flow so payments never break."""
    return active() == "razorpay" and bool(settings.RAZORPAY_KEY and settings.RAZORPAY_SECRET)


def upi_id() -> str:
    return settings.UPI_ID or settings.PAYTM_UPI_ID


def qr_image() -> str:
    return settings.PAYMENT_QR_PATH or settings.PAYTM_QR_IMAGE


async def create_order(amount_inr: int, receipt: str, email: str = "") -> dict | None:
    """Create a Razorpay order for the given amount (INR). Returns the order
    JSON (with `id`) or None if Razorpay isn't configured / the call fails.
    Uses the REST API directly so we don't depend on the razorpay SDK."""
    if not is_razorpay():
        return None
    try:
        import httpx
    except Exception:
        return None
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.razorpay.com/v1/orders",
                auth=(settings.RAZORPAY_KEY, settings.RAZORPAY_SECRET),
                json={
                    "amount": int(amount_inr) * 100,  # paise
                    "currency": "INR",
                    "receipt": receipt[:40],
                    "payment_capture": 1,
                    "notes": {"receipt": receipt, "email": email},
                },
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:  # network / auth error — fall back to manual
        print(f"[payment_provider] razorpay order failed: {exc}")
        return None


def verify_checkout_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """Verify Razorpay Checkout's success payload. Razorpay signs
    `<order_id>|<payment_id>` with HMAC-SHA256 using your key secret."""
    if not (order_id and payment_id and signature and settings.RAZORPAY_SECRET):
        return False
    expected = hmac.new(
        settings.RAZORPAY_SECRET.encode(),
        f"{order_id}|{payment_id}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
