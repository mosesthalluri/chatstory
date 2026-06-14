import json
import secrets
from datetime import datetime
from pathlib import Path

from ..settings import JOBS_DIR, settings


PAYMENTS_FILE = JOBS_DIR.parent / "payments.json"
SCREENSHOT_DIR = JOBS_DIR.parent / "payment_screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def price_for(export_type: str) -> int:
    return settings.COMBINED_EXPORT_PRICE if export_type == "combined" else settings.SINGLE_EXPORT_PRICE


def _load() -> list[dict]:
    if not PAYMENTS_FILE.exists():
        return []
    return json.loads(PAYMENTS_FILE.read_text(encoding="utf-8"))


def _save(records: list[dict]) -> None:
    PAYMENTS_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")


def create_intent(job_id: str, product: str, email: str, export_type: str,
                  amount: int | None = None) -> dict:
    records = _load()
    existing = next((r for r in records if r["job_id"] == job_id and r["email"] == email.strip().lower()), None)
    if existing and existing["status"] in {"pending", "submitted", "verified"}:
        return existing
    record = {
        "id": secrets.token_urlsafe(12),
        "job_id": job_id,
        "product": product,
        "email": email.strip().lower(),
        "export_type": export_type,
        "amount": amount if amount else price_for(export_type),
        "status": "pending",
        "transaction_id": "",
        "screenshot_path": "",
        # Payment provider + gateway order id (used by razorpay; blank for manual).
        "provider": "",
        "order_id": "",
        # 6-char access code (OTP) the user saves to re-open the download page
        # after approval, without needing to log in.
        "access_code": secrets.token_hex(3).upper(),
        # Optional note the admin sends back to the user (shown on the page).
        "admin_message": "",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "verified_at": "",
        "verified_by": "",
    }
    records.append(record)
    _save(records)
    return record


def submit_payment(job_id: str, transaction_id: str, screenshot: bytes | None, filename: str = "") -> dict:
    records = _load()
    record = next((r for r in records if r["job_id"] == job_id and r["status"] in {"pending", "submitted"}), None)
    if record is None:
        raise ValueError("Unlock request not found")
    record["transaction_id"] = transaction_id.strip()
    record["status"] = "submitted"
    if screenshot:
        safe_name = Path(filename or "screenshot.jpg").name
        path = SCREENSHOT_DIR / f"{record['id']}_{safe_name}"
        path.write_bytes(screenshot)
        record["screenshot_path"] = str(path)
    record["updated_at"] = datetime.now().isoformat()
    _save(records)
    return record


def verify(payment_id: str, admin_email: str, approved: bool) -> dict:
    records = _load()
    record = next((r for r in records if r["id"] == payment_id), None)
    if record is None:
        raise ValueError("Payment not found")
    record["status"] = "verified" if approved else "rejected"
    record["verified_by"] = admin_email
    record["verified_at"] = datetime.now().isoformat()
    record["updated_at"] = record["verified_at"]
    _save(records)
    return record


def set_order(payment_id: str, provider: str, order_id: str) -> dict | None:
    """Attach a payment-gateway order id (e.g. Razorpay) to a record."""
    records = _load()
    record = next((r for r in records if r["id"] == payment_id), None)
    if record is None:
        return None
    record["provider"] = provider
    record["order_id"] = order_id
    record["updated_at"] = datetime.now().isoformat()
    _save(records)
    return record


def verify_by_order(order_id: str, transaction_id: str, source: str) -> dict | None:
    """Mark the record matching a gateway order id as verified (used after a
    successful, signature-verified Razorpay checkout)."""
    records = _load()
    record = next((r for r in records if r.get("order_id") == order_id), None)
    if record is None:
        return None
    record["transaction_id"] = transaction_id
    record["status"] = "verified"
    record["verified_by"] = source
    record["verified_at"] = datetime.now().isoformat()
    record["updated_at"] = record["verified_at"]
    _save(records)
    return record


def is_unlocked(job_id: str) -> bool:
    return any(r["job_id"] == job_id and r["status"] == "verified" for r in _load())


def for_job(job_id: str) -> dict | None:
    records = [r for r in _load() if r["job_id"] == job_id]
    return sorted(records, key=lambda r: r["updated_at"], reverse=True)[0] if records else None


def get(payment_id: str) -> dict | None:
    return next((r for r in _load() if r["id"] == payment_id), None)


def set_admin_message(payment_id: str, message: str) -> dict | None:
    records = _load()
    record = next((r for r in records if r["id"] == payment_id), None)
    if record is None:
        return None
    record["admin_message"] = message.strip()
    record["updated_at"] = datetime.now().isoformat()
    _save(records)
    return record


def set_admin_message_by_job(job_id: str, message: str) -> dict | None:
    """Convenience for the Telegram '/msg <job_id> <text>' command."""
    rec = for_job(job_id)
    return set_admin_message(rec["id"], message) if rec else None


def check_access(job_id: str, code: str) -> bool:
    """True if there's a VERIFIED payment for this job whose access code
    matches (case-insensitive). Lets a paying user re-open downloads with
    just their OTP, no login required."""
    code = (code or "").strip().upper()
    if not code:
        return False
    for r in _load():
        if (r["job_id"] == job_id and r.get("status") == "verified"
                and (r.get("access_code") or "").upper() == code):
            return True
    return False


def all_payments() -> list[dict]:
    return sorted(_load(), key=lambda r: r["updated_at"], reverse=True)


def macro_verify(amount: int, transaction_id: str, reference: str, source: str = "macrodroid") -> dict | None:
    """Future MacroDroid hook: match by job id/reference and amount."""
    records = _load()
    for record in records:
        if record["job_id"] == reference and record["amount"] == amount and record["status"] in {"pending", "submitted"}:
            record["transaction_id"] = transaction_id
            record["status"] = "verified"
            record["verified_by"] = source
            record["verified_at"] = datetime.now().isoformat()
            record["updated_at"] = record["verified_at"]
            _save(records)
            return record
    return None
