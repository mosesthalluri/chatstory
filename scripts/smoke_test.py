"""Quick smoke test of the WhatsApp parser. Has no external dependencies
beyond Python stdlib so it runs anywhere.

Run: python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

# Stub out modules we don't want to import for the smoke test
import types
stub_settings = types.ModuleType("app.settings")
stub_settings.BACKEND_ROOT = Path(__file__).resolve().parent.parent / "backend"
stub_settings.PROJECT_ROOT = stub_settings.BACKEND_ROOT.parent
stub_settings.STORAGE_ROOT = stub_settings.BACKEND_ROOT / "storage"
stub_settings.UPLOADS_DIR = stub_settings.STORAGE_ROOT / "uploads"
stub_settings.JOBS_DIR = stub_settings.STORAGE_ROOT / "jobs"
stub_settings.OUTPUT_DIR = stub_settings.STORAGE_ROOT / "output"
stub_settings.TEMPLATES_DIR = stub_settings.PROJECT_ROOT / "frontend" / "templates"
stub_settings.STATIC_DIR = stub_settings.PROJECT_ROOT / "frontend" / "static"

class _S:
    USE_OLLAMA = False
    GROQ_API_KEY = ""
    GEMINI_API_KEY = ""
stub_settings.settings = _S()
sys.modules["app.settings"] = stub_settings

# Now import parsers
from app.parsers.whatsapp import parse as parse_whatsapp
from app.parsers.detect import detect_format
from app.models import MessageKind

sample_path = Path(__file__).resolve().parent.parent / "backend" / "storage" / "sample_chat.txt"

print(f"Reading: {sample_path}")
print(f"Detected format: {detect_format(sample_path)}")
print()

result = parse_whatsapp(sample_path)
print(f"Senders: {result.senders}")
print(f"Total messages: {result.message_count}")
print(f"Date range: {result.messages[0].timestamp.date()} → {result.messages[-1].timestamp.date()}")
print()

kinds = {}
for m in result.messages:
    kinds[m.kind.value] = kinds.get(m.kind.value, 0) + 1
print(f"Message kinds: {kinds}")
print()

print("First 3 messages:")
for m in result.messages[:3]:
    print(f"  [{m.timestamp.strftime('%Y-%m-%d %H:%M')}] {m.sender}: {m.text}")
print()

print("Warnings:")
for w in result.parser_warnings:
    print(f"  - {w}")

print("\n✓ SMOKE TEST PASSED")
