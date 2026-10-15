"""Paths, business rules, Unicode cleanup and heading-based KB chunks."""
from __future__ import annotations
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
EVAL = ROOT / "eval"
CONFIG = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8"))
LEVELS = ["low", "medium", "high", "urgent"]
BASE_PRIORITY = {
    "payment_issue": "high", "complaint": "high",
    **{k: "medium" for k in ["get_refund", "track_refund", "cancel_order",
       "change_order", "change_shipping_address", "track_order", "recover_password",
       "registration_problems", "delete_account", "contact_human_agent"]},
}
URGENCY = re.compile(r"\b(urgent|urgently|asap|immediately|emergency|furious|fraud|scam)\b|!!|عاجل|فوراً|فورا|بسرعة|الحين", re.I)

def clean_text(text: str) -> str:
    # Preserve Arabic letters and negation: aggressive cleanup changes meaning.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()

def language(text: str) -> str:
    return "ar" if re.search(r"[\u0600-\u06ff]", text) else "en"

def mask_personal_data(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    return re.sub(r"(?<!\w)(?:\+?966|00966|0)5\d{8}(?!\w)", "[PHONE]", text)

def canonical(text: str) -> str:
    text = clean_text(text).casefold()
    text = re.sub(r"\{\{.*?\}\}|\b[A-Z]-\d+\b|\bINV-\d+-\d+\b|\d+", " ID ", text, flags=re.I)
    return re.sub(r"\W+", " ", text).strip()

def group_id(text: str) -> str:
    return hashlib.sha256(canonical(text).encode()).hexdigest()[:16]

def priority_for(intent: str | None, text: str) -> str:
    # This is an explicitly documented store rule, not a trained priority model.
    level = LEVELS.index(BASE_PRIORITY.get(intent, "medium" if intent is None else "low"))
    return LEVELS[min(level + bool(URGENCY.search(text)), len(LEVELS) - 1)]

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

def load_kb() -> list[dict]:
    chunks = []
    for path in sorted((DATA / "kb").glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title, heading, body = lines[0].lstrip("# "), None, []
        for line in lines[1:] + ["## __end__"]:
            if line.startswith("## "):
                if heading and "\n".join(body).strip():
                    chunks.append({"id": f"{path.stem}#{slugify(heading)}", "doc": path.stem,
                        "title": title, "section": heading, "text": "\n".join(body).strip(),
                        "source": f"data/kb/{path.name}"})
                heading, body = line[3:].strip(), []
            elif heading:
                body.append(line)
    if not chunks:
        raise FileNotFoundError("No KB documents found. Run python -m src.prepare_data")
    return chunks

def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
