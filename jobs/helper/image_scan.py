"""Image-embedded prompt-injection: OCR the image(s), then scan the text.

A rising agentic threat is text hidden *inside an image* — "ignore previous
instructions and…" rendered as pixels, invisible to text scanners. This module
extracts image media from a trace event and OCRs it to text; the caller then runs
that text through the normal injection / jailbreak / skeleton-key scanners
(see ``scanners.scan(image_texts=…)``), so all the mature text detection is
reused with zero duplication.

The OCR backend is **pluggable and fail-open**: a caller may inject an ``ocr_fn``
(used in tests / when a vision service is configured); otherwise it lazily tries
``pytesseract`` then ``easyocr`` (the worker already ships torch) and, if neither
is available, quietly extracts nothing. Media follows the same rule as the vision
judge — a stored ``_media_ref`` is only usable when it carries a URL; base64
payloads aren't stored in the trace.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

OcrFn = Callable[[bytes, Optional[str]], str]

_MAX_IMAGES = 6
_MAX_BYTES = 8 * 1024 * 1024  # don't fetch/decode absurdly large images


# ── media extraction (URL or inline data) ────────────────────────────────────

def _image_from_part(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ref = part.get("_media_ref")
    if isinstance(ref, dict):
        if ref.get("kind") != "image":
            return None
        if ref.get("data"):  # small base64 kept inline → usable
            return {"data": ref["data"], "mime": ref.get("mime")}
        if ref.get("source") == "url" and ref.get("url"):
            return {"url": ref["url"], "mime": ref.get("mime")}
        return None  # large base64 ref → only a hash, payload not stored
    iu = part.get("image_url")
    if isinstance(iu, dict) and iu.get("url"):
        return {"url": iu["url"]}
    if isinstance(iu, str) and iu:
        return {"url": iu}
    src = part.get("source")
    if isinstance(src, dict):
        if src.get("url"):
            return {"url": src["url"], "mime": src.get("media_type")}
        if src.get("data"):
            return {"data": src["data"], "mime": src.get("media_type")}
    idata = part.get("inline_data") or part.get("inlineData")
    if isinstance(idata, dict) and idata.get("data"):
        return {"data": idata["data"], "mime": idata.get("mime_type") or idata.get("mimeType")}
    return None


def extract_images(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect usable image media (url or inline data) from a trace event."""
    out: List[Dict[str, Any]] = []
    messages = event.get("messages") or event.get("contents") or event.get("input")
    seqs = messages if isinstance(messages, list) else []
    for msg in seqs:
        content = msg.get("content") if isinstance(msg, dict) else (msg if isinstance(msg, list) else None)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    item = _image_from_part(part)
                    if item:
                        out.append(item)
    return out[:_MAX_IMAGES]


# ── OCR backends (lazy, optional) ─────────────────────────────────────────────

_default_ocr: Optional[OcrFn] = None
_ocr_probed = False


def _load_default_ocr() -> Optional[OcrFn]:
    global _default_ocr, _ocr_probed
    if _ocr_probed:
        return _default_ocr
    _ocr_probed = True
    # pytesseract (fast, needs the tesseract binary)
    try:
        import io
        import pytesseract
        from PIL import Image

        def _tess(data: bytes, mime: Optional[str]) -> str:
            return pytesseract.image_to_string(Image.open(io.BytesIO(data))) or ""

        _default_ocr = _tess
        logger.info("[SECURITY] image OCR backend: pytesseract")
        return _default_ocr
    except Exception:
        pass
    # easyocr (torch-based — the worker already has torch)
    try:
        import io
        import easyocr
        import numpy as np
        from PIL import Image

        reader = easyocr.Reader(["en"], gpu=False)

        def _easy(data: bytes, mime: Optional[str]) -> str:
            img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
            return "\n".join(reader.readtext(img, detail=0) or [])

        _default_ocr = _easy
        logger.info("[SECURITY] image OCR backend: easyocr")
        return _default_ocr
    except Exception:
        logger.info("[SECURITY] no image OCR backend available; image-injection scan disabled")
        _default_ocr = None
        return None


def _image_bytes(item: Dict[str, Any]) -> Optional[bytes]:
    if item.get("data"):
        try:
            return base64.b64decode(item["data"])
        except Exception:
            return None
    url = item.get("url")
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        try:
            import requests
            resp = requests.get(url, timeout=8, stream=True)
            resp.raise_for_status()
            data = resp.raw.read(_MAX_BYTES + 1)
            return data if data and len(data) <= _MAX_BYTES else None
        except Exception:
            return None
    return None  # gs://, file://, etc. — not fetched here


def ocr_event_images(event: Dict[str, Any], ocr_fn: Optional[OcrFn] = None) -> List[Tuple[str, str]]:
    """Return ``[(source_label, extracted_text)]`` for each image with text.

    Fail-open: any fetch/decode/OCR error just skips that image. Returns [] when
    no OCR backend is available."""
    images = extract_images(event)
    if not images:
        return []
    ocr = ocr_fn or _load_default_ocr()
    if ocr is None:
        return []
    out: List[Tuple[str, str]] = []
    for i, item in enumerate(images):
        data = _image_bytes(item)
        if not data:
            continue
        try:
            text = (ocr(data, item.get("mime")) or "").strip()
        except Exception:
            logger.warning("[SECURITY] OCR failed for image[%d]", i, exc_info=True)
            continue
        if text:
            out.append((f"image[{i}]", text))
    return out
