"""Document-AI / OCR testing: do scanned-quality documents still come out right end to end?

Pipeline under test:   document image  ->  OCR (Tesseract)  ->  agent  ->  case

For each golden document we render an image, damage it in controlled ways (what real scans look like),
OCR it, then run the agent on the OCR text and compare with ground truth. Two things are measured:
  * OCR quality   text similarity between OCR output and the true text (0..1)
  * end to end    category accuracy, field F1, share sent to a human, and BAD CASES (wrong data stored)

Important: the grounding guardrail checks values against the OCR text. If OCR itself misreads a digit the
value is "grounded" in wrong text. That is a real limit of text-only checks; the `bad_case_rate` column shows it.
"""
import random

import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from rapidfuzz.distance import Levenshtein

from app.agent import DocumentIntakeAgent
from app.tools import ToolBox

from . import metrics

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

QUALITIES = {
    "clean":        dict(),
    "low_contrast": dict(contrast=0.35),
    "blur":         dict(blur=1.6),
    "noise":        dict(noise=40),
    "rotated":      dict(rotate=4.0),
    "low_res":      dict(scale=0.45),
    "bad_scan":     dict(blur=1.2, noise=30, rotate=2.5, contrast=0.55, scale=0.6),
}


def render(text: str, size: int = 22) -> Image.Image:
    font = ImageFont.truetype(FONT, size)
    lines = text.splitlines()
    w = max(int(font.getlength(l)) for l in lines) + 60
    h = int(size * 1.5) * len(lines) + 60
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((30, 30 + i * int(size * 1.5)), line, font=font, fill=0)
    return img


def degrade(img: Image.Image, seed: int, contrast=None, blur=None, noise=None, rotate=None, scale=None) -> Image.Image:
    rng = np.random.default_rng(seed)
    if scale:
        w, h = img.size
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale)))).resize((w, h))
    if rotate:
        img = img.rotate(rotate, expand=True, fillcolor=255)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if contrast:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if noise:
        arr = np.asarray(img, dtype=np.float32) + rng.normal(0, noise, np.asarray(img).shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def ocr(img: Image.Image) -> str:
    return pytesseract.image_to_string(img, config="--psm 6")


def similarity(a: str, b: str) -> float:
    return Levenshtein.normalized_similarity(" ".join(a.split()), " ".join(b.split()))


def evaluate_quality(golden: list, llm_factory, quality: str, as_of) -> dict:
    opts = QUALITIES[quality]
    agent = DocumentIntakeAgent(llm_factory(), toolbox=ToolBox(), as_of=as_of)
    sims, records = [], []
    for i, d in enumerate(golden):
        text = ocr(degrade(render(d["text"]), seed=i, **opts))
        sims.append(similarity(text, d["text"]))
        records.append((d, agent.process(d["id"], text)))
    n = len(records)
    cls_ok = sum(r.category == d["expected_category"] for d, r in records)
    ok_docs = [(d, r) for d, r in records if r.category == d["expected_category"]]
    fld = metrics.field_prf([(d["expected_fields"], r.raw_fields) for d, r in ok_docs])
    bad = sum(r.status == "case_created" and (r.category != d["expected_category"] or
              any(r.fields.get(f) != v for f, v in d["expected_fields"].items())) for d, r in records)
    return {"quality": quality, "ocr_similarity": sum(sims) / n, "classification_accuracy": cls_ok / n,
            "field_f1": fld["f1"], "flag_rate": sum(r.status == "flagged" for _, r in records) / n,
            "bad_case_rate": bad / n, "n": n}


def run_all(golden, llm_factory, as_of, qualities=None) -> list:
    return [evaluate_quality(golden, llm_factory, q, as_of) for q in (qualities or QUALITIES)]
