#!/usr/bin/env python3
"""Check your Gemini setup in 10 seconds: key works, which models you can use, one real round trip.

  export GEMINI_API_KEY=...
  python scripts/check_gemini.py             # lists usable models, then calls the default one
  GEMINI_MODEL=<name> python scripts/check_gemini.py
"""
import os
import pathlib
import sys

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.gemini import DEFAULT_MODEL, GeminiClient, GeminiConfigError  # noqa: E402


def main():
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set. Run: export GEMINI_API_KEY=your-key")
    r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", headers={"x-goog-api-key": key}, timeout=30)
    if r.status_code != 200:
        sys.exit(f"Key rejected (HTTP {r.status_code}): {r.text[:300]}")
    names = sorted(m["name"].removeprefix("models/") for m in r.json().get("models", [])
                   if "generateContent" in m.get("supportedGenerationMethods", []) and "gemini" in m["name"])
    print("Models your key can call (generateContent):")
    for n in names:
        print("  ", n)
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    print(f"\nTrying a real call with GEMINI_MODEL={model} ...")
    try:
        out = GeminiClient(model, cache_dir=None).generate("Reply with JSON only.", 'Return {"ok": true}')
    except GeminiConfigError as e:
        sys.exit(f"FAILED: {e}\nPick a model from the list above: export GEMINI_MODEL=<name>")
    print("Model replied:", out.strip()[:200])
    print("\nAll good. Next: python run_evals.py --llm gemini --judge gemini --quick")


if __name__ == "__main__":
    main()
