#!/usr/bin/env python3
"""Send a text file to the running API (start it first: make api  /  INTAKE_LLM=gemini make api).

  python scripts/send_doc.py samples/my_passport.txt
  python scripts/send_doc.py my_doc.txt --id my-doc-1
"""
import argparse
import json
import pathlib

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--id", default=None)
ap.add_argument("--url", default="http://127.0.0.1:8000")
a = ap.parse_args()
p = pathlib.Path(a.file)
r = httpx.post(f"{a.url}/intake", json={"doc_id": a.id or p.stem, "text": p.read_text()}, timeout=120)
print(r.status_code)
print(json.dumps(r.json(), indent=2))
