#!/usr/bin/env python3
"""OCR / document-AI evaluation: how does scan quality affect the whole pipeline? Writes results/ocr/."""
import json
import pathlib

from app.llm import MockLLM
from evalkit import ocr_eval, runner

out = pathlib.Path(__file__).parent / "results/ocr"
out.mkdir(parents=True, exist_ok=True)
rows = ocr_eval.run_all(runner.load("data/golden/documents.jsonl"), lambda: MockLLM("v1"), runner.AS_OF)
(out / "ocr_results.json").write_text(json.dumps(rows, indent=2))
lines = ["| scan quality | OCR text similarity | category accuracy | field F1 | sent to human | WRONG data stored |", "|---|---|---|---|---|---|"]
lines += [f"| {r['quality']} | {r['ocr_similarity']:.3f} | {r['classification_accuracy']:.0%} | {r['field_f1']:.2f} | {r['flag_rate']:.0%} | {r['bad_case_rate']:.0%} |" for r in rows]
(out / "ocr_results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
