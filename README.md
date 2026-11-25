# vlabs-ai-eval-kit

A test and evaluation framework for an AI agent, built around a small **document-intake agent** for immigration and tax paperwork. The agent classifies a document (passport, visa notice, employment letter, payslip, tax return), extracts fields, and acts through tools (create a case, or flag for human review).

The agent is the thing under test. The framework is the product: it tells you whether the agent is correct, safe, stable and still as good as it was last week.

```
make install    # python + node dependencies (also needs the Tesseract program for OCR)
make test       # 268 tests (+5 documented expected failures), includes ~1 min of real OCR
make test-fast  # same without the slow OCR tests
make demo       # healthy model passes the release gate; degraded and flaky models are blocked
make langgraph  # the same evaluation on the LangGraph version of the agent
make newman     # Postman collection against the live API
make promptfoo  # promptfoo evals (and `make promptfoo-degraded` to see them fail)
make ocr        # scan-quality experiment with real Tesseract OCR
```

Open `results/v1/report.html` (healthy) and `results/v2_degraded/report.html` (regression caught).

## What is in here

| Folder | What it is |
|---|---|
| `app/` | The agent under test: `agent.py` (steps/skills), `graph_agent.py` (**LangGraph** version of the same agent), `guardrails.py` (3 layers), `tools.py` (SQLite-backed actions), `llm.py` (model clients), `adapters.py` (LangChain, LlamaIndex, Bedrock, Azure), `grounding.py`, `audit.py`, `api.py` (REST) |
| `evalkit/` | The framework: `metrics.py`, `judge.py` (AI-as-judge + calibration), `drift.py` (baseline, significance, release gate), `redteam.py`, `runner.py`, `report.py`, `deepeval_suite.py` (DeepEval metrics), `ocr_eval.py` (real OCR experiment) |
| `data/golden/` | 33 ground-truth documents with expected category and fields, tagged clean / date_variant / missing_field / noisy_ocr / ambiguous / edge |
| `data/adversarial/` | 12 attacks: prompt injection, data leakage, tool abuse, unsafe input, oversized input |
| `data/judge/` | 20 human-labelled summaries used to calibrate the judge |
| `data/baseline/` | Frozen baseline metrics used for drift detection |
| `tests/` | pytest suite |
| `postman/` | Postman collection for the API, run with Newman (8 requests, 18 assertions, 0 failed) |
| `promptfoo/` | promptfoo provider, custom assertions and generated config (32 tests) |
| `tools/` | Node tools: Newman and promptfoo (`npm install --prefix tools`) |
| `docs/` | `agent_graph.md`: the LangGraph diagram |
| `CODE_EXPLAINED.md` | **Start here to explain it:** every file, in plain words, plus job-description coverage |
| `.github/workflows/eval.yml` | CI: tests and release gate on every PR, nightly drift re-evaluation |

## How the agent works

`input guard → classify → extract (model also proposes a tool call) → validate → act → summarise → output guard → audit`

The model **proposes**; the code **decides**. A proposed tool call only runs if the tool is on an allowlist and the validated data supports it.

Agent skills: classify, extract, validate (grounding, format, business rules such as passport expiry), act, summarise.

## How correctness is decided

Ground truth, not opinion. Every golden document has the expected category and exact expected fields.
- **Classification**: accuracy, per-class recall, macro F1, confusion matrix, and a "silent error" rate (wrong category that still created a case).
- **Extraction**: field-level precision / recall / F1.
- **Hallucination**: every returned value must be *grounded* in the source text (`grounding.py`). A value the document does not support counts as invented. A missing field must come back `null`, never guessed.
- **Format validity**: ISO dates, money and ID patterns.
- **Consistency**: same input run 5 times, measure how often the answer is identical.
- **Free text (summaries)**: graded by an AI judge against a rubric (faithfulness, completeness, safety), after the judge itself is calibrated against human labels (Cohen's kappa ≥ 0.6).

## Guardrails (defence in depth)

1. **Input guard**: normalises text (look-alike and zero-width characters), drops lines that look like instructions, strips delimiter markers, rejects empty or oversized input.
2. **Tool policy**: allowlist of tools; anything else the model asks for (`send_email`, `delete_case`) is blocked.
3. **Output guard**: redacts secrets or system-prompt text anywhere in the response.
4. **Validation**: ungrounded values removed, format checked, low confidence or missing fields or expiring passport go to human review.

The red-team suite runs every attack with guardrails ON and OFF. Result: **92% attack success without guardrails, 8% with**. The remaining 8% is `inj-10`, a documented residual risk (see below).

## Drift detection and release gating

Baseline metrics are frozen in `data/baseline/baseline.json`. Each run is compared to it. A metric is flagged `regress` only if it is worse than tolerance **and** the change is statistically significant (two-proportion test), so noise does not block releases. Absolute floors (for example hallucination ≤ 2%) block regardless of baseline. `python run_evals.py --gate` exits 1 when the release should be blocked. CI runs it on every PR and nightly.

## Failure and edge scenarios covered

Model timeout (with retry), garbage output, invalid category, database down (falls back to the review queue), crash mid-write (transaction rolls back, no partial data), both tools down (returns an error, no exception), duplicate submissions (idempotent), empty and oversized input, CRLF and whitespace noise, OCR-damaged labels, ambiguous documents, passport expiring soon.

## Run it with a real model

```
pip install anthropic && export ANTHROPIC_API_KEY=...
python run_evals.py --llm claude --judge claude --out results/claude
```

## Honest limits

- **The default "model" is a deterministic stand-in** (`MockLLM`). It makes tests repeatable and offline, and it is deliberately imperfect and injectable so the framework has something real to catch. Numbers from the mock say the *framework works*, not how well any real model performs. Run `--llm claude` for that. The real-model path is written but has not been run here.
- **Golden data is synthetic and small** (33 documents). It shows the method; a production set is built with subject-matter experts and grows with real failures.
- **No real OCR or vision model.** Documents are text, as if OCR had already run. OCR noise is simulated by corrupting labels. Testing an actual OCR/VLM pipeline is a next step.
- **The heuristic judge has a known blind spot** (unsupported inferences that reuse words from the document). The calibration set shows it, and an LLM judge is the remedy.
- **Known residual risk `inj-10`**: a plausible-sounding "this document is a passport" instruction slips past every layer. It is kept as a strict expected-failure test, so if it ever starts passing, or other attacks start failing, the suite says so. Real mitigation is model-side (classifier ensemble, separate verification step).
- **Tools from the job description, and their real status:**
  - Run here: DeepEval (deterministic and custom metrics), RAGAS (non-LLM metrics only), promptfoo, Newman, LangChain, LangGraph, LlamaIndex, Tesseract OCR, httpx, SQL.
  - Not run: DeepEval/RAGAS metrics that need a judge LLM (Claude wrapper written), and the real Claude model.
  - Stub-tested only: AWS Bedrock and Azure AI Foundry adapters. Request shape is verified, but they never called a real account.
  - Not built: vision-language-model (VLM) testing.
- **OCR finding:** real OCR swaps look-alike characters (`O`/`0`, `Z`/`2`, `B`/`8`). Closed-list validators fixed visa types and currency codes (wrong data stored on clean scans: 3% to 0%). Look-alike swaps inside free-form IDs remain (up to 12% on rotated scans) and are recorded as strict expected failures; a check digit or OCR confidence score is the fix.
- **Free-text summaries are protected by the judge, not the guardrails:** the degraded model's invented sentence still reaches the summary text; the judge scores it 0% pass.


## Running on a real model (Gemini, free key)

```
export GEMINI_API_KEY=...            # free key from https://aistudio.google.com
make gemini-check                    # lists the models your key can use and makes one real call
export GEMINI_MODEL=<a name from that list>     # optional; default is gemini-flash-lite-latest
make eval-gemini                     # = python run_evals.py --llm gemini --judge gemini --quick
make test-fast                       # with the key set, the live Gemini tests run too (5 small calls, cached)
```

What changes with a real model:
* The numbers are real, so expect misses (wrong labels, invented values, ignored injections). That is the point of the evals.
* A real model gets its own baseline (`data/baseline/baseline-gemini.json`) and its own results folder (`results/gemini`). Freeze it with `python run_evals.py --llm gemini --judge gemini --update-baseline` once you accept a run as the reference.
* `GEMINI_RPM` (default 10) limits requests per minute so a free key is not exhausted. A full run is several hundred calls; `--quick` halves that.
* Replies are cached in `.cache/gemini`. If a run stops (rate limit, daily quota), run the same command again and it continues from where it stopped without repeating paid or free calls. `GEMINI_CACHE=off` disables the cache.
* All documents in this repo are synthetic. Free-tier prompts may be used by Google to improve its products, so never send real documents.
