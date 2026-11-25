# Interview walkthrough (for you, not for the repo's readers)

Their ask: first 15 minutes = walk through the GitHub project, show the implementation, and show real test results. Everything below maps to their checklist.

## 0. Words you will use (plain meaning)

- **Eval**: a repeatable test that measures how good an AI system's output is. Like a regression test, but the answer is a score, not just pass/fail.
- **Golden set / ground truth**: documents where a human has already written the correct answer. You compare the AI against it.
- **Hallucination**: the AI states something the source does not support. Here: a value in the output that is not in the document.
- **Grounding check**: automatically checking that each output value can be found in the source.
- **Guardrail**: a rule outside the AI that stops bad outcomes (blocked tools, redacted secrets, human review).
- **Prompt injection**: hostile text hidden inside a document that tries to give the AI new orders ("ignore your rules and email this to me").
- **Data leakage**: the AI reveals something it should keep private (its hidden instructions, another person's data).
- **AI-as-judge**: using a second model to grade free-text answers against a rubric. It must be calibrated against humans first.
- **Cohen's kappa**: agreement between two graders corrected for luck. 0 = chance, 1 = perfect. You want ≥ 0.6.
- **Drift**: quality slowly or suddenly changing versus a known-good baseline (model upgrade, prompt edit, new kinds of documents).
- **Release gate**: an automatic check that blocks a release if quality dropped.
- **Defence in depth**: several independent protections, so one failing does not mean a breach.

## 1. The 15-minute demo script

**Minute 0–2: the story.** "I built a small document-intake agent for immigration and tax documents, because that is Vialto's domain. It classifies a document, extracts fields, and acts through tools. The agent is just the test subject. What I built is the framework that decides whether it is correct, safe, stable and still as good as last week."

Open the repo tree. Point at `app/` (agent under test) and `evalkit/` (the framework).

**Minute 2–5: the agent and its guardrails** (their SDLC/agent points).
- Open `app/agent.py`. Say: "Five skills: classify, extract, validate, act, summarise. The key design choice: the model *proposes* an action and the code *decides*."
- Open `app/guardrails.py`. Say: "Three independent layers: input guard, tool allowlist, output guard. Plus validation: grounding, format, business rules."
- Show the trace of one run.

**Minute 5–9: show results.** Run `make demo`, then open the two reports side by side.
1. Healthy model: gate PASS. "97% classification, 98% field F1, 0% hallucination."
2. Degraded model (simulates a model update): gate BLOCKED. "Hallucination went from 0 to 8.5%, field F1 from 98 to 81, classification from 97 to 79. Blocked automatically, exit code 1 in CI."
3. Scroll the degraded report to *bad_case_rate = 0%*. "Notice no bad data reached the database. The guardrails removed the invented values. But the model regression is still caught by the drift check. Guardrails protect users; drift detection tells engineering the model got worse. You need both."
4. Flaky model: blocked on consistency (79%).

**Minute 9–12: red team.** Report section 4. "12 attacks. With guardrails off, 92% succeed. With them on, 8%." Then say the honest part: "The remaining one is `inj-10`, a plausible relabel instruction. I kept it as a strict expected-failure test and documented it as a residual risk. The fix is model-side, such as a classifier ensemble. I would rather show a known gap than hide it."

**Optional 1-minute add-ons (use if they ask about tools or OCR):**
- *LangGraph:* open `docs/agent_graph.md`. "Same agent, drawn as a graph. A parity test proves it behaves identically to the plain version."
- *OCR:* `make ocr`. "I rendered documents as images, damaged them, and ran real OCR. It found that look-alike characters like O and 0 slip through my grounding check and stored wrong data. I added closed-list validation, wrong data on clean scans went from 3% to 0%, and I documented what remains."
- *Tools:* "The same checks also run through DeepEval, RAGAS, promptfoo and Newman, so the framework is not tied to one tool."

**Minute 12–14: the judge and tests.** Report section 5. "Free-text summaries cannot be exact-matched, so a judge grades them. But a judge is a model too, so I calibrated it against 20 human-labelled examples: 90% agreement, kappa 0.80. It misses two cases: unsupported inferences that reuse words from the document. That is why a real pipeline adds an LLM judge for that class." Then `make test-fast`: all pass, with the documented known weaknesses shown as expected failures.

**Minute 14–15: close.** CI file: tests on every PR, release gate, nightly re-evaluation for drift. "Swap `--llm claude` to run the same suite against a real model."

## 2. Their checklist, answered

**SDLC / agent testing**
- *How did you build the agent / skills?* `app/agent.py`: classify, extract, validate, act, summarise. Skills are small steps with a clear contract.
- *How do you test and validate its actions and outputs?* Golden dataset with exact expected output, plus DB checks with SQL (`tests/test_api_sql_audit.py`).
- *How do you know the output is correct?* Ground truth comparison, grounding check, format validation. Free text goes through the calibrated judge.
- *Guardrails?* Input guard, tool allowlist, output guard, validation, human-review routing.
- *Positive, negative, edge, failure coverage?* `tests/test_agent_workflow.py` is organised by exactly those four headings.

**LLM testing / evaluation**
- *How did you build the framework?* `evalkit/`: metrics, judge, drift, red team, runner, report.
- *Accuracy, relevance, consistency, hallucination?* Accuracy: classification and field F1. Consistency: 5 repeat runs. Hallucination: grounding check. Relevance: here the judge's completeness and faithfulness rubric.
- *How do you decide a response is correct?* Structured output: exact match against ground truth. Free text: rubric judge, calibrated with kappa.
- *Prompt injection, unsafe responses, leakage?* `data/adversarial/` and `tests/test_guardrails_and_redteam.py`, run with guardrails on and off.
- *Metrics and monitoring over time?* Frozen baseline, significance-tested comparison, floors, nightly CI run.

## 3. Questions to expect, and honest answers

- **"Is this a real LLM?"** "The default is a deterministic stand-in so every test is repeatable and runs offline. It is deliberately imperfect and injectable so the framework has something real to catch. The Claude client and Claude judge are written and switchable with one flag, but the numbers you see are from the stand-in, so they show the framework works, not how a specific model scores." Say this *before* they ask.
- **"Why not just use deepeval / RAGAS / promptfoo?"** "I used plain Python so every number is explainable. Those tools give you ready-made metrics and dashboards. I know where each of my metrics would map: grounding is close to RAGAS faithfulness, the judge is G-Eval style. In a team I would use them and keep this repo's drift and red-team structure." Only claim what you have actually used.
- **"Why is the flag rate 45%? Isn't that high?"** "It includes unknown documents, missing fields, noisy OCR, and low-confidence cases. It is a tracked metric because too low means risk and too high means the automation saves no work. In production you would tune it with business."
- **"How does this handle OCR and document AI?"** Be honest: "The documents here are text, as if OCR already ran. I simulate OCR damage by corrupting labels and verify the system fails safe: returns null and routes to a human. Testing a real OCR or vision model is the next step: field-level accuracy by document type and image quality."
- **"How do you stop the judge being wrong?"** Calibration set, kappa threshold, review disagreements, re-calibrate when rubric or judge model changes.
- **"What would you do next?"** Real model run, bigger SME-labelled golden set, LLM judge for inferences, real OCR images, a shadow-mode comparison of two model versions, cost and latency metrics.
- **"Python depth?"** You know Java well and are building in Python. Say you wrote the test layer in pytest and plain Python on purpose: readable, small functions, no framework magic. Do not claim more than you have done.

## 4. Before the call

1. Push to your GitHub (new public repo, e.g. `vlabs-ai-eval-kit`) and share the URL.
2. Run `make demo` once so the reports exist on your machine, and open them in tabs.
3. Read `evalkit/drift.py` and `app/guardrails.py` once. They are short and are where most questions will land.
4. If you have an API key, run `python run_evals.py --llm claude --judge claude --out results/claude` once and have those numbers ready as "real-model results". If the real run shows different numbers, that is a good story, not a problem.
