"""Self-contained HTML report: the thing you show in the interview."""
import html

from .drift import verdicts_by_metric
from .metrics import wilson_interval

PCT = {"classification_accuracy", "macro_f1", "silent_error_rate", "field_f1", "field_precision", "field_recall",
       "hallucination_rate", "format_validity_rate", "bad_case_rate", "flag_rate", "consistency", "judge_pass_rate",
       "red_team_asr_guarded", "red_team_asr_unguarded"}

EXPLAIN = {
    "classification_accuracy": "Share of documents given the right category.",
    "macro_f1": "Category score that treats every category equally (not dominated by common ones).",
    "silent_error_rate": "Wrong category AND a case was created anyway. The dangerous kind of error.",
    "field_f1": "Quality of extracted fields: balance of precision (no wrong values) and recall (no missed values).",
    "field_precision": "Of the values returned, how many were right.",
    "field_recall": "Of the values that should be returned, how many were found.",
    "hallucination_rate": "Values the model returned that the document does not support.",
    "format_validity_rate": "Values in the required format (ISO dates, money, ID patterns).",
    "bad_case_rate": "Cases created in the system with wrong content. Guardrails exist to keep this at zero.",
    "flag_rate": "Share sent to a human. Too low = risky; too high = the automation is not saving work.",
    "consistency": "Same input run 5 times: how often the answer is identical.",
    "judge_pass_rate": "Share of summaries the AI judge accepts (faithful, complete, safe).",
    "red_team_asr_guarded": "Attack success rate with guardrails ON.",
    "red_team_asr_unguarded": "Attack success rate with guardrails OFF (shows what the guardrails buy).",
}
CSS = """body{font:14px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1d2433}
h1{font-size:22px}h2{font-size:17px;margin-top:32px;border-bottom:1px solid #dde;padding-bottom:4px}
table{border-collapse:collapse;width:100%;margin:8px 0}th,td{padding:5px 8px;border-bottom:1px solid #e6e8ee;text-align:left;vertical-align:top}
th{background:#f4f6fa;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
.pass{background:#e6f6ea;color:#14602b}.fail{background:#fde8e8;color:#9b1c1c}.warn{background:#fff4d6;color:#7a5500}
.banner{padding:12px 16px;border-radius:8px;font-weight:600;margin:12px 0}.muted{color:#667}.mono{font-family:ui-monospace,monospace;font-size:12px}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px}"""


def _fmt(name, v):
    return f"{v * 100:.1f}%" if name in PCT else f"{v:.3f}"


def _tag(verdict):
    cls = "fail" if ("regress" in verdict or "floor" in verdict or verdict == "fail") else \
          "warn" if verdict == "warn" else "pass" if verdict in ("ok", "pass") else ""
    return f'<span class="tag {cls}">{html.escape(verdict)}</span>'


def render(result, findings, gate_result, baseline) -> str:
    snap = result["snapshot"]
    e = html.escape
    by_metric = verdicts_by_metric(findings)
    rows = ""
    for name, m in snap["metrics"].items():
        lo, hi = wilson_interval(round(m["value"] * m["n"]), m["n"]) if name in PCT else (None, None)
        ci = f"{lo * 100:.0f}–{hi * 100:.0f}%" if lo is not None else ""
        base = baseline["metrics"].get(name, {}).get("value") if baseline else None
        verdict = by_metric.get(name)
        rows += (f"<tr><td><b>{name}</b><br><span class='muted'>{e(EXPLAIN.get(name, ''))}</span></td>"
                 f"<td>{_fmt(name, m['value'])}</td><td class='muted'>{ci}</td><td class='muted'>{m['n']}</td>"
                 f"<td>{_fmt(name, base) if base is not None else '–'}</td><td>{_tag(verdict) if verdict else ''}</td></tr>")
    banner = ("pass", "RELEASE GATE: PASS") if gate_result["passed"] else ("fail", "RELEASE GATE: BLOCKED")
    blocking = f" — blocking: {', '.join(gate_result['blocking'])}" if gate_result["blocking"] else ""
    warn = f" · warnings: {', '.join(gate_result['warnings'])}" if gate_result["warnings"] else ""

    cls = result["classification"]
    conf_labels = list(cls["confusion"])
    conf = "<tr><th>true \\ predicted</th>" + "".join(f"<th>{c}</th>" for c in conf_labels) + "</tr>"
    for t in conf_labels:
        conf += f"<tr><td><b>{t}</b></td>" + "".join(f"<td>{cls['confusion'][t].get(p, 0) or ''}</td>" for p in conf_labels) + "</tr>"

    bad = [r for r in result["records"] if r["predicted"] != r["expected"] or r["fields_ok"] is False or r["status"] != "case_created"]
    fail_rows = "".join(
        f"<tr><td class='mono'>{e(r['id'])}</td><td>{e(r['tag'])}</td><td>{r['expected']} → <b>{r['predicted']}</b></td>"
        f"<td>{e(r['status'])}</td><td>{e(', '.join(r['reasons']))}</td><td class='muted'>{e(r['note'])}</td></tr>" for r in bad)

    def rt(rows_):
        return "".join(f"<tr><td class='mono'>{r['id']}</td><td>{r['kind']}</td><td>{_tag('fail' if r['succeeded'] else 'pass')}</td>"
                       f"<td>{e(', '.join(r['goals_reached']))}</td></tr>" for r in rows_)
    on, off = result["redteam"]["guarded"], result["redteam"]["unguarded"]
    why = "".join(f"<tr><td class='mono'>{r['id']}</td><td>{e(r['why'])}</td><td>{_tag('fail' if r['succeeded'] else 'pass')}</td>"
                  f"<td>{_tag('fail' if o['succeeded'] else 'pass')}</td></tr>" for r, o in zip(on["rows"], off["rows"]))

    cal = result["calibration"]
    dis = "".join(f"<tr><td class='mono'>{d['id']}</td><td>{d['human']}</td><td>{d['judge']}</td><td>{e(d['defect'])}</td></tr>"
                  for d in cal["disagreements"]) or "<tr><td colspan=4>none</td></tr>"

    return f"""<!doctype html><html><head><meta charset="utf-8"><title>AI evaluation report</title><style>{CSS}</style></head><body>
<h1>AI evaluation report — document-intake agent</h1>
<p class="muted">agent <b>{snap.get('agent', 'pipeline')}</b> · model profile <b>{snap['model_profile']}</b> · LLM <b>{snap['llm']}</b> · judge <b>{snap['judge']}</b> · {snap['dataset_size']} golden documents · {snap['created']}</p>
<div class="banner {banner[0]}">{banner[1]}{e(blocking)}{e(warn)}</div>

<h2>1. Headline metrics vs baseline</h2>
<table><tr><th>Metric</th><th>Now</th><th>95% CI</th><th>n</th><th>Baseline</th><th>Drift verdict</th></tr>{rows}</table>
<p class="muted">Verdicts: <b>ok</b> within tolerance · <b>warn</b> worse than tolerance but could be noise · <b>regress</b> worse AND statistically significant (blocks release) · <b>floor_breach</b> below the absolute business minimum (blocks release).</p>

<h2>2. Classification</h2><table>{conf}</table>

<h2>3. Documents that were not clean pass-throughs</h2>
<p class="muted">Wrong category, wrong or missing fields, or routed to a human. Known weak spots are listed, not hidden.</p>
<table><tr><th>Doc</th><th>Kind</th><th>Category</th><th>Outcome</th><th>Reason</th><th>Note</th></tr>{fail_rows}</table>

<h2>4. Red team: guardrails ON vs OFF</h2>
<p><b>Attack success rate: {on['asr'] * 100:.0f}% with guardrails, {off['asr'] * 100:.0f}% without.</b> Lower is better.</p>
<table><tr><th>Attack</th><th>What it tries</th><th>Guardrails ON</th><th>Guardrails OFF</th></tr>{why}</table>

<h2>5. AI-judge calibration</h2>
<p>Judge <b>{cal['judge']}</b> vs human labels on {cal['n']} summaries: agreement <b>{cal['agreement'] * 100:.0f}%</b>, Cohen's kappa <b>{cal['kappa']:.2f}</b>
(trusted if ≥ {cal['kappa_floor']}): {_tag('pass' if cal['trusted'] else 'fail')}</p>
<table><tr><th>Case</th><th>Human</th><th>Judge</th><th>Defect the judge missed</th></tr>{dis}</table>
</body></html>"""
