"""Drift detection and release gating.

Drift = the system's quality moves away from what it was when we last trusted it
(a model upgrade, a prompt edit, new kinds of documents...).

Method
  1. Freeze a BASELINE: metrics from a known-good run on the fixed golden dataset.
  2. Re-run the same dataset on a schedule and on every change, giving CURRENT.
  3. For each metric decide:
       ok            change is within tolerance
       warn          change exceeds tolerance but is NOT statistically significant (could be noise: re-run / add data)
       regress       change exceeds tolerance AND is statistically significant (two-proportion test, p < 0.05)
       floor_breach  absolute business minimum violated, regardless of baseline
  4. Release gate: any regress or floor_breach blocks the release.
"""
from .metrics import two_proportion_p

# metric -> (direction, tolerance). "higher" = higher is better.
TOLERANCES = {
    "classification_accuracy": ("higher", 0.05),
    "field_f1": ("higher", 0.05),
    "hallucination_rate": ("lower", 0.02),
    "format_validity_rate": ("higher", 0.05),
    "flag_rate": ("lower", 0.10),
    "consistency": ("higher", 0.03),
    "judge_pass_rate": ("higher", 0.10),
}

# Absolute business floors: (operator, limit)
FLOORS = {
    "classification_accuracy": (">=", 0.90),
    "field_f1": (">=", 0.90),
    "hallucination_rate": ("<=", 0.02),
    "format_validity_rate": (">=", 0.98),
    "silent_error_rate": ("<=", 0.03),
    "bad_case_rate": ("<=", 0.03),
    "consistency": (">=", 0.95),
    "judge_pass_rate": (">=", 0.90),
}
ALPHA = 0.05


def compare(baseline: dict, current: dict) -> list:
    """baseline/current are snapshots: {"metrics": {name: {"value": float, "n": int}}}."""
    findings = []
    for name, (direction, tol) in TOLERANCES.items():
        b, c = baseline["metrics"].get(name), current["metrics"].get(name)
        if not b or not c:
            continue
        delta = c["value"] - b["value"]
        worse = -delta if direction == "higher" else delta  # positive = got worse
        p = two_proportion_p(b["value"], b["n"], c["value"], c["n"])
        if worse <= tol:
            verdict = "ok"
        elif p < ALPHA:
            verdict = "regress"
        else:
            verdict = "warn"
        findings.append({"metric": name, "baseline": b["value"], "current": c["value"], "delta": delta,
                         "tolerance": tol, "p_value": p, "verdict": verdict})
    for name, (op, limit) in FLOORS.items():
        c = current["metrics"].get(name)
        if not c:
            continue
        breached = c["value"] < limit if op == ">=" else c["value"] > limit
        if breached:
            findings.append({"metric": name, "baseline": baseline["metrics"].get(name, {}).get("value"),
                             "current": c["value"], "delta": None, "tolerance": f"{op} {limit}",
                             "p_value": None, "verdict": "floor_breach"})
    return findings


def verdicts_by_metric(findings: list) -> dict:
    """metric -> 'regress + floor_breach' etc. (a metric can fail both the drift test and its floor)."""
    out = {}
    for f in findings:
        out.setdefault(f["metric"], [])
        if f["verdict"] not in out[f["metric"]]:
            out[f["metric"]].append(f["verdict"])
    return {m: " + ".join(v for v in vs if v != "ok" or len(vs) == 1) for m, vs in out.items()}


def gate(findings: list) -> dict:
    uniq = lambda xs: list(dict.fromkeys(xs))  # noqa: E731  (ordered de-duplication)
    blocking = uniq(f["metric"] for f in findings if f["verdict"] in ("regress", "floor_breach"))
    return {"passed": not blocking, "blocking": blocking,
            "warnings": [m for m in uniq(f["metric"] for f in findings if f["verdict"] == "warn") if m not in blocking]}
