#!/usr/bin/env python3
"""Run the full evaluation, write results, and (optionally) act as a release gate.

  python run_evals.py                          # healthy model, compare with baseline
  python run_evals.py --profile v2_degraded    # simulate a model regression
  python run_evals.py --update-baseline        # freeze the current run as the new baseline (deliberate action)
  python run_evals.py --gate                   # exit code 1 if the release should be blocked (for CI)
  python run_evals.py --llm claude --judge claude   # real Claude model + judge (needs ANTHROPIC_API_KEY)
  python run_evals.py --llm gemini --judge gemini   # real Gemini model + judge (needs GEMINI_API_KEY, free)
  python run_evals.py --llm gemini --quick          # about half the model calls
"""
import argparse
import json
import pathlib
import sys

from app.gemini import GeminiConfigError, GeminiQuotaExhausted
from evalkit import drift, judge, report, runner

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="v1", choices=["v1", "v2_degraded", "flaky"], help="mock model behaviour")
    ap.add_argument("--agent", default="pipeline", choices=["pipeline", "langgraph"], help="same agent, two wirings")
    ap.add_argument("--llm", default="mock", choices=["mock", "claude", "gemini"])
    ap.add_argument("--judge", default="heuristic", choices=["heuristic", "claude", "gemini"])
    ap.add_argument("--quick", action="store_true", help="fewer consistency runs (3 x 8 docs): about half the model calls")
    ap.add_argument("--out", default=None, help="output folder (default results/<profile>)")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--gate", action="store_true", help="exit 1 when the release gate blocks")
    args = ap.parse_args()

    if args.quick:
        runner.CONSISTENCY_RUNS, runner.CONSISTENCY_DOCS = 3, 8
    try:
        the_judge = {"claude": judge.ClaudeJudge, "gemini": judge.GeminiJudge, "heuristic": judge.HeuristicJudge}[args.judge]()
        result = runner.full_eval(args.profile, args.llm, the_judge, args.agent)
    except (GeminiConfigError, GeminiQuotaExhausted) as e:
        sys.exit(f"\nSTOPPED: {e}\nNothing was written. Re-running later resumes from the cache, so finished calls are not repeated.")

    base_path = runner.baseline_path(args.llm)
    if args.update_baseline:
        base_path.write_text(json.dumps(result["snapshot"], indent=2))
        print(f"baseline written -> {base_path.relative_to(ROOT)}")
    baseline = runner.load_baseline(args.llm)
    findings, gate = runner.evaluate_release(result, baseline)

    out_name = args.profile if args.llm == "mock" else args.llm  # a real model has no mock "profile"
    out = ROOT / (args.out or f"results/{out_name}" + ("" if args.agent == "pipeline" else f"-{args.agent}"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({**result, "findings": findings, "gate": gate}, indent=2, default=str))
    (out / "report.html").write_text(report.render(result, findings, gate, baseline))

    print(f"\nagent={args.agent} profile={args.profile} llm={args.llm} judge={the_judge.name} docs={result['snapshot']['dataset_size']}")
    by = drift.verdicts_by_metric(findings)
    for name, m in result["snapshot"]["metrics"].items():
        pct = name in report.PCT
        val = f"{m['value'] * 100:6.1f}%" if pct else f"{m['value']:7.3f}"
        print(f"  {name:26} {val}   n={m['n']:<4} {by.get(name, '')}")
    cal = result["calibration"]
    print(f"  judge calibration: agreement={cal['agreement']:.0%} kappa={cal['kappa']:.2f} trusted={cal['trusted']}")
    print(f"\nRELEASE GATE: {'PASS' if gate['passed'] else 'BLOCKED'}  {gate['blocking'] or ''}  warnings={gate['warnings'] or 'none'}")
    print(f"report: {out.relative_to(ROOT)}/report.html")
    if args.gate and not gate["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
