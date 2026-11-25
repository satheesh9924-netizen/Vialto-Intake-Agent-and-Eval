#!/usr/bin/env python3
"""Watch the agent run: prints every function call in app/ in the exact order it executes.

  python scripts/trace_run.py samples/my_passport.txt                 # mock model (free, instant)
  python scripts/trace_run.py samples/my_passport.txt --llm gemini    # real Gemini (needs GEMINI_API_KEY)
  python scripts/trace_run.py samples/my_passport_attack.txt          # see the guardrails react
  python scripts/trace_run.py samples/my_passport.txt --no-guardrails # same document, safety layers OFF
  python scripts/trace_run.py my_doc.txt --graph                      # the LangGraph version of the agent

Indentation = "who called whom". The number is the order of execution. file:line is where the function is defined.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKIP = {"<genexpr>", "<listcomp>", "<dictcomp>", "<lambda>", "<module>", "__init__", "_now", "to_dict", "walk", "_trace",
        "__post_init__", "__repr__", "__eq__"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--llm", default="mock", choices=["mock", "gemini"])
    ap.add_argument("--no-guardrails", action="store_true")
    ap.add_argument("--graph", action="store_true", help="use the LangGraph version of the agent")
    ap.add_argument("--all", action="store_true", help="also show tiny helper functions")
    args = ap.parse_args()

    from app.graph_agent import agent_class
    from app.tools import ToolBox

    text = pathlib.Path(args.file).read_text()
    if args.llm == "gemini":
        from app.gemini import GeminiLLM
        llm = GeminiLLM()
    else:
        from app.llm import MockLLM
        llm = MockLLM("v1")
    agent = agent_class("langgraph" if args.graph else "pipeline")(llm, toolbox=ToolBox(), guardrails=not args.no_guardrails)

    app_dir = str(ROOT / "app")
    state = {"n": 0, "depth": 0}
    lines = []

    def profiler(frame, event, arg):
        code = frame.f_code
        if not code.co_filename.startswith(app_dir):
            return
        name = code.co_name
        if event == "call":
            state["depth"] += 1
            if name in SKIP and not args.all:
                return
            state["n"] += 1
            f = pathlib.Path(code.co_filename).relative_to(ROOT)
            hint = ""
            if name == "_call":
                hint = f"   <- asks the model: TASK={frame.f_locals.get('task')}"
            elif name == "execute":
                hint = f"   <- tool: {frame.f_locals.get('name')}"
            lines.append(f"{state['n']:>3}  {'  ' * (state['depth'] - 1)}{f}:{code.co_firstlineno}  {name}(){hint}")
        elif event == "return":
            state["depth"] -= 1

    print(f"\nDocument: {args.file}   model: {getattr(llm, 'profile', 'mock')}   guardrails: {not args.no_guardrails}   "
          f"agent: {'langgraph' if args.graph else 'pipeline'}\n")
    sys.setprofile(profiler)
    try:
        result = agent.process("trace-1", text)
    finally:
        sys.setprofile(None)
    print("ORDER OF EXECUTION (number = order, indent = called by the line above it)\n")
    print("\n".join(lines))

    print("\nSTEP TRACE (what each step recorded):")
    for t in result.trace:
        print("  ", json.dumps(t))
    print("\nRESULT")
    print(json.dumps({k: v for k, v in result.to_dict().items() if k not in ("trace", "raw_fields")}, indent=2))
    print("\nSaved case:", agent.tools.get_case("trace-1"))
    print("Review queue rows:", agent.tools.count("reviews"), "  Emails sent (must be 0):", len(agent.tools.outbox))


if __name__ == "__main__":
    main()
