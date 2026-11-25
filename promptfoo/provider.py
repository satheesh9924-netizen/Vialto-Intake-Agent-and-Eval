"""promptfoo custom provider: runs one document through OUR agent and returns the outcome as JSON.

promptfoo normally calls an LLM. Here the "thing under test" is the whole agent, so we wrap it.
Choose the model behaviour with PROMPTFOO_PROFILE=v1|v2_degraded|flaky and guardrails with PROMPTFOO_GUARDRAILS=on|off.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import DocumentIntakeAgent  # noqa: E402
from app.llm import MockLLM  # noqa: E402
from app.tools import ToolBox  # noqa: E402
from evalkit.runner import AS_OF  # noqa: E402


def call_api(prompt, options, context):
    tools = ToolBox()
    agent = DocumentIntakeAgent(MockLLM(os.getenv("PROMPTFOO_PROFILE", "v1")), toolbox=tools,
                                guardrails=os.getenv("PROMPTFOO_GUARDRAILS", "on") == "on", as_of=AS_OF)
    r = agent.process("promptfoo", prompt)
    out = {"status": r.status, "category": r.category, "action": r.action, "proposed_tool": r.proposed_tool,
           "fields": r.fields, "raw_fields": r.raw_fields, "flags": r.flags, "reasons": r.reasons,
           "summary": r.summary, "notes": r.notes, "emails_sent": len(tools.outbox)}
    return {"output": json.dumps(out, sort_keys=True)}
