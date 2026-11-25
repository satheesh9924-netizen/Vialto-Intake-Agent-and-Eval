"""DocumentIntakeAgent: classify -> extract -> validate -> act, with guardrails and failure handling.

Agent "skills" (capabilities) are the four steps below. The model proposes; the code decides.
"""
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date

from .audit import AuditLog
from .grounding import is_grounded
from .guardrails import InputGuard, OutputGuard, ToolPolicy
from .llm import SYSTEM_PROMPT, LLMTimeout
from .schemas import CATEGORIES, REQUIRED_FIELDS, days_until, is_valid_format
from .tools import ToolBox, ToolError

STEP_LABELS = {"input_guard": "Input guard", "classify": "Classify", "extract": "Extract", "validate": "Validate",
               "act": "Act", "finish": "Summarise + output guard"}
CONF_THRESHOLD = 0.70
PASSPORT_MIN_VALIDITY_DAYS = 180


class ModelUnavailable(Exception):
    pass


@dataclass
class Ctx:
    """Working state that travels between steps (the result being built is kept separately)."""
    text: str
    doc: str = ""                                    # the text the model will see (after the input guard)
    issues: list = field(default_factory=list)       # reasons to send the document to a human
    proposed: dict = field(default_factory=dict)     # the tool call the model proposed
    done: bool = False                               # True = stop early, skip the remaining steps


@dataclass
class AgentResult:
    doc_id: str
    run_id: str
    status: str = "error"          # case_created | case_exists | flagged | rejected | error | executed:<tool>
    category: str | None = None
    confidence: float | None = None
    action: str | None = None      # the tool that actually ran
    fields: dict = field(default_factory=dict)       # after guardrails
    raw_fields: dict = field(default_factory=dict)   # exactly what the model returned (for model-quality metrics)
    reasons: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    proposed_tool: str | None = None
    summary: str | None = None
    notes: str | None = None       # free text the model volunteered (never needed; a leak risk)
    guardrails: bool = True
    model: str = ""
    trace: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    return json.loads(text)


class DocumentIntakeAgent:
    def __init__(self, llm, toolbox: ToolBox | None = None, guardrails: bool = True,
                 as_of: date | None = None, retries: int = 1, backoff: float = 0.0,
                 audit: AuditLog | None = None):
        self.llm = llm
        self.tools = toolbox or ToolBox()
        self.guardrails = guardrails
        self.as_of = as_of or date.today()
        self.retries = retries
        self.backoff = backoff
        self.audit = audit or AuditLog()
        self.input_guard, self.tool_policy, self.output_guard = InputGuard(), ToolPolicy(), OutputGuard()

    # ------------------------------------------------------------------ model calls
    def _call(self, task: str, doc: str, category: str | None = None, fields: dict | None = None) -> dict:
        user = f"TASK: {task}\n"
        if category:
            user += f"CATEGORY: {category}\n"
        if task == "extract" and category in REQUIRED_FIELDS:   # tell the model the exact keys to use
            user += f"FIELD_NAMES: {json.dumps(REQUIRED_FIELDS[category])}\n"
        if fields is not None:
            user += f"FIELDS: {json.dumps(fields)}\n"
        user += f"<<<DOC\n{doc}\nDOC>>>"
        last = None
        for attempt in range(self.retries + 1):
            try:
                data = _extract_json(self.llm.generate(SYSTEM_PROMPT, user))
                if not isinstance(data, dict):
                    raise ValueError("not an object")
                if task == "classify" and data.get("category") not in CATEGORIES:
                    raise ValueError("invalid category")
                return data
            except (LLMTimeout, ValueError) as e:  # json.JSONDecodeError is a ValueError
                last = e
                if self.backoff:
                    time.sleep(self.backoff * (attempt + 1))
        raise ModelUnavailable(str(last))

    # ------------------------------------------------------------------ orchestration
    def process(self, doc_id: str, text: str) -> AgentResult:
        r = AgentResult(doc_id=doc_id, run_id=uuid.uuid4().hex[:8], guardrails=self.guardrails,
                        model=getattr(self.llm, "profile", type(self.llm).__name__))
        try:
            self._run(r, text)
        finally:
            self.audit.record(text, r)
        return r

    # the order of the steps. The plain pipeline walks this list; the LangGraph version draws it as a graph.
    STEPS = ["input_guard", "classify", "extract", "validate", "act", "finish"]

    def _run(self, r: AgentResult, text: str):
        ctx = Ctx(text=text, doc=text)
        for name in self.STEPS:
            getattr(self, f"step_{name}")(r, ctx)
            if ctx.done:
                break

    @staticmethod
    def _trace(r, name, **kw):
        r.trace.append({"step": name, **kw})

    # ------------------------------------------------------------------ the steps (skills)
    def step_input_guard(self, r, ctx):
        """Layer 1: inspect the untrusted document before any model sees it."""
        if not self.guardrails:
            return
        g = self.input_guard.scan(ctx.text)
        self._trace(r, "input_guard", flags=g.flags, rejected=g.rejected)
        r.flags += g.flags
        if g.rejected:
            r.status, r.reasons, ctx.done = "rejected", [g.rejected], True
            return
        ctx.doc = g.cleaned

    def step_classify(self, r, ctx):
        try:
            cls = self._call("classify", ctx.doc)
        except ModelUnavailable:
            return self._flag(r, ctx, "model_unavailable")
        r.category, r.confidence = cls["category"], float(cls.get("confidence", 0))
        self._trace(r, "classify", category=r.category, confidence=r.confidence)
        if self.guardrails and r.confidence < CONF_THRESHOLD:
            self._flag(r, ctx, "low_confidence")

    def step_extract(self, r, ctx):
        """The model returns fields AND proposes which tool to call. It only proposes."""
        try:
            ext = self._call("extract", ctx.doc, category=r.category)
        except ModelUnavailable:
            return self._flag(r, ctx, "model_unavailable")
        raw = ext.get("fields") if isinstance(ext.get("fields"), dict) else {}
        r.raw_fields = {f: raw.get(f) for f in REQUIRED_FIELDS[r.category]}  # unexpected keys are ignored
        ctx.proposed = ext.get("tool_call") if isinstance(ext.get("tool_call"), dict) else {}
        r.proposed_tool = ctx.proposed.get("name")
        r.notes = ext.get("notes") if isinstance(ext.get("notes"), str) else None
        r.fields = dict(r.raw_fields)
        self._trace(r, "extract", proposed_tool=r.proposed_tool)

    def step_validate(self, r, ctx):
        """Layer 2 + checks: grounding, format, completeness, business rules, tool allowlist."""
        if not self.guardrails:
            return
        issues = ctx.issues
        for f, v in list(r.fields.items()):
            if v is not None and not is_grounded(f, v, ctx.doc):
                r.fields[f] = None          # drop invented values rather than trust them
                r.flags.append(f"ungrounded_field_removed:{f}")
                issues.append("hallucinated_field")
        bad = [f for f, v in r.fields.items() if not is_valid_format(f, v)]
        if bad:
            issues.append("invalid_format")
            r.flags += [f"invalid_format:{f}" for f in bad]
        if any(v is None for v in r.fields.values()) and r.category != "unknown":
            issues.append("missing_fields")
        if r.category == "unknown":
            issues.append("unrecognized_document")
        exp = r.fields.get("expiry_date") if r.category == "passport" else None
        if exp and is_valid_format("expiry_date", exp) and days_until(exp, self.as_of) < PASSPORT_MIN_VALIDITY_DAYS:
            issues.append("passport_expiring_soon")
        if "prompt_injection_suspected" in r.flags:
            issues.append("possible_prompt_injection")
        ok, why = self.tool_policy.check(ctx.proposed)
        if not ok:
            r.flags.append(why)
            issues.append("blocked_tool_request")
        self._trace(r, "validate", issues=sorted(set(issues)))

    def step_act(self, r, ctx):
        """The code decides. With guardrails: any issue -> human review. Without: do what the model said."""
        if self.guardrails:
            if ctx.issues:
                name, args = "flag_for_review", {"reason": ctx.issues[0]}
                r.reasons = sorted(set(ctx.issues))
            else:
                name, args = "create_case", {}
        else:
            name, args = ctx.proposed.get("name", "flag_for_review"), ctx.proposed.get("args") or {}
        self._act(r, ctx, name, args)

    def step_finish(self, r, ctx):
        """Summarise for humans, then Layer 3: scrub the output."""
        if r.category != "unknown" and r.status in ("case_created", "case_exists", "flagged"):
            try:
                r.summary = self._call("summarize", ctx.doc, category=r.category, fields=r.fields).get("summary")
            except ModelUnavailable:
                r.summary = None
        if self.guardrails:
            cleaned, flags = self.output_guard.clean(r.to_dict())
            if flags:
                r.flags += flags
                r.summary, r.notes = cleaned["summary"], cleaned["notes"]
                r.fields, r.raw_fields = cleaned["fields"], cleaned["raw_fields"]
                r.trace, r.reasons = cleaned["trace"], cleaned["reasons"]

    # ------------------------------------------------------------------ acting with failure handling
    def _act(self, r, ctx, name, args):
        tool_ctx = {"doc_id": r.doc_id, "category": r.category, "fields": r.fields}
        for attempt in range(self.retries + 1):
            try:
                out = self.tools.execute(name, args, tool_ctx)
                r.status, r.action = out["status"], name
                self._trace(r, "act", tool=name, status=r.status)
                return
            except ToolError as e:
                self._trace(r, "act_failed", tool=name, error=str(e), attempt=attempt)
        if name != "flag_for_review":      # retries exhausted: fall back to the safe action
            r.reasons.append("tool_failure")
            return self._flag(r, ctx, "tool_failure", stop=False)
        r.status = "error"
        r.reasons.append("review_queue_unavailable")

    def _flag(self, r, ctx, reason, stop=True):
        """Send to the human review queue and (by default) end the run."""
        r.reasons.append(reason)
        tool_ctx = {"doc_id": r.doc_id, "category": r.category or "unknown", "fields": r.fields}
        try:
            self.tools.execute("flag_for_review", {"reason": reason}, tool_ctx)
            r.status, r.action = "flagged", "flag_for_review"
        except ToolError:
            r.status = "error"
        self._trace(r, "flag", reason=reason, status=r.status)
        ctx.done = stop
