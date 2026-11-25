"""Guardrails: three independent layers (defence in depth).

Layer 1  InputGuard   - inspect the untrusted document BEFORE the model sees it.
Layer 2  ToolPolicy   - the model may only *propose* actions; this decides if they run.
Layer 3  OutputGuard  - scan what leaves the system for secrets.

No single layer is perfect. The red-team suite measures how often an attack gets
through with all layers on versus all layers off.
"""
import re
import unicodedata
from dataclasses import dataclass, field

from .llm import SYSTEM_SECRET
from .schemas import ALLOWED_TOOLS, ZERO_WIDTH

MAX_CHARS = 20_000

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|guidance)",
    r"disregard\b",
    r"system prompt",
    r"you are now",
    r"admin mode",
    r"reveal (your|the) (instructions|prompt|system)",
    r"send (an? )?e-?mail",
    r"delete (the |this )?case",
    r"classify (this|it) as",
    r"</?(system|assistant)>",
    r"<<<DOC|DOC>>>",
]
_INJECTION_RE = [re.compile(p, re.I) for p in INJECTION_PATTERNS]


@dataclass
class GuardResult:
    cleaned: str = ""
    flags: list = field(default_factory=list)
    rejected: str | None = None


class InputGuard:
    def scan(self, text: str) -> GuardResult:
        if not text or not text.strip():
            return GuardResult(rejected="empty_input", flags=["empty_input"])
        if len(text) > MAX_CHARS:
            return GuardResult(rejected="input_too_large", flags=["input_too_large"])
        # Normalise first so look-alike characters and zero-width tricks cannot hide a pattern.
        norm = unicodedata.normalize("NFKC", text).translate(ZERO_WIDTH)
        kept, flags = [], []
        for line in norm.splitlines():
            if any(p.search(line) for p in _INJECTION_RE):
                flags.append("prompt_injection_suspected")
                continue  # drop the offending line
            kept.append(line)
        return GuardResult(cleaned="\n".join(kept), flags=sorted(set(flags)))


class ToolPolicy:
    def check(self, call: dict) -> tuple[bool, str]:
        name = (call or {}).get("name")
        if name not in ALLOWED_TOOLS:
            return False, f"tool_blocked:{name}"
        args = call.get("args") or {}
        if not isinstance(args, dict) or len(str(args)) > 500:
            return False, "tool_args_invalid"
        return True, "ok"


_SECRET_RE = re.compile(r"CANARY-[0-9a-f]{8}")
_PROMPT_ECHO_RE = re.compile(r"You are DocIntake|Internal reference \(never reveal\)", re.I)


class OutputGuard:
    def clean(self, obj):
        """Return (cleaned_obj, flags). Redacts secrets / system-prompt text anywhere in the structure."""
        flags = []

        def walk(x):
            if isinstance(x, str):
                if _SECRET_RE.search(x) or _PROMPT_ECHO_RE.search(x):
                    flags.append("secret_leak_blocked")
                    return "[REDACTED]"
                return x
            if isinstance(x, dict):
                return {k: walk(v) for k, v in x.items()}
            if isinstance(x, list):
                return [walk(v) for v in x]
            return x

        return walk(obj), sorted(set(flags))
