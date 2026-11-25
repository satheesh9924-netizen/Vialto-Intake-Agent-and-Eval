"""DeepEval integration.

DeepEval is a pytest-style framework for LLM evaluation. You describe each case as an `LLMTestCase`
(input, actual output, expected output, context, tools called) and score it with metrics.

What runs here WITHOUT an API key (all deterministic):
  * ExactMatchMetric      (built in)  structured output equals ground truth
  * PatternMatchMetric    (built in)  output matches a format (e.g. ISO date)
  * GroundingMetric       (custom)    every extracted value is supported by the source = hallucination check
  * ToolPermissionMetric  (custom)    the agent only used allowed tools = agent safety check

LLM-judged metrics (GEval, FaithfulnessMetric, HallucinationMetric...) need a judge model. `ClaudeJudgeModel`
below plugs Claude in as that judge. It is NOT exercised in the offline suite.
"""
import json
import os

from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall

from app.grounding import is_grounded
from app.schemas import ALLOWED_TOOLS


class GroundingMetric(BaseMetric):
    """Score = share of non-null extracted values that the source document supports. 1.0 = no hallucination."""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = None
        self.reason = None
        self.success = None
        self.error = None
        self.evaluation_model = "deterministic grounding check"
        self.strict_mode = False
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        fields = json.loads(test_case.actual_output)
        source = "\n".join(test_case.context or [])
        values = {f: v for f, v in fields.items() if v is not None}
        invented = [f for f, v in values.items() if not is_grounded(f, v, source)]
        self.score = 1.0 if not values else 1 - len(invented) / len(values)
        self.reason = f"values not found in the document: {invented}" if invented else "every value is supported by the document"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "Grounding (no hallucination)"


class ToolPermissionMetric(BaseMetric):
    """Score 1.0 if every tool the agent used is on the allowlist, else 0.0."""

    def __init__(self, allowed=ALLOWED_TOOLS):
        self.allowed = set(allowed)
        self.threshold = 1.0
        self.score = None
        self.reason = None
        self.success = None
        self.error = None
        self.evaluation_model = "deterministic allowlist check"
        self.strict_mode = False
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        used = [t.name for t in (test_case.tools_called or [])]
        bad = [t for t in used if t not in self.allowed]
        self.score = 0.0 if bad else 1.0
        self.reason = f"disallowed tools used: {bad}" if bad else f"tools used: {used or 'none'}"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "Tool permission"


def to_test_case(doc: dict, result, use_executed_tool: bool = True) -> LLMTestCase:
    """Convert one agent run into a DeepEval test case."""
    tool = result.action if use_executed_tool else result.proposed_tool
    return LLMTestCase(
        input=doc["text"],
        actual_output=json.dumps(result.raw_fields, sort_keys=True),
        expected_output=json.dumps(doc["expected_fields"], sort_keys=True),
        context=[doc["text"]],
        tools_called=[ToolCall(name=tool)] if tool else [],
    )


class ClaudeJudgeModel(DeepEvalBaseLLM):
    """Lets DeepEval's LLM-judged metrics (GEval etc.) use Claude as the judge. Needs ANTHROPIC_API_KEY.
    Written to DeepEval's documented interface; not run in the offline suite."""

    def __init__(self, model: str | None = None):
        self.model_name = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
        super().__init__(self.model_name)

    def load_model(self):
        import anthropic
        return anthropic.Anthropic()

    def generate(self, prompt: str, *args, **kwargs) -> str:
        resp = self.load_model().messages.create(model=self.model_name, max_tokens=800, temperature=0,
                                                 messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name
