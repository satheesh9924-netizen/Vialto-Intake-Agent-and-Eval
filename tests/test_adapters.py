"""The agent runs unchanged on LangChain, LlamaIndex, Bedrock and Azure model stacks.

LangChain / LlamaIndex: real framework classes wrapping the mock model, so the full golden set must give
identical results to calling the mock directly.
Bedrock / Azure: request shape checked offline (botocore Stubber validates against the real AWS service
model; Azure uses a mock HTTP transport). No real cloud call is made.
"""
import json
from typing import Any

import httpx
import pytest
from botocore.stub import Stubber
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from llama_index.core.llms import ChatMessage, ChatResponse, CompletionResponse, CustomLLM, LLMMetadata, MessageRole

from app.adapters import AzureFoundryLLM, BedrockLLM, LangChainLLM, LlamaIndexLLM
from app.agent import DocumentIntakeAgent
from app.llm import MockLLM
from app.tools import ToolBox
from conftest import AS_OF, GOLDEN


class MockChat(BaseChatModel):
    """A genuine LangChain chat model whose 'brain' is our mock."""
    inner: Any

    @property
    def _llm_type(self) -> str:
        return "mock-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system, user = messages[0].content, messages[-1].content
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.inner.generate(system, user)))])


class MockLlamaIndex(CustomLLM):
    """A genuine LlamaIndex LLM whose 'brain' is our mock."""
    inner: Any

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="mock-llamaindex")

    def chat(self, messages, **kwargs):
        system = next(m.content for m in messages if m.role == MessageRole.SYSTEM)
        user = next(m.content for m in messages if m.role == MessageRole.USER)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=self.inner.generate(system, user)))

    def complete(self, prompt, formatted=False, **kwargs):
        return CompletionResponse(text="")

    def stream_complete(self, prompt, formatted=False, **kwargs):
        yield CompletionResponse(text="")


def run_all(llm):
    agent = DocumentIntakeAgent(llm, toolbox=ToolBox(), as_of=AS_OF)
    out = []
    for d in GOLDEN:
        r = agent.process(d["id"], d["text"]).to_dict()
        r.pop("run_id"), r.pop("model")
        out.append(json.dumps(r, sort_keys=True))
    return out


@pytest.fixture(scope="module")
def reference():
    return run_all(MockLLM("v1"))


def test_langchain_stack_gives_identical_results(reference):
    assert run_all(LangChainLLM(MockChat(inner=MockLLM("v1")))) == reference


def test_llamaindex_stack_gives_identical_results(reference):
    assert run_all(LlamaIndexLLM(MockLlamaIndex(inner=MockLLM("v1")))) == reference


def test_langchain_adapter_flattens_content_blocks():
    class Blocks(MockChat):
        def _generate(self, messages, **kw):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=[{"type": "text", "text": '{"a":'}, {"type": "text", "text": "1}"}]))])
    assert LangChainLLM(Blocks(inner=None)).generate("s", "u") == '{"a":1}'


# ---- Bedrock: request validated by botocore against the real AWS service definition --------------------------
def test_bedrock_request_and_response_shape():
    import boto3
    client = boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="y")
    llm = BedrockLLM(model_id="anthropic.claude-test", client=client)
    with Stubber(client) as stub:
        stub.add_response(
            "converse",
            {"output": {"message": {"role": "assistant", "content": [{"text": '{"category": "passport", "confidence": 0.9}'}]}},
             "stopReason": "end_turn", "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
             "metrics": {"latencyMs": 1}},
            expected_params={"modelId": "anthropic.claude-test", "system": [{"text": "SYS"}],
                             "messages": [{"role": "user", "content": [{"text": "USER"}]}],
                             "inferenceConfig": {"maxTokens": 800, "temperature": 0}})
        assert json.loads(llm.generate("SYS", "USER"))["category"] == "passport"


# ---- Azure: mock HTTP transport ---------------------------------------------------------------------------------
def test_azure_request_and_response_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "gpt-test",
                                         "choices": [{"index": 0, "finish_reason": "stop",
                                                      "message": {"role": "assistant", "content": '{"category":"payslip","confidence":0.9}'}}]})

    llm = AzureFoundryLLM(deployment="my-deploy", endpoint="https://example.openai.azure.com", api_key="k",
                          http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert json.loads(llm.generate("SYS", "USER"))["category"] == "payslip"
    assert "/deployments/my-deploy/chat/completions" in seen["url"]
    assert seen["body"]["temperature"] == 0
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]
