"""Adapters: plug any model stack into the agent.

The agent only needs `generate(system, user) -> str`. Each class here wraps a different stack to that shape,
so the SAME agent and the SAME evaluation suite run against any of them. This is what makes the framework
reusable across "features, models, and releases".

  LangChainLLM      any LangChain chat model (ChatAnthropic, ChatOpenAI, AzureChatOpenAI, ChatBedrock...)
  LlamaIndexLLM     any LlamaIndex LLM
  BedrockLLM        AWS Bedrock via the Converse API (boto3)
  AzureFoundryLLM   Azure AI Foundry / Azure OpenAI deployment (openai SDK)

Verification status: LangChain and LlamaIndex adapters are tested end to end against real framework classes
wrapping the mock model. Bedrock and Azure are tested only against stubs (botocore Stubber, mock HTTP
transport) which check the request shape; they have never called a real cloud account.
"""
import os


def _text(content) -> str:
    """Chat models return a string or a list of content blocks; flatten to text."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


class LangChainLLM:
    def __init__(self, chat_model):
        self.chat_model = chat_model
        self.profile = f"langchain:{type(chat_model).__name__}"

    def generate(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        return _text(self.chat_model.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content)


class LlamaIndexLLM:
    def __init__(self, llm):
        self.llm = llm
        self.profile = f"llamaindex:{type(llm).__name__}"

    def generate(self, system: str, user: str) -> str:
        from llama_index.core.llms import ChatMessage, MessageRole
        resp = self.llm.chat([ChatMessage(role=MessageRole.SYSTEM, content=system),
                              ChatMessage(role=MessageRole.USER, content=user)])
        return _text(resp.message.content)


class BedrockLLM:
    def __init__(self, model_id: str | None = None, region: str | None = None, client=None):
        import boto3
        self.model_id = model_id or os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-5-5-v1:0")
        self.client = client or boto3.client("bedrock-runtime", region_name=region or os.getenv("AWS_REGION", "us-east-1"))
        self.profile = f"bedrock:{self.model_id}"

    def generate(self, system: str, user: str) -> str:
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": 800, "temperature": 0},
        )
        return "".join(b.get("text", "") for b in resp["output"]["message"]["content"])


class AzureFoundryLLM:
    def __init__(self, deployment: str | None = None, endpoint: str | None = None, api_key: str | None = None,
                 api_version: str = "2024-10-21", http_client=None):
        from openai import AzureOpenAI
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.client = AzureOpenAI(azure_endpoint=endpoint or os.environ["AZURE_OPENAI_ENDPOINT"],
                                  api_key=api_key or os.environ["AZURE_OPENAI_API_KEY"],
                                  api_version=api_version, http_client=http_client)
        self.profile = f"azure:{self.deployment}"

    def generate(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.deployment, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return resp.choices[0].message.content or ""
