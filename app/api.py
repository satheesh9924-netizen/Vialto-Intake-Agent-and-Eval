"""Small REST API around the agent, so the pipeline can be tested at API level (Python + Postman/Newman)."""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import DocumentIntakeAgent
from .llm import MockLLM
from .tools import ToolBox


class IntakeRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=100_000)


def create_app(db_path: str | None = None, llm=None) -> FastAPI:
    app = FastAPI(title="VLabs document intake (sample)")
    tools = ToolBox(db_path or os.getenv("INTAKE_DB", ":memory:"))
    if llm is None and os.getenv("INTAKE_LLM", "mock").lower() == "gemini":   # INTAKE_LLM=gemini uses the real model
        from .gemini import GeminiLLM
        llm = GeminiLLM()
    agent = DocumentIntakeAgent(llm or MockLLM("v1"), toolbox=tools)
    app.state.tools, app.state.agent = tools, agent

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/intake")
    def intake(req: IntakeRequest):
        res = agent.process(req.doc_id, req.text)
        body = res.to_dict()
        body.pop("trace", None)
        body.pop("raw_fields", None)
        return body

    @app.get("/cases/{doc_id}")
    def get_case(doc_id: str):
        case = tools.get_case(doc_id)
        if not case:
            raise HTTPException(status_code=404, detail="case not found")
        return case

    return app


app = create_app()
