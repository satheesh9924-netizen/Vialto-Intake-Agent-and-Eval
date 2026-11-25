"""The same agent, expressed as a LangGraph state machine.

Why have two? `DocumentIntakeAgent` walks its steps in a for-loop. `GraphAgent` draws the SAME steps
as a graph: nodes (steps) and edges (what runs next). The business logic is not duplicated: every
node calls the same `step_*` method. Only the wiring differs.

What the graph form buys you:
  * the flow is explicit and can be drawn (see `mermaid()`), which is easy to explain and review
  * conditional edges make "stop early" a first-class idea (rejected input, low confidence, model down)
  * it is the shape you would extend with loops, human-in-the-loop pauses, or checkpoints
A test (`tests/test_graph_parity.py`) proves both agents give identical results on every document.
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .agent import STEP_LABELS, AgentResult, Ctx, DocumentIntakeAgent


class State(TypedDict):
    result: AgentResult   # what we are building and will return
    ctx: Ctx              # working data passed between steps


class GraphAgent(DocumentIntakeAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(State)
        names = self.STEPS

        def make_node(step_name):
            fn = getattr(self, f"step_{step_name}")

            def node(state: State) -> State:
                fn(state["result"], state["ctx"])
                return {"result": state["result"], "ctx": state["ctx"]}
            return node

        for n in names:
            g.add_node(n, make_node(n))
        g.add_edge(START, names[0])
        for cur, nxt in zip(names, names[1:]):
            # conditional edge: stop early if a step set ctx.done (rejected, low confidence, model down...)
            g.add_conditional_edges(cur, lambda s, nxt=nxt: END if s["ctx"].done else nxt, [nxt, END])
        g.add_edge(names[-1], END)
        return g.compile()

    def _run(self, r: AgentResult, text: str):
        self.graph.invoke({"result": r, "ctx": Ctx(text=text, doc=text)})

    def mermaid(self) -> str:
        """Diagram source you can paste into any Mermaid viewer (GitHub renders it natively)."""
        return self.graph.get_graph().draw_mermaid()


AGENTS = {"pipeline": DocumentIntakeAgent, "langgraph": GraphAgent}


def agent_class(name: str):
    return AGENTS[name]
