from langgraph.graph import StateGraph, END
from state import NegotiationState
from buyer_agent import buyer_node
from supplier_agent import supplier_node
from evaluator import evaluator_node
from summarizer import final_summary


def build_graph():
    """
    Negotiation flow:
      Buyer -> Supplier -> Evaluator -> (Buyer again or Summarizer -> END)
    """
    graph = StateGraph(NegotiationState)

    # Nodes
    graph.add_node("buyer", buyer_node)
    graph.add_node("supplier", supplier_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("summarizer", final_summary)

    # Edges (turn order)
    graph.add_edge("buyer", "supplier")
    graph.add_edge("supplier", "evaluator")

    # Router: guarantee termination
    def route_from_evaluator(state: NegotiationState):
        # stop if agreement OR max rounds hit OR outcome already decided
        if state.agreement or state.round_number >= state.max_rounds or state.outcome in ("success", "failure"):
            if state.outcome == "pending":
                state.outcome = "success" if state.agreement else "failure"
            return "summarizer"
        return "buyer"

    graph.add_conditional_edges(
        "evaluator",
        route_from_evaluator,
        {"summarizer": "summarizer", "buyer": "buyer"},
    )

    graph.add_edge("summarizer", END)
    graph.set_entry_point("buyer")

    # ⬇️ DO NOT LOSE THIS LINE
    return graph.compile()
