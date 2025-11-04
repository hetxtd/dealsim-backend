from typing import List, Literal, Optional
from pydantic import BaseModel

# Define allowed outcomes
Outcome = Literal["pending", "success", "failure"]

class Message(BaseModel):
    """
    Represents one message in the negotiation.
    """
    agent: Literal["buyer", "supplier"]
    text: str
    reasoning: Optional[str] = None
    offer: Optional[float] = None  # numeric price if present


class NegotiationState(BaseModel):
    """
    The shared memory both agents can read and update.
    """
    history: List[Message] = []
    round_number: int = 0
    max_rounds: int = 10

    buyer_offer: Optional[float] = None
    supplier_offer: Optional[float] = None
    buyer_budget_ceiling: float = 2000.0
    supplier_floor_price: float = 2200.0

    agreement: bool = False
    outcome: Outcome = "pending"
    summary: Optional[str] = None
    reasoning_summary: Optional[str] = None


def last_message_by(state: NegotiationState, who: Literal["buyer", "supplier"]) -> Optional[Message]:
    """
    Returns the most recent message from the given agent.
    """
    for m in reversed(state.history):
        if m.agent == who:
            return m
    return None
