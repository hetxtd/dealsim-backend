from state import NegotiationState

def evaluator_node(state: NegotiationState) -> NegotiationState:
    """
    The Evaluator checks whether the negotiation should continue or end.
    """
    # 1️⃣ Success if either agent already accepted
    if state.agreement:
        state.outcome = "success"
        return state

    # 2️⃣ Soft success: if offers are close enough (within £50)
    if state.buyer_offer is not None and state.supplier_offer is not None:
        if abs(state.buyer_offer - state.supplier_offer) <= 50:
            state.agreement = True
            state.outcome = "success"
            return state

    # 3️⃣ Failure: if too many rounds have happened
    if state.round_number >= state.max_rounds:
        state.outcome = "failure"
        return state

    # 4️⃣ Otherwise, keep looping
    return state
