from openai import OpenAI
from state import NegotiationState

client = OpenAI()

def _compute_final_price(state: NegotiationState):
    if state.buyer_offer is None or state.supplier_offer is None:
        return None
    if abs(state.buyer_offer - state.supplier_offer) < 1e-9:
        return float(state.buyer_offer)
    if state.agreement or abs(state.buyer_offer - state.supplier_offer) <= 50:
        return float(state.supplier_offer)
    return None

def final_summary(state: NegotiationState) -> NegotiationState:
    transcript = "\n".join([f"{m.agent.upper()}: {m.text}" for m in state.history])
    status = "success" if state.outcome == "success" else "failure"
    rounds = int(state.round_number)
    final_price = _compute_final_price(state)

    # Basic metric mirrors for the summary (optional if None)
    buyer_budget = getattr(state, "buyer_budget_ceiling", None)
    supplier_floor = getattr(state, "supplier_floor_price", None)

    discount_pct = None
    if final_price is not None and supplier_floor is not None:
        anchor_price = max(float(supplier_floor), float(state.supplier_offer or supplier_floor))
        discount_pct = max(0.0, (anchor_price - float(final_price)) / anchor_price)
        discount_pct = round(discount_pct, 4)

    prompt = f"""
You are an impartial observer of a business negotiation.

Outcome: {status.upper()}
ROUNDS_IN_STATE: {rounds}
FINAL_PRICE_IN_STATE: {final_price if final_price is not None else "N/A"}
DISCOUNT_PCT_IN_STATE: {discount_pct if discount_pct is not None else "N/A"}

Write a concise, bullet-style summary that MUST use those exact values for:
- "Number of Rounds": use ROUNDS_IN_STATE exactly.
- "Final Agreed Price": show FINAL_PRICE_IN_STATE, or "None" if N/A.
- If DISCOUNT_PCT_IN_STATE is present, display it as a percentage (e.g., 7.5%).
Also include:
- main reason for success/failure,
- one suggestion to improve next time.

Keep it crisp.

Transcript:
{transcript}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    state.summary = resp.choices[0].message.content.strip()
    return state
