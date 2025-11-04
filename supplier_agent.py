import json
from openai import OpenAI
from state import NegotiationState, Message, last_message_by

client = OpenAI()

def _safe_number_from_text(text: str):
    import re
    nums = re.findall(r"\d+(?:\.\d+)?", text or "")
    return float(nums[-1]) if nums else None

def supplier_node(state: NegotiationState) -> NegotiationState:
    """
    Supplier responds to the buyer using structured JSON: {message, offer, reasoning}.
    """
    buyer_last = last_message_by(state, "buyer")

    system = f"""You are the SUPPLIER agent in a price negotiation.
Floor price: {state.supplier_floor_price}.
Buyer's last offer: {state.buyer_offer}.
Goal: maximize profit but close reasonable deals. Be concise and cooperative."""

    user = """Return a STRICT JSON object with these keys ONLY:
{
  "message": "<what you say to the buyer, 1-2 sentences>",
  "offer": <number or null>,
  "reasoning": "<1 short sentence explaining your move>"
}

Rules:
- If buyer_offer >= floor, accept and set "message" to 'Accepted at <price>.' and "offer" to that price.
- Otherwise, counter with a realistic offer (number) moving toward the buyer's budget.
- Consider offering terms (quantity, faster payment) briefly in the message.
- No extra keys, no markdown, no prose around the JSON.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    # Parse structured JSON
    try:
        payload = json.loads(resp.choices[0].message.content)
    except Exception:
        text = resp.choices[0].message.content
        return _supplier_fallback(state, text)

    message = str(payload.get("message", "")).strip()
    offer = payload.get("offer", None)
    reasoning = str(payload.get("reasoning", "")).strip()

    # Auto-accept safeguard
    if state.buyer_offer is not None and state.buyer_offer >= state.supplier_floor_price:
        state.agreement = True
        offer = float(state.buyer_offer)
        message = f"Accepted at {offer}."

    if offer is None and "accept" not in message.lower():
        offer = _safe_number_from_text(message)

    state.supplier_offer = float(offer) if offer is not None else None
    state.history.append(Message(agent="supplier", text=message, reasoning=reasoning or "Supplier move.", offer=state.supplier_offer))
    return state


def _supplier_fallback(state: NegotiationState, text: str) -> NegotiationState:
    """Heuristic fallback if JSON parsing fails."""
    offer = _safe_number_from_text(text)
    if state.buyer_offer is not None and state.buyer_offer >= state.supplier_floor_price:
        state.agreement = True
        offer = float(state.buyer_offer)
        text = f"Accepted at {offer}."
    state.supplier_offer = offer
    state.history.append(Message(agent="supplier", text=text, reasoning="Supplier (fallback).", offer=offer))
    return state
