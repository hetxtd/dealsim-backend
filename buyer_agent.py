import json
from openai import OpenAI
from state import NegotiationState, Message, last_message_by

client = OpenAI()

def _safe_number_from_text(text: str):
    import re
    nums = re.findall(r"\d+(?:\.\d+)?", text or "")
    return float(nums[-1]) if nums else None

def buyer_node(state: NegotiationState) -> NegotiationState:
    """
    Buyer reads supplier's last move + constraints and proposes/accepts.
    Uses structured JSON so we always get {message, offer, reasoning}.
    """
    supplier_last = last_message_by(state, "supplier")

    system = f"""You are the BUYER agent in a price negotiation.
Budget ceiling: {state.buyer_budget_ceiling}.
Supplier's last offer: {state.supplier_offer}.
Goal: buy at or below the ceiling. Be cooperative and concise."""

    user = """Return a STRICT JSON object with these keys ONLY:
{
  "message": "<what you say to the supplier, 1-2 sentences>",
  "offer": <number or null>,
  "reasoning": "<1 short sentence explaining your move>"
}

Rules:
- If supplier_offer <= ceiling, accept and set "message" to 'I accept <price>.' and "offer" to that price.
- Otherwise, counter with a realistic offer (number) moving towards an agreement.
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
        # Fallback to heuristic if model misbehaves (rare with response_format)
        text = resp.choices[0].message.content
        return _buyer_fallback(state, text)

    message = str(payload.get("message", "")).strip()
    offer = payload.get("offer", None)
    reasoning = str(payload.get("reasoning", "")).strip()

    # Auto-accept safeguard in code as well
    if state.supplier_offer is not None and state.supplier_offer <= state.buyer_budget_ceiling:
        state.agreement = True
        offer = float(state.supplier_offer)
        message = f"I accept {offer}."

    # If model omitted an offer when countering, try to recover
    if offer is None and "accept" not in message.lower():
        offer = _safe_number_from_text(message)

    state.buyer_offer = float(offer) if offer is not None else None
    state.history.append(Message(agent="buyer", text=message, reasoning=reasoning or "Buyer move.", offer=state.buyer_offer))
    state.round_number += 1
    return state


def _buyer_fallback(state: NegotiationState, text: str) -> NegotiationState:
    """Heuristic fallback if JSON parsing fails."""
    offer = _safe_number_from_text(text)
    if state.supplier_offer is not None and state.supplier_offer <= state.buyer_budget_ceiling:
        state.agreement = True
        offer = float(state.supplier_offer)
        text = f"I accept {offer}."
    state.buyer_offer = offer
    state.history.append(Message(agent="buyer", text=text, reasoning="Buyer (fallback).", offer=offer))
    state.round_number += 1
    return state
