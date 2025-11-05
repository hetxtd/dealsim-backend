from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from state import NegotiationState
from graph import build_graph
import os, traceback

app = FastAPI(title="DealSim API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://frontend-git-main-nasirs-projects-51b6d1bf.vercel.app",  # your Vercel live site
        "http://localhost:3000",  # for local dev
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Build the graph once at startup
graph = build_graph()

# Simple OpenAI client for /health
client = OpenAI()


@app.get("/")
def root():
    return {"message": "Welcome to DealSim — AI Agent Negotiation Simulator"}


@app.get("/health")
def health():
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply READY"}],
            temperature=0,
        )
        return {"ok": True, "sample": r.choices[0].message.content}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": str(e),
                "has_key": bool(os.getenv("OPENAI_API_KEY")),
            },
        )


@app.post("/run")
def run_simulation(
    buyer_budget: float = 2000.0,
    supplier_floor: float = 2200.0,
    max_rounds: int = 10,
):
    try:
        # 1) Seed state
        state = NegotiationState(
            buyer_budget_ceiling=buyer_budget,
            supplier_floor_price=supplier_floor,
            max_rounds=max_rounds,
        )

        # 2) Run graph
        final_state = graph.invoke(state, config={"recursion_limit": 100})

        # 3) Extract state (supports dict or pydantic object)
        if isinstance(final_state, dict):
            outcome = final_state.get("outcome")
            rounds = final_state.get("round_number")
            summary = final_state.get("summary")
            history = final_state.get("history", [])
            buyer_offer = final_state.get("buyer_offer")
            supplier_offer = final_state.get("supplier_offer")
        else:
            outcome = final_state.outcome
            rounds = final_state.round_number
            summary = final_state.summary
            history = [m.dict() for m in final_state.history]
            buyer_offer = final_state.buyer_offer
            supplier_offer = final_state.supplier_offer

        # 4) Deterministic final price
        final_price = None
        if buyer_offer is not None and supplier_offer is not None:
            if abs(buyer_offer - supplier_offer) < 1e-9:
                final_price = float(buyer_offer)
            elif outcome == "success" and abs(buyer_offer - supplier_offer) <= 50:
                final_price = float(supplier_offer)

        # 5) Deal metrics (derived)
        discount_pct = None
        buyer_satisfaction = None
        supplier_satisfaction = None

        if final_price is not None:
            # Anchor for discount: prefer last supplier offer, else supplier floor
            anchor_price = float(supplier_offer) if supplier_offer is not None else float(supplier_floor)
            anchor_price = max(anchor_price, float(supplier_floor))

            discount_pct = max(0.0, (anchor_price - float(final_price)) / anchor_price)
            discount_pct = round(discount_pct, 4)

            # Satisfaction heuristics within a ±25% band
            def clamp01(x: float) -> float:
                return max(0.0, min(1.0, x))

            buyer_range = max(1e-6, float(buyer_budget) * 0.25)
            supplier_range = max(1e-6, float(supplier_floor) * 0.25)

            buyer_satisfaction = 10.0 * clamp01((float(buyer_budget) - float(final_price)) / buyer_range)
            supplier_satisfaction = 10.0 * clamp01((float(final_price) - float(supplier_floor)) / supplier_range)

            buyer_satisfaction = round(buyer_satisfaction, 1)
            supplier_satisfaction = round(supplier_satisfaction, 1)

        # 6) Response
        return {
            "outcome": outcome,
            "rounds": rounds,
            "final_price": final_price,
            "discount_pct": discount_pct,
            "buyer_satisfaction": buyer_satisfaction,
            "supplier_satisfaction": supplier_satisfaction,
            "summary": summary,
            "history": history,
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
