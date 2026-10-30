"""
FastAPI service for the Monte Carlo Stock Price Simulator.

Run with: uvicorn api:app --reload
Docs:     http://localhost:8000/docs
"""
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from simulator import GBMSimulator, HestonSimulator

app = FastAPI(
    title="Monte Carlo Stock Simulator",
    description="Monte Carlo stock price simulation using GBM and Heston models",
    version="1.0.0",
)

VALID_MODELS = ("gbm", "heston")


# ---------- Models ----------

class SimulationRequest(BaseModel):
    stock_price: float = Field(..., gt=0, description="Current stock price")
    volatility: float = Field(..., ge=0, le=2, description="Annual volatility (0-2)")
    drift: float = Field(default=0.05, ge=0, le=0.2, description="Annual drift")
    time_horizon: float = Field(default=1.0, gt=0, le=10, description="Time horizon in years")
    num_simulations: int = Field(default=10000, ge=100, le=1_000_000)
    num_steps: int = Field(default=252, ge=1, le=1000)
    model: str = Field(default="gbm", description="'gbm' or 'heston'")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class SimulationStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    percentile_5: float
    percentile_95: float
    var_95: float


class SimulationResponse(BaseModel):
    stats: SimulationStats
    config: Dict
    model_used: str
    message: str


# ---------- Helpers ----------

def run_model(req: SimulationRequest):
    """Run the requested model and return (simulator, price_paths)."""
    if req.model not in VALID_MODELS:
        raise HTTPException(status_code=400, detail="model must be 'gbm' or 'heston'")

    if req.model == "gbm":
        sim = GBMSimulator(
            req.stock_price, req.volatility, req.drift, req.time_horizon, req.num_steps
        )
        paths = sim.simulate(req.num_simulations, seed=req.seed)
    else:
        sim = HestonSimulator(
            req.stock_price, req.volatility, req.drift, req.time_horizon, req.num_steps
        )
        paths, _vol_paths = sim.simulate(req.num_simulations, seed=req.seed)

    return sim, paths


# ---------- Endpoints ----------

@app.get("/", tags=["System"])
def root():
    return {"service": "monte-carlo-simulator", "docs": "/docs"}


@app.get("/health", tags=["System"])
def health():
    return {"status": "healthy", "service": "monte-carlo-simulator", "version": app.version}


@app.get("/info", tags=["System"])
def info():
    return {
        "service": "monte-carlo-simulator",
        "version": app.version,
        "models": list(VALID_MODELS),
        "endpoints": [
            "GET /health",
            "GET /info",
            "POST /simulate",
            "POST /compare",
            "GET /stats/percentiles",
        ],
        "capabilities": [
            "GBM and Heston price simulation",
            "Summary statistics and VaR (5th percentile)",
            "Side-by-side model comparison",
            "Final-price percentiles (1% to 99%)",
            "Reproducible runs via seed",
        ],
    }


@app.post("/simulate", response_model=SimulationResponse, tags=["Simulation"])
def simulate(req: SimulationRequest):
    """Run a full simulation with the chosen model and return summary statistics."""
    sim, paths = run_model(req)
    stats = sim.compute_statistics(paths)
    return SimulationResponse(
        stats=SimulationStats(**stats),
        config=req.model_dump(),
        model_used=req.model,
        message=f"{req.model.upper()} simulation completed with {req.num_simulations} paths",
    )


@app.post("/compare", tags=["Analytics"])
def compare(req: SimulationRequest):
    """Run GBM and Heston with the same inputs and compare their statistics."""
    _, gbm_paths = run_model(req.model_copy(update={"model": "gbm"}))
    _, heston_paths = run_model(req.model_copy(update={"model": "heston"}))

    gbm_sim = GBMSimulator(
        req.stock_price, req.volatility, req.drift, req.time_horizon, req.num_steps
    )
    heston_sim = HestonSimulator(
        req.stock_price, req.volatility, req.drift, req.time_horizon, req.num_steps
    )
    gbm_stats = gbm_sim.compute_statistics(gbm_paths)
    heston_stats = heston_sim.compute_statistics(heston_paths)

    return {
        "gbm": gbm_stats,
        "heston": heston_stats,
        "difference": {
            "mean_diff": heston_stats["mean"] - gbm_stats["mean"],
            "std_diff": heston_stats["std"] - gbm_stats["std"],
            "var_diff": heston_stats["var_95"] - gbm_stats["var_95"],
        },
    }


@app.get("/stats/percentiles", tags=["Analytics"])
def percentiles(
    stock_price: float = Query(..., gt=0, description="Current stock price"),
    volatility: float = Query(..., ge=0, le=2, description="Annual volatility (0-2)"),
    time_horizon: float = Query(1.0, gt=0, le=10, description="Time horizon in years"),
    num_simulations: int = Query(50000, ge=100, le=1_000_000, description="Number of paths"),
):
    """Final-price percentiles from a GBM simulation."""
    sim = GBMSimulator(stock_price, volatility, 0.05, time_horizon, 252)
    paths = sim.simulate(num_simulations)
    final_prices = paths[:, -1]

    return {
        "stock_price": stock_price,
        "volatility": volatility,
        "time_horizon": time_horizon,
        "percentiles": {
            f"p{p}": float(np.percentile(final_prices, p))
            for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        },
    }
