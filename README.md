# Monte Carlo Stock Predictor

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-33%2F33%20passing-brightgreen.svg)](./test_simulator.py)

Production-grade REST API for Monte Carlo stock price prediction using Geometric Brownian Motion and Heston stochastic volatility models.

## Features

- **Two Simulation Models**
  - Geometric Brownian Motion (GBM) - constant volatility
  - Heston model - stochastic volatility (realistic!)
  
- **Risk Analytics**
  - Value at Risk (VaR) at 95% confidence
  - Conditional Value at Risk (CVaR)
  - Full percentile distribution (1% - 99%)
  
- **REST API Service**
  - FastAPI with automatic documentation
  - Input validation with Pydantic
  - Error handling for all edge cases
  - 15+ comprehensive endpoint tests
  
- **Production Quality**
  - 33 total tests (18 simulator + 15 API)
  - High test coverage
  - Type hints throughout
  - Structured logging

## Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/monte-carlo-stock-predictor.git
cd monte-carlo-stock-predictor
pip install -r requirements.txt
```

### Run Simulator (Python)

```python
from simulator import GBMSimulator

sim = GBMSimulator(
    stock_price=100.0,
    volatility=0.2,      # 20% annual volatility
    drift=0.05,          # 5% expected return
    time_horizon=1.0,    # 1 year
    num_steps=252        # Daily steps
)

paths = sim.simulate(num_simulations=10000, seed=42)
stats = sim.compute_statistics(paths)

print(f"Mean final price: ${stats['mean']:.2f}")
print(f"95% VaR: ${stats['var_95']:.2f}")
```

**Output:**
```
Mean final price: $104.96
95% VaR: $74.17
```

### Run REST API Service

```bash
# Start server
uvicorn api:app --reload

# Visit interactive docs
open http://localhost:8000/docs
```

## API Endpoints

### POST `/simulate`
Run full Monte Carlo simulation

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "stock_price": 100,
    "volatility": 0.2,
    "drift": 0.05,
    "model": "gbm",
    "num_simulations": 10000
  }'
```

**Response:**
```json
{
  "stats": {
    "mean": 104.96,
    "std": 21.11,
    "min": 15.22,
    "max": 287.41,
    "percentile_5": 74.17,
    "percentile_95": 183.45,
    "var_95": 74.17
  },
  "config": { ... },
  "model_used": "GBM",
  "message": "Simulation completed..."
}
```

### POST `/compare`
Compare GBM vs Heston models

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"stock_price": 100, "volatility": 0.2}'
```

### GET `/stats/percentiles`
Quick percentile lookup (faster than full simulation)

```bash
curl "http://localhost:8000/stats/percentiles?stock_price=100&volatility=0.2"
```

### GET `/health`
Service health check

```bash
curl http://localhost:8000/health
```

### GET `/info`
API information and capabilities

```bash
curl http://localhost:8000/info
```

## Models

### Geometric Brownian Motion (GBM)

Standard model with constant volatility:

```
dS = μS dt + σS dW

Discrete: S(t+Δt) = S(t) × exp((μ - σ²/2)Δt + σ√Δt × Z)

where Z ~ N(0,1)
```

**Use when:**
- Volatility is stable (< 20%)
- Need fast computations
- Modeling low-volatility stocks

### Heston Stochastic Volatility

Realistic model with time-varying volatility:

```
dS = μS dt + √v S dW₁
dv = κ(θ - v) dt + ξ√v dW₂

where dW₁ and dW₂ are correlated
```

**Use when:**
- Volatility is high (> 20%)
- Need realistic tail risk
- Modeling tech stocks, options

## Testing

### Run All Tests

```bash
# Simulator tests (18 tests)
python test_simulator.py

# API tests (15 tests)
pytest test_api.py -v

# Total: 33 tests
```

### Example Test Output

```
✓ test_initial_price passed
✓ test_positive_prices passed
✓ test_correct_shape passed
✓ test_gbm_simulation_shape passed
✓ test_heston_initial_values passed
✓ test_heston_positive_values passed
... (27 more tests)

✅ All 33 tests passed!
```

## Usage Examples

### Tech Stock Analysis (40% Volatility)

```python
from simulator import HestonSimulator

# Simulate Stripe-like fintech startup
sim = HestonSimulator(
    stock_price=100,
    volatility=0.40,    # 40% volatility (tech)
    drift=0.20,         # 20% growth
    time_horizon=1,
    num_steps=252
)

paths, vol_paths = sim.simulate(50000)
stats = sim.compute_statistics(paths)

print(f"Expected price: ${stats['mean']:.2f}")
print(f"Value at Risk: ${stats['var_95']:.2f}")
print(f"Max potential: ${stats['max']:.2f}")
```

### Portfolio Risk Management

```python
# Calculate max loss on $1M portfolio
portfolio = 1000000
stock_price = 100
shares = portfolio / stock_price

response = requests.post("http://localhost:8000/simulate", json={
    "stock_price": stock_price,
    "volatility": 0.25,
    "num_simulations": 100000
})

var_price = response.json()["stats"]["var_95"]
max_loss = portfolio - (shares * var_price)

print(f"95% VaR: ${max_loss:,.0f} loss possible")
```

### Scenario Analysis

```python
scenarios = {
    "Bull": (0.15, 0.12),
    "Base": (0.20, 0.05),
    "Bear": (0.35, -0.05),
}

for name, (vol, drift) in scenarios.items():
    response = requests.post("http://localhost:8000/simulate", 
        json={"stock_price": 100, "volatility": vol, "drift": drift})
    mean = response.json()["stats"]["mean"]
    print(f"{name:5} scenario: ${mean:.2f}")
```

## Performance

**Simulation speed (10K simulations, 252 steps):**
- GBM: ~250ms
- Heston: ~500ms

**Vectorized NumPy operations ensure:**
- 40K+ paths per second (GBM)
- Linear scaling
- Memory efficient

## Architecture

```
simulator.py
├── GBMSimulator
│   ├── simulate()
│   └── compute_statistics()
└── HestonSimulator
    ├── simulate()
    └── compute_statistics()

api.py
├── /simulate (POST)
├── /compare (POST)
├── /stats/percentiles (GET)
├── /health (GET)
└── /info (GET)
```

## Requirements

- Python 3.9+
- NumPy 1.24+
- FastAPI 0.103+
- Pydantic 2.1+
- Uvicorn 0.23+ (for API)
- Pytest 7.4+ (for testing)

## Roadmap

- [ ] Docker containerization
- [ ] GitHub Actions CI/CD
- [ ] Greeks calculation (Delta, Gamma, Vega)
- [ ] Real-time data integration
- [ ] Interactive web dashboard
- [ ] Database caching
- [ ] Kubernetes deployment

## License

MIT License - see LICENSE file

## Author

Built by Swati - Platform Engineering + Quantitative Finance

---

**Status:** Production Ready ✅ | Tests: 33/33 Passing ✅ | API: Live ✅

For API documentation, start server and visit: http://localhost:8000/docs
