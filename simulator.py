"""
Monte Carlo Stock Price Simulator - Day 1 Implementation
Geometric Brownian Motion (GBM) Model

Formula: S(t+Δt) = S(t) * exp((μ - σ²/2)Δt + σ√Δt * Z)
where Z ~ N(0,1)
"""

import numpy as np
from typing import Dict, Tuple


class GBMSimulator:
    """
    Geometric Brownian Motion simulator for stock price prediction
    
    Implements the discrete-time GBM model commonly used in quantitative finance
    for modeling asset prices that cannot go negative.
    """
    
    def __init__(self, stock_price: float, volatility: float, drift: float, 
                 time_horizon: float, num_steps: int):
        """
        Initialize GBM simulator
        
        Args:
            stock_price: Initial stock price (e.g., 100.0)
            volatility: Annual volatility as decimal (e.g., 0.2 = 20%)
            drift: Expected annual return as decimal (e.g., 0.05 = 5%)
            time_horizon: Time period in years (e.g., 1.0)
            num_steps: Number of time steps (e.g., 252 for daily)
        
        Example:
            >>> sim = GBMSimulator(100, 0.2, 0.05, 1, 252)
            >>> paths = sim.simulate(1000)
        """
        self.S0 = stock_price
        self.sigma = volatility
        self.mu = drift
        self.T = time_horizon
        self.N = num_steps
        self.dt = time_horizon / num_steps
        
    def simulate(self, num_simulations: int = 100, seed: int = None) -> np.ndarray:
        """
        Simulate stock price paths using GBM
        
        Args:
            num_simulations: Number of price paths to simulate
            seed: Random seed for reproducibility (optional)
        
        Returns:
            Array of shape (num_simulations, num_steps + 1)
            Each row is one complete price path
            First column is initial price (stock_price for all rows)
            Last column is final price after time_horizon
        
        Example:
            >>> paths = sim.simulate(1000, seed=42)
            >>> paths.shape
            (1000, 253)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize storage for all price paths
        paths = np.zeros((num_simulations, self.N + 1))
        paths[:, 0] = self.S0  # Set initial prices
        
        # Pre-compute drift and diffusion coefficients (same for all steps)
        drift_term = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        diffusion_coeff = self.sigma * np.sqrt(self.dt)
        
        # Generate all price steps
        for step in range(self.N):
            # Generate random normals for all simulations at this step
            Z = np.random.standard_normal(num_simulations)
            
            # Apply GBM formula: S(t+dt) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
            paths[:, step + 1] = paths[:, step] * np.exp(drift_term + diffusion_coeff * Z)
        
        return paths
    
    def compute_statistics(self, paths: np.ndarray) -> Dict[str, float]:
        """
        Compute statistics from simulated paths
        
        Args:
            paths: Output from simulate() method
        
        Returns:
            Dictionary with computed statistics
        
        Example:
            >>> stats = sim.compute_statistics(paths)
            >>> print(f"Mean: ${stats['mean']:.2f}")
            Mean: $104.96
        """
        final_prices = paths[:, -1]
        
        return {
            "mean": float(np.mean(final_prices)),
            "std": float(np.std(final_prices)),
            "min": float(np.min(final_prices)),
            "max": float(np.max(final_prices)),
            "percentile_5": float(np.percentile(final_prices, 5)),
            "percentile_95": float(np.percentile(final_prices, 95)),
            "var_95": float(np.percentile(final_prices, 5)),  # Value at Risk (5th percentile)
        }


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Monte Carlo Stock Price Simulator - Day 1")
    print("=" * 60)
    
    # Create simulator with reasonable parameters
    sim = GBMSimulator(
        stock_price=100.0,      # Start at $100
        volatility=0.2,         # 20% annual volatility
        drift=0.05,             # 5% expected annual return
        time_horizon=1.0,       # 1 year
        num_steps=252           # Daily time steps (trading days)
    )
    
    # Run 10,000 simulations
    print("\nRunning 10,000 simulations...")
    paths = sim.simulate(num_simulations=10000, seed=42)
    
    # Compute statistics
    stats = sim.compute_statistics(paths)
    
    # Print results
    print(f"\nResults after 1 year:")
    print(f"  Mean final price:     ${stats['mean']:.2f}")
    print(f"  Std deviation:        ${stats['std']:.2f}")
    print(f"  Min price:            ${stats['min']:.2f}")
    print(f"  Max price:            ${stats['max']:.2f}")
    print(f"  5th percentile (VaR): ${stats['var_95']:.2f}")
    print(f"  95th percentile:      ${stats['percentile_95']:.2f}")
    print("\n" + "=" * 60)
    
class HestonSimulator:
    """
    Heston stochastic volatility simulator
    
    Models both price  AND volatility as stochastic processes:
    dS = μS dt + √v S dW₁
    dv = κ(θ - v) dt + ξ√v dW₂
    
    where dW₁ and dW₂ are correlated with ρ
    """
    
    def __init__(self, stock_price: float, volatility: float, drift: float,
                 time_horizon: float, num_steps: int,
                 kappa: float = 2.0, xi: float = 0.1, rho: float = -0.7):
        """Initialize Heston simulator
        Args:
            stock_price: Initial stock price
            volatility: Initial volatility (also used for theta if not specified)
            drift: Expected return
            time_horizon: Time period in years
            num_steps: Number of time steps
            kappa: Mean reversion speed (how fast vol reverts to long-term)
            xi: Volatility of volatility (how volatile is the volatility?)
            rho: Correlation between price and vol shocks
        """
        self.S0 = stock_price
        self.sigma_0 = volatility
        self.mu = drift
        self.T = time_horizon
        self.N = num_steps
        self.dt = time_horizon / num_steps
        
        # Heston-specific parameters
        self.kappa = kappa  # Mean reversion speed
        self.theta = volatility ** 2  # Long-term variance (use vol² as estimate)
        self.xi = xi  # Volatility of volatility
        self.rho = rho  # Correlation between price and vol shocks
        
    def simulate(self, num_simulations: int = 100, seed: int = None) -> tuple:
        """
        Simulate price paths using Heston model
        
        Returns:
           Tuple of (price_paths, volatility_paths)
           - price_paths: shape (num_simulations, num_steps + 1)
           - volatility_paths: shape (num_simulations, num_steps + 1) 
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize storage
        price_paths = np.zeros((num_simulations, self.N + 1))
        vol_paths = np.zeros((num_simulations, self.N + 1))
        
        # Set initial values
        price_paths[:, 0] = self.S0
        vol_paths[:, 0] = self.sigma_0 ** 2  # Store variance, not volatility
        
        # Generate price and volatility paths
        for step in range(self.N):
            # Generate two correlated random normals
            # Z1: Drives price (dS)
            # Z2: Drives volatility (dv)
            Z1 = np.random.standard_normal(num_simulations)
            Z2 = np.random.standard_normal(num_simulations)
            
            # Correlate Z2 with Z1 using Cholesky decomposition
            # Z2_corr = rho * Z1 + sqrt(1 - rho²) * Z2
            Z2_corr = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * Z2
            
            # Current variance and volatility
            v_t = vol_paths[:, step]
            s_t = price_paths[:, step]
            
            # Update variance (Euler scheme)
            # dv = κ(θ - v) dt + ξ√v √dt Z1
            dv = (self.kappa * (self.theta - - v_t) * self.dt +
                  self.xi * np.sqrt(np.maximum(v_t, 0)) * np.sqrt(self.dt) * Z1)
            
            v_new = v_t + dv
            # Ensure variance stays positive (Feller condition)
            v_new = np.maximum(v_new, 1e-6)
            vol_paths[:, step + 1] = v_new
            
            # Update price (using NEW variance)
            # dS = μS dt + √v S √dt Z2
            # S_new = S * exp(μ dt + √v √dt Z2)
            drift_term = self.mu * self.dt
            diffusion_term = np.sqrt(v_new * self.dt) * Z2_corr
            
            price_paths[:, step + 1] = S_t * np.exp(drift_term + diffusion_term)
            
        return price_paths, vol_paths
    
    def compute_statistics(self, paths: np.ndarray) -> Dict[str, float]:
        """Compute statistics from price paths"""
        final_prices = paths[:, -1]
        
        return {
            "mean": float(np.mean(final_prices)),
            "std": float(np.std(final_prices)),
            "min": float(np.min(final_prices)),
            "max": float(np.max(final_prices)),
            "percentile_5": float(np.percentile(final_prices, 5)),
            "percentile_95": float(np.percentile(final_prices, 95)),
            "var_95": float(np.percentile(final_prices, 5)),
        }