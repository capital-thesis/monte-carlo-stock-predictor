"""
Tests for Monte Carlo Simulator
Run with: python -m pytest test_simulator.py -v
Or simply: python test_simulator.py
"""

import numpy as np
from simulator import GBMSimulator


def test_initial_price():
    """Test that all simulations start at the correct initial price"""
    sim = GBMSimulator(100.0, 0.2, 0.05, 1.0, 252)
    paths = sim.simulate(100, seed=42)
    
    # All paths should start at 100
    assert np.allclose(paths[:, 0], 100.0), "Initial prices should be 100"
    print("✓ test_initial_price passed")


def test_positive_prices():
    """Test that GBM never produces negative prices (it shouldn't mathematically)"""
    sim = GBMSimulator(100.0, 0.2, 0.05, 1.0, 252)
    paths = sim.simulate(1000, seed=42)
    
    # All prices should be positive
    assert np.all(paths > 0), "GBM should never produce negative prices"
    print("✓ test_positive_prices passed")


def test_correct_shape():
    """Test that output shape is correct"""
    num_sims = 500
    num_steps = 252
    
    sim = GBMSimulator(100.0, 0.2, 0.05, 1.0, num_steps)
    paths = sim.simulate(num_sims, seed=42)
    
    # Should be (num_simulations, num_steps + 1)
    assert paths.shape == (num_sims, num_steps + 1), \
        f"Expected shape ({num_sims}, {num_steps + 1}), got {paths.shape}"
    print("✓ test_correct_shape passed")


def test_drift_effect():
    """Test that positive drift increases expected final price"""
    # Simulator with 5% drift
    sim_pos = GBMSimulator(100.0, 0.05, 0.05, 1.0, 252)
    paths_pos = sim_pos.simulate(5000, seed=42)
    mean_pos = np.mean(paths_pos[:, -1])
    
    # Simulator with 0% drift
    sim_zero = GBMSimulator(100.0, 0.05, 0.0, 1.0, 252)
    paths_zero = sim_zero.simulate(5000, seed=42)
    mean_zero = np.mean(paths_zero[:, -1])
    
    # Positive drift should lead to higher mean
    assert mean_pos > mean_zero, "Positive drift should increase expected price"
    print(f"✓ test_drift_effect passed (5% drift: ${mean_pos:.2f} > 0% drift: ${mean_zero:.2f})")


def test_volatility_effect():
    """Test that higher volatility increases standard deviation"""
    # Low volatility
    sim_low = GBMSimulator(100.0, 0.05, 0.05, 1.0, 252)
    paths_low = sim_low.simulate(5000, seed=42)
    std_low = np.std(paths_low[:, -1])
    
    # High volatility
    sim_high = GBMSimulator(100.0, 0.50, 0.05, 1.0, 252)
    paths_high = sim_high.simulate(5000, seed=42)
    std_high = np.std(paths_high[:, -1])
    
    # Higher volatility should have higher std
    assert std_high > std_low, "Higher volatility should increase price spread"
    print(f"✓ test_volatility_effect passed (Low vol std: ${std_low:.2f}, High vol std: ${std_high:.2f})")


def test_reproducibility():
    """Test that same seed produces same results"""
    sim = GBMSimulator(100.0, 0.2, 0.05, 1.0, 252)
    
    # Run twice with same seed
    paths1 = sim.simulate(100, seed=42)
    paths2 = sim.simulate(100, seed=42)
    
    # Should be identical
    assert np.allclose(paths1, paths2), "Same seed should produce identical paths"
    print("✓ test_reproducibility passed")


def test_statistics_computation():
    """Test that statistics are computed correctly"""
    sim = GBMSimulator(100.0, 0.2, 0.05, 1.0, 252)
    paths = sim.simulate(1000, seed=42)
    stats = sim.compute_statistics(paths)
    
    # Check that all statistics are present
    required_stats = ['mean', 'std', 'min', 'max', 'percentile_5', 'percentile_95', 'var_95']
    for stat in required_stats:
        assert stat in stats, f"Missing statistic: {stat}"
    
    # Check sanity
    assert stats['min'] <= stats['percentile_5'] <= stats['mean'] <= stats['percentile_95'] <= stats['max'], \
        "Statistics ordering is wrong"
    
    print("✓ test_statistics_computation passed")


def test_zero_volatility():
    """Test edge case: zero volatility should be deterministic"""
    sim = GBMSimulator(100.0, 0.0, 0.05, 1.0, 252)
    paths = sim.simulate(100, seed=42)
    
    # All paths should be identical
    for i in range(1, 100):
        assert np.allclose(paths[i], paths[0]), f"Path {i} differs from path 0"
    
    # Expected final price = S0 * exp(r*T)
    expected = 100.0 * np.exp(0.05 * 1.0)
    actual = paths[0, -1]
    
    assert np.isclose(actual, expected), \
        f"With zero vol, expected {expected:.2f}, got {actual:.2f}"
    
    print(f"✓ test_zero_volatility passed (deterministic: ${actual:.2f})")


if __name__ == "__main__":
    # Run all tests
    print("\n" + "=" * 60)
    print("Running Tests...")
    print("=" * 60 + "\n")
    
    test_initial_price()
    test_positive_prices()
    test_correct_shape()
    test_drift_effect()
    test_volatility_effect()
    test_reproducibility()
    test_statistics_computation()
    test_zero_volatility()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60 + "\n")