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

# ============================================================
# HESTON MODEL TESTS
# ============================================================

def test_heston_simulation_shape():
    """Test Heston returns correct shapes"""
    from simulator import HestonSimulator
    
    num_sims = 500
    num_steps = 252
    
    sim = HestonSimulator(100.0, 0.2, 0.05, 1.0, num_steps)
    price_paths, vol_paths = sim.simulate(num_sims, seed=42)
    
    assert price_paths.shape == (num_sims, num_steps + 1)
    assert vol_paths.shape == (num_sims, num_steps + 1)
    print("✓ test_heston_simulation_shape passed")
    
def test_heston_initial_values():
    """Test Heston starts at correct initial values"""
    from simulator import HestonSimulator

    sim = HestonSimulator(100.0, 0.2, 0.05, 1.0, 252)
    price_paths, vol_paths = sim.simulate(100, seed=42)
        
    # All prices should start at 100
    assert np.allclose(price_paths[:, 0], 100.0)
    # All volatilites should start at 0.2^2 = 0.04
    assert np.allclose(vol_paths[:, 0], 0.04)
    print("✓ test_heston_initial_values passed")
    
def test_heston_positive_values():
    """Test Heston prices and volatilites stay positive"""
    from simulator import HestonSimulator

    sim = HestonSimulator(100.0, 0.2, 0.05, 1.0, 252)
    price_paths, vol_paths = sim.simulate(1000, seed=42)
        
    assert np.all(price_paths > 0), "Prices should be positive"
    assert np.all(vol_paths > 0), "Volatilities should be positive"
    print("✓ test_heston_positive_values passed")
        
def test_heston_volatility_mean_reversion():
    """Test that Heston volatility mean-reverts"""
    from simulator import HestonSimulator
    
    # Use mean-reverting parameters
    sim = HestonSimulator(100.0, 0.2, 0.05, 1.0, 252, kappa=2.0, xi=0.1)
    price_paths, vol_paths = sim.simulate(5000, seed=42)
    
    # Get volatilities (convert variance to volatility)
    vols = np.sqrt(vol_paths[:, -1])
    
    # Mean should be close to initial (mean reversion)
    initial_vol = 0.2
    final_vol_mean = np.mean(vols)
    
    # Should be reasonably close to initial (not wandering too far)
    assert abs(final_vol_mean - initial_vol) < initial_vol * 0.5, \
        f"Vol mean {final_vol_mean:.3f} should be close to initial {initial_vol}"
    
    print(f"✓ test_heston_volatility_mean_reversion passed")
    print(f"  Initial vol: {initial_vol:.3f}, Final vol mean: {final_vol_mean:.3f}")


def test_heston_vs_gbm_std_comparison():
    """Test that Heston typically has higher std than GBM"""
    from simulator import GBMSimulator, HestonSimulator
    
    # Both with same parameters
    gbm = GBMSimulator(100.0, 0.2, 0.05, 1.0, 252)
    heston = HestonSimulator(100.0, 0.2, 0.05, 1.0, 252)
    
    gbm_paths = gbm.simulate(10000, seed=42)
    heston_paths, _ = heston.simulate(10000, seed=42)
    
    gbm_std = np.std(gbm_paths[:, -1])
    heston_std = np.std(heston_paths[:, -1])
    
    # Heston should capture more tail risk (higher std)
    # But not too different (within 20%)
    assert heston_std > gbm_std * 0.9, "Heston should have similar or higher std"
    assert heston_std < gbm_std * 1.5, "Heston std shouldn't be wildly different"
    
    print(f"✓ test_heston_vs_gbm_std_comparison passed")
    print(f"  GBM std: ${gbm_std:.2f}, Heston std: ${heston_std:.2f}")


def test_heston_reproducibility():
    """Test Heston with same seed produces same results"""
    from simulator import HestonSimulator
    
    sim = HestonSimulator(100.0, 0.2, 0.05, 1.0, 252)
    
    paths1_p, paths1_v = sim.simulate(100, seed=42)
    paths2_p, paths2_v = sim.simulate(100, seed=42)
    
    assert np.allclose(paths1_p, paths2_p), "Prices should be identical with same seed"
    assert np.allclose(paths1_v, paths2_v), "Vols should be identical with same seed"
    print("✓ test_heston_reproducibility passed")


def test_heston_statistics_computation():
    """Test Heston statistics are computed correctly"""
    from simulator import HestonSimulator
    
    sim = HestonSimulator(100.0, 0.2, 0.05, 1.0, 252)
    price_paths, _ = sim.simulate(1000, seed=42)
    stats = sim.compute_statistics(price_paths)
    
    # Check all required stats are present
    required = ['mean', 'std', 'min', 'max', 'percentile_5', 'percentile_95', 'var_95']
    for stat in required:
        assert stat in stats, f"Missing stat: {stat}"
    
    # Check ordering is correct
    assert stats['min'] <= stats['percentile_5'] <= stats['mean']
    assert stats['mean'] <= stats['percentile_95'] <= stats['max']
    
    print("✓ test_heston_statistics_computation passed")


def test_heston_high_kappa():
    """Test high kappa (fast mean reversion) keeps volatility stable"""
    from simulator import HestonSimulator
    
    # High kappa = fast mean reversion = stable vol
    sim = HestonSimulator(100.0, 0.2, 0.05, 1.0, 252, kappa=5.0, xi=0.1)
    _, vol_paths = sim.simulate(5000, seed=42)
    
    initial_vol = 0.2
    final_vols = np.sqrt(vol_paths[:, -1])
    
    # With high kappa, volatility should stay very close to initial
    assert abs(final_vols.mean() - initial_vol) < initial_vol * 0.3
    print("✓ test_heston_high_kappa passed")


def test_heston_low_kappa():
    """Test low kappa (slow mean reversion) allows vol drift"""
    from simulator import HestonSimulator
    
    # Low kappa = slow mean reversion = vol can wander
    sim = HestonSimulator(100.0, 0.2, 0.05, 1.0, 252, kappa=0.5, xi=0.3)
    _, vol_paths = sim.simulate(5000, seed=42)
    
    vols = np.sqrt(vol_paths[:, -1])
    
    # With low kappa, volatility std should be higher (more wandering)
    vol_std = np.std(vols)
    
    # Should have meaningful volatility spread
    assert vol_std > 0.05, f"With low kappa, vol should vary: {vol_std:.4f}"
    print(f"✓ test_heston_low_kappa passed (vol std: {vol_std:.4f})")


def test_heston_negative_correlation():
    """Test negative correlation (realistic leverage effect)"""
    from simulator import HestonSimulator
    
    # Negative correlation = when prices down, vol up (realistic)
    sim = HestonSimulator(100.0, 0.2, 0.05, 1.0, 252, rho=-0.7)
    price_paths, vol_paths = sim.simulate(5000, seed=42)
    
    # Calculate returns and volatility changes
    returns = np.diff(price_paths, axis=1) / price_paths[:, :-1]
    vol_changes = np.sqrt(vol_paths[:, 1:]) - np.sqrt(vol_paths[:, :-1])
    
    # Correlation should be negative
    correlation = np.mean([np.corrcoef(returns[i], vol_changes[i])[0, 1] 
                          for i in range(min(100, returns.shape[0]))])
    
    assert correlation < -0.3, "With rho=-0.7, should see negative correlation"
    print(f"✓ test_heston_negative_correlation passed (corr: {correlation:.3f})")

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