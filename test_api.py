"""
Day 3: API Unit Tests
Run with: pytest test_api.py -v

Tests the FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from api import app

# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint"""
    
    def test_health_check_returns_200(self):
        """Health endpoint should return 200"""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_check_has_required_fields(self):
        """Health response should have status, service, version"""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert data["status"] == "healthy"
    
    print("✓ Health endpoint tests ready")


class TestSimulateEndpoint:
    """Tests for /simulate endpoint"""
    
    def test_simulate_gbm_returns_200(self):
        """GBM simulation should succeed"""
        response = client.post("/simulate", json={
            "stock_price": 100,
            "volatility": 0.2,
            "model": "gbm",
            "num_simulations": 100
        })
        assert response.status_code == 200
    
    def test_simulate_heston_returns_200(self):
        """Heston simulation should succeed"""
        response = client.post("/simulate", json={
            "stock_price": 100,
            "volatility": 0.2,
            "model": "heston",
            "num_simulations": 100
        })
        assert response.status_code == 200
    
    def test_simulate_response_has_stats(self):
        """Response should contain statistics"""
        response = client.post("/simulate", json={
            "stock_price": 100,
            "volatility": 0.2,
            "num_simulations": 100
        })
        data = response.json()
        assert "stats" in data
        assert "mean" in data["stats"]
        assert "std" in data["stats"]
        assert "var_95" in data["stats"]
    
    def test_simulate_response_has_config(self):
        """Response should echo back configuration"""
        response = client.post("/simulate", json={
            "stock_price": 100,
            "volatility": 0.2,
            "num_simulations": 500
        })
        data = response.json()
        assert "config" in data
        assert data["config"]["stock_price"] == 100
        assert data["config"]["num_simulations"] == 500
    
    def test_simulate_stats_values_make_sense(self):
        """Simulation stats should be sensible"""
        response = client.post("/simulate", json={
            "stock_price": 100,
            "volatility": 0.2,
            "num_simulations": 1000
        })
        stats = response.json()["stats"]
        
        # Min should be less than mean
        assert stats["min"] < stats["mean"]
        # Mean should be less than max
        assert stats["mean"] < stats["max"]
        # Percentiles should be in order
        assert stats["percentile_5"] < stats["percentile_95"]
        # Std should be positive
        assert stats["std"] > 0
    
    def test_simulate_with_seed_is_reproducible(self):
        """Same seed should produce same results"""
        request_data = {
            "stock_price": 100,
            "volatility": 0.2,
            "num_simulations": 100,
            "seed": 42
        }
        
        response1 = client.post("/simulate", json=request_data)
        response2 = client.post("/simulate", json=request_data)
        
        stats1 = response1.json()["stats"]
        stats2 = response2.json()["stats"]
        
        assert stats1["mean"] == stats2["mean"]
        assert stats1["std"] == stats2["std"]
    
    print("✓ Simulate endpoint tests ready")


class TestInputValidation:
    """Tests for input validation"""
    
    def test_negative_stock_price_rejected(self):
        """Negative stock price should be rejected"""
        response = client.post("/simulate", json={
            "stock_price": -100,
            "volatility": 0.2
        })
        assert response.status_code == 422
    
    def test_invalid_model_rejected(self):
        """Invalid model name should be rejected"""
        response = client.post("/simulate", json={
            "stock_price": 100,
            "volatility": 0.2,
            "model": "invalid_model"
        })
        assert response.status_code == 400
    
    def test_volatility_too_high_rejected(self):
        """Volatility > 2.0 should be rejected"""
        response = client.post("/simulate", json={
            "stock_price": 100,
            "volatility": 5.0
        })
        assert response.status_code == 422
    
    def test_zero_stock_price_rejected(self):
        """Stock price = 0 should be rejected"""
        response = client.post("/simulate", json={
            "stock_price": 0,
            "volatility": 0.2
        })
        assert response.status_code == 422
    
    def test_missing_required_field(self):
        """Missing stock_price should be rejected"""
        response = client.post("/simulate", json={
            "volatility": 0.2
        })
        assert response.status_code == 422
    
    print("✓ Input validation tests ready")


class TestCompareEndpoint:
    """Tests for /compare endpoint"""
    
    def test_compare_returns_both_models(self):
        """Compare should return both GBM and Heston results"""
        response = client.post("/compare", json={
            "stock_price": 100,
            "volatility": 0.2,
            "num_simulations": 100
        })
        assert response.status_code == 200
        data = response.json()
        assert "gbm" in data
        assert "heston" in data
        assert "difference" in data
    
    def test_compare_shows_differences(self):
        """Compare should calculate differences"""
        response = client.post("/compare", json={
            "stock_price": 100,
            "volatility": 0.2,
            "num_simulations": 1000
        })
        data = response.json()
        assert "mean_diff" in data["difference"]
        assert "std_diff" in data["difference"]
        assert "var_diff" in data["difference"]
    
    print("✓ Compare endpoint tests ready")


class TestPercentileEndpoint:
    """Tests for /stats/percentiles endpoint"""
    
    def test_percentiles_returns_all_percentiles(self):
        """Should return percentiles from 1% to 99%"""
        response = client.get("/stats/percentiles?stock_price=100&volatility=0.2")
        assert response.status_code == 200
        data = response.json()
        assert "percentiles" in data
        percentiles = data["percentiles"]
        
        # Check all percentiles are present
        for p in ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]:
            assert p in percentiles
    
    def test_percentiles_are_ordered(self):
        """Percentiles should be in increasing order"""
        response = client.get("/stats/percentiles?stock_price=100&volatility=0.2")
        data = response.json()
        p = data["percentiles"]
        
        assert p["p1"] < p["p5"] < p["p10"] < p["p25"] < p["p50"]
        assert p["p50"] < p["p75"] < p["p90"] < p["p95"] < p["p99"]
    
    print("✓ Percentile endpoint tests ready")


class TestInfoEndpoint:
    """Tests for /info endpoint"""
    
    def test_info_returns_200(self):
        """Info endpoint should return 200"""
        response = client.get("/info")
        assert response.status_code == 200
    
    def test_info_has_required_fields(self):
        """Info should have service, version, endpoints"""
        response = client.get("/info")
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data
        assert "capabilities" in data
    
    print("✓ Info endpoint tests ready")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    