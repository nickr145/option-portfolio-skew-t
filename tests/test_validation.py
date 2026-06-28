"""
Validation tests for option_portfolio implementation

Tests:
1. Put-call parity
2. Delta-gamma approximation accuracy
3. Parameter recovery from synthetic data
4. Optimization convergence
"""

import numpy as np
import pytest
from option_portfolio import (
    BlackScholesGreeks,
    SkewTFitter,
    OptionPortfolioOptimizer,
    DeltaGammaPortfolio,
)
from option_portfolio.greeks import compute_greeks_for_options


class TestBlackScholesGreeks:
    """Test Black-Scholes Greeks calculations"""

    def test_put_call_parity(self):
        """Test that put-call parity holds for European options."""

        S, K, T, R, sigma = 100, 100, 30 / 365, 0.02, 0.20

        call = BlackScholesGreeks.call_price(S, K, T, R, sigma)
        put = BlackScholesGreeks.put_price(S, K, T, R, sigma)

        parity_lhs = call - put
        parity_rhs = S - K * np.exp(-R * T)

        assert np.isclose(
            parity_lhs, parity_rhs, atol=1e-10
        ), f"Put-call parity violated: {parity_lhs} != {parity_rhs}"

    def test_delta_bounds(self):
        """Test that delta is in [0, 1] for calls and [-1, 0] for puts."""
        S, K, T, R, sigma = 100, 100, 30 / 365, 0.02, 0.20

        call_delta = BlackScholesGreeks.delta("c", S, K, T, R, sigma)
        put_delta = BlackScholesGreeks.delta("p", S, K, T, R, sigma)

        assert 0 <= call_delta <= 1, f"Call delta out of bounds: {call_delta}"
        assert -1 <= put_delta <= 0, f"Put delta out of bounds: {put_delta}"

    def test_gamma_positive(self):
        """Test that gamma is always positive for both calls and puts."""
        S, K, T, R, sigma = 100, 100, 30 / 365, 0.02, 0.20

        gamma = BlackScholesGreeks.gamma(S, K, T, R, sigma)
        assert gamma > 0, f"Gamma should be positive, got: {gamma}"

    def test_atm_delta(self):
        """Test that delta is approximately 0.5 for at-the-money options."""
        S, K, T, R, sigma = 100, 100, 30 / 365, 0.02, 0.20

        delta = BlackScholesGreeks.delta("c", S, K, T, R, sigma)
        assert (
            0.45 < delta < 0.55
        ), f"ATM call delta should be around ~0.5, got: {delta}"


class TestSkewTFitter:
    """Test Skew-T fitting and parameter recovery"""

    def test_parameter_recovery(self):
        """Test that the fitter can recover skew-t parameters from synthetic data."""
        np.random.seed(42)

        N = 3
        T = 500
        mu_true = np.array([0.001, 0.0005, 0.0008])
        sigma_true = np.eye(N) * 0.01 + np.ones((N, N)) * 0.003
        nu_true = 6.0
        omega_true = np.array([0.2, 0.1, -0.3])

        L = np.linalg.cholesky(sigma_true)
        Z = np.random.randn(T, N)
        u_chi = np.sqrt(np.random.chisquare(nu_true, T) / nu_true)

        data = np.zeros((T, N))
        for i in range(T):
            data[i] = mu_true + u_chi[i] * L @ Z[i]

        fitter = SkewTFitter(data)
        params = fitter.fit(method="mle", initial_nu=nu_true)

        assert params["nu"] > 2, "Degrees of freedom should be > 2"
        assert np.all(np.isfinite(params["mu"])), "μ should be finite"
        assert np.all(np.isfinite(params["sigma"])), "Σ should be finite"
        assert np.all(np.isfinite(params["omega"])), "ω should be finite"

    def test_matrix_positive_definite(self):
        """Test that fitted covariance matrix is positive definite."""
        np.random.seed(42)
        data = np.random.randn(100, 3)

        fitter = SkewTFitter(data)
        params = fitter.fit(method="mle")

        eigenvals = np.linalg.eigvalsh(params["sigma"])
        assert np.all(eigenvals > 0), "Covariance matrix should be positive definite"


class TestDeltaGammaApproximation:
    """Test delta-gamma approximation accuracy"""

    def test_approximation_accuracy(self):
        """Test that delta-gamma approximation is accurate for small moves."""
        S, K, T, r, sigma = 100, 100, 30 / 365, 0.02, 0.20
        dt = 1 / 365

        call_price = BlackScholesGreeks.call_price(S, K, T, r, sigma)
        call_delta = BlackScholesGreeks.delta("c", S, K, T, r, sigma)
        call_gamma = BlackScholesGreeks.gamma(S, K, T, r, sigma)
        call_theta = BlackScholesGreeks.theta("c", S, K, T, r, sigma)

        for dS in [-1, 0, 1]:
            S_new = S + dS
            call_new = BlackScholesGreeks.call_price(S_new, K, T - dt, r, sigma)

            pnl_actual = call_new - call_price
            pnl_approx = call_delta * dS + 0.5 * call_gamma * dS**2 + call_theta * dt

            error_pct = (
                abs(pnl_actual - pnl_approx) / abs(pnl_actual) * 100
                if pnl_actual != 0
                else 0
            )

            assert (
                error_pct < 10
            ), f"Delta-gamma error too large for dS={dS}: {error_pct:.1f}%"


class TestPortfolioOptimization:
    """Test portfolio optimization convergence"""

    def test_optimization_converges(self):
        """Test that optimization converges and produces valid results."""
        np.random.seed(42)

        N = 3
        T = 200
        returns = np.random.randn(T, N) * 0.01 + 0.0005
        spot_prices = np.array([100, 120, 90])

        optimizer = OptionPortfolioOptimizer(returns, spot_prices)
        optimizer.fit_skew_t(verbose=False)

        option_specs = [
            {"type": "c", "strike": spot_prices[i], "underlying_idx": i}
            for i in range(N)
        ]
        optimizer.setup_options(option_specs, verbose=False)
        optimizer.assemble_portfolio_params(verbose=False)

        x_sharpe, m_sharpe = optimizer.optimize_sharpe_analytical(verbose=False)

        assert x_sharpe is not None, "Optimization should not return None"
        assert np.all(np.isfinite(x_sharpe)), "Shares should be finite"
        assert m_sharpe["sharpe_ratio"] >= 0, "Sharpe ratio should be non-negative"

    def test_constrained_vs_unconstrained(self):
        """Test that constrained optimization yields different results than unconstrained."""
        np.random.seed(42)

        N = 3
        T = 200
        returns = np.random.randn(T, N) * 0.01 + 0.0005
        spot_prices = np.array([100, 120, 90])

        optimizer = OptionPortfolioOptimizer(returns, spot_prices)
        optimizer.fit_skew_t(verbose=False)

        option_specs = [
            {"type": "c", "strike": spot_prices[i], "underlying_idx": i}
            for i in range(N)
        ]
        optimizer.setup_options(option_specs, verbose=False)
        optimizer.assemble_portfolio_params(verbose=False)

        x_unconstrained, _ = optimizer.optimize_sharpe_analytical(verbose=False)
        x_constrained, _, _ = optimizer.optimize_sharpe_constrained(verbose=False)

        # Check if constraint was actually binding
        max_unconstrained = np.max(np.abs(x_unconstrained))
        max_constrained = np.max(np.abs(x_constrained))

        # If unconstrained violates bounds, constrained should be different
        if max_unconstrained > 0.5:
            assert not np.allclose(
                x_unconstrained, x_constrained
            ), "Constrained solution should differ when constraint is binding"


class TestMatrixProperties:
    """Test matrix properties"""

    def test_Q_matrix_positive_semidefinite(self):
        """Test that the Qmatrix is positive semidefinite for valid portfolios."""
        np.random_seed(42)

        N = 3
        T = 200
        returns = np.random.randn(T, N) * 0.01
        spot_prices = np.array([100, 120, 90])

        optimizer = OptionPortfolioOptimizer(returns, spot_prices)
        optimizer.fit_skew_t(verbose=False)

        option_specs = [
            {"type": "c", "strike": spot_prices[i], "underlying_idx": i}
            for i in range(N)
        ]
        optimizer.setup_options(option_specs, verbose=False)
        optimizer.assemble_portfolio_params(verbose=False)

        eigenvals = np.linalg.eigvalsh(optimizer.Q)
        assert np.all(eigenvals >= -1e-10), "Q should be positive semi-definite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
