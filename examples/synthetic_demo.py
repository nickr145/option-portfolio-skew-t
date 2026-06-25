"""
Step-by-step tutorial and validation script demonstrating the full
replication workflow.

Time to run: ~2-3 minutes
"""

import numpy as np
import matplotlib.pyplot as plt

from option_portfolio import SkewTFitter, BlackScholesGreeks, OptionPortfolioOptimizer
from scipy.special import gamma as gamma_fn


def section(title):
    """Print formatted section header."""
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def demo_1_skew_t_fitting():
    """Demo 1: Fit skew-t distribution to returns."""
    section("DEMO 1: Skew-t Distribution Fitting")

    print("\nGenerating synthetic returns data with known skew-t parameters...")
    np.random.seed(42)
    N = 3
    T = 500

    mu_true = np.array([0.0005, 0.0003, 0.0007])
    sigma_true = np.array(
        [[0.010, 0.003, 0.002], [0.003, 0.012, 0.004], [0.002, 0.004, 0.011]]
    )
    nu_true = 5.5
    omega_true = np.array([0.2, 0.1, -0.3])

    print(f"\nTrue Parameters:")
    print(f"  μ: {mu_true}")
    print(f"  ν (degrees of freedom): {nu_true}")
    print(f"  ω (skewness): {omega_true}")

    print(f"\nGenerating {T} samples...")
    L = np.linalg(cholesky(sigma_true))
    Z = np.random.randn(T, N)
    u_chi = np.sqrt(np.random.chisquare(nu_true, T) / nu_true)

    data = np.zeros((T, N))
    for i in range(T):
        data[i] = mu_true + u_chi[i] * L @ Z[i]

    print(f"Data shape: {data.shape}")
    print(f"Sample mean: {data.mean(axis=0)}")
    print(f"Sample cov diag: {np.diag(np.cov(data.T))}")

    print("\nFitting skew-t distribution...")
    fitter = SkewTFitter(data)
    params = fitter.fit(method="mle", initial_nu=nu_true)

    print("\n" + "-" * 75)
    print("Parameter Recovery:")
    print("-" * 75)
    print(f"μ true:    {mu_true}")
    print(f"μ fitted:  {params['mu']}")
    print(f"Error:     {np.linalg.norm(params['mu'] - mu_true):.6f}")
    print()
    print(f"ν true:    {nu_true:.3f}")
    print(f"ν fitted:  {params['nu']:.3f}")
    print(f"Error:     {abs(params['nu'] - nu_true):.3f}")
    print()
    print(f"ω true:    {omega_true}")
    print(f"ω fitted:  {params['omega']}")
    print(f"Error:     {np.linalg.norm(params['omega'] - omega_true):.6f}")

    return fitter, params


def demo_2_black_scholes_greeks():
    """Demo 2: Compute Black-Scholes Greeks."""
    section("DEMO 2: Black-Scholes Greeks Computation")

    print("\nComputing Greeks for a single option...")
    print("-" * 75)

    S = 100.0
    K = 100.0
    T = 30 / 365
    r = 0.02
    sigma = 0.20

    print(f"Spot:           ${S:.2f}")
    print(f"Strike:         ${K:.2f} (ATM)")
    print(f"Time to expiry: {T*365:.0f} days ({T:.4f} years)")
    print(f"Risk-free rate: {r:.1%}")
    print(f"Volatility:     {sigma:.1%}")
    print()

    call_price = BlackScholesGreeks.call_price(S, K, T, r, sigma)
    call_delta = BlackScholesGreeks.delta("c", S, K, T, r, sigma)
    call_gamma = BlackScholesGreeks.gamma(S, K, T, r, sigma)
    call_theta = BlackScholesGreeks.theta("c", S, K, T, r, sigma)
    call_vega = BlackScholesGreeks.vega(S, K, T, r, sigma)

    print("CALL OPTION:")
    print(f"  Price:           ${call_price:.4f}")
    print(f"  Delta (∂V/∂S):   {call_delta:.4f}")
    print(f"  Gamma (∂²V/∂S²): {call_gamma:.6f} per $1")
    print(f"  Theta (∂V/∂t):   ${call_theta:.6f} per day")
    print(f"  Vega (∂V/∂σ):    ${call_vega:.4f} per 1% vol")

    put_price = BlackScholesGreeks.put_price(S, K, T, r, sigma)
    put_delta = BlackScholesGreeks.delta("p", S, K, T, r, sigma)

    print("\nPUT OPTION:")
    print(f"  Price:           ${put_price:.4f}")
    print(f"  Delta (∂V/∂S):   {put_delta:.4f}")

    print("\n" + "-" * 75)
    print("Put-Call Parity Check (should equal S - K*exp(-rT)):")

    parity_lhs = call_price - put_price
    parity_rhs = S - K * np.exp(-r * T)

    print(f"  Call - Put:      {parity_lhs:.6f}")
    print(f"  S - K*exp(-rT):  {parity_rhs:.6f}")
    print(f"  Error:           {abs(parity_lhs - parity_rhs):.10f}")


def demo_3_delta_gamma_approximation():
    """Demo 3: Delta-gamma approximation for portfolio P&L."""
    section("DEMO 3: Delta-Gamma Approximation")

    print("\nDelta-gamma approximation writes portfolio P&L as:")
    print("  ΔV = θ·Δt + δ'·ΔS + (1/2)·(ΔS)'·Γ·(ΔS)")
    print("\nwhere:")
    print("  θ = theta (time decay)")
    print("  δ = delta vector (first-order sensitivity)")
    print("  Γ = gamma matrix (second-order sensitivity)")
    print("\nThis captures the nonlinearity of option payoffs without simulation.")

    print("\n" + "-" * 75)
    print("Example: Long call option")
    print("-" * 75)

    S, K, T, r, sigma = 100.0, 100.0, 30 / 365, 0.02, 0.20

    call_price = BlackScholesGreeks.call_price(S, K, T, r, sigma)
    call_delta = BlackScholesGreeks.delta("c", S, K, T, r, sigma)
    call_gamma = BlackScholesGreeks.gamma(S, K, T, r, sigma)
    call_theta = BlackScholesGreeks.theta("c", S, K, T, r, sigma)

    print(f"\nCurrent option value: ${call_price:.4f}")
    print(f"Delta: {call_delta:.4f}")
    print(f"Gamma: {call_gamma:.6f}")
    print(f"Theta (daily): ${call_theta:.6f}")

    print("\n" + "-" * 75)
    print("P&L under different spot price changes (next day):")
    print("-" * 75)

    dt = 1 / 365
    for dS in [-2, -1, 0, 1, 2]:
        S_new = S + dS
        call_new = BlackScholesGreeks.call_price(S_new, K, T - dt, r, sigma)
        pnl_actual = call_new - call_price
        pnl_approx = call_delta * dS + 0.5 * call_gamma * dS**2 + call_theta * dt

        error = abs(pnl_actual - pnl_approx)

        print(f"\n  ΔS = ${dS:+.1f}:")
        print(f"    Actual P&L:     ${pnl_actual:+.4f}")
        print(f"    Approx P&L:     ${pnl_approx:+.4f}")
        print(f"    Error:          ${error:.4f} ({error/abs(pnl_actual)*100:.1f}%)")


def demo_4_full_portfolio_optimization():
    """Demo 4: Full portfolio optimization workflow."""
    section("DEMO 4: Full Portfolio Optimization Workflow")

    print("\nCreating synthetic returns data...")

    np.random.seed(42)
    N = 5
    T = 500

    mu = np.array([0.0005, 0.0003, 0.0007, 0.0004, 0.0006])
    sigma = np.eye(N) * 0.01 + np.ones((N, N)) * 0.003
    returns = np.random.multivariate_normal(mu, sigma, T)
    spot_prices = np.array([100, 120, 90, 110, 105])

    print(f"Generated {T} observations on {N} assets")
    print(f"Spot prices: {spot_prices}")

    optimizer = OptionPortfolioOptimizer(
        returns, spot_prices, asset_names=[f"Stock_{i+1}" for i in range(N)]
    )

    print("\n" + "-" * 75)
    print("Step 1: Fitting skew-t distribution...")
    print("-" * 75)
    optimizer.fit_skew_t(verbose=False)
    print(f"✓ Fitted parameters:")
    print(f"  ν = {optimizer.skew_t_params['nu']:.3f}")
    print(f"  ||ω|| = {np.linalg.norm(optimizer.skew_t_params['omega']):.4f}")

    # Setup options
    print("\n" + "-" * 75)
    print("Step 2: Setting up option portfolio...")
    print("-" * 75)

    option_specs = [
        {"type": "c", "strike": spot_prices[i], "underlying_idx": i}
        for i in range(len(spot_prices))
    ]
    optimizer.setup_options(
        option_specs, T=30 / 365, r=0.02, sigma_surface=0.25, verbose=False
    )

    print(f"✓ Created {optimizer.M} call options (ATM)")
    print(f"  Option values: {optimizer.v}")

    print("\n" + "-" * 75)
    print("Step 3: Computing portfolio parameters (u and Q)...")
    print("-" * 75)
    optimizer.assemble_portfolio_params(verbose=False)
    print(f"✓ Assembled u and Q from Greeks + distributional parameters")
    print(f"  Condition number of Q: {np.linalg.cond(optimizer.Q):.2e}")

    print("\n" + "-" * 75)
    print("Step 4: Portfolio optimization...")
    print("-" * 75)

    x_sharpe, m_sharpe = optimizer.optimize_sharpe_analytical(verbose=False)
    print(f"\n✓ Sharpe ratio optimization:")
    print(f"  Sharpe ratio: {m_sharpe['sharpe_ratio']:.4f}")
    print(f"  Expected return: {m_sharpe['exp_return']:.4f}")
    print(f"  Volatility: {m_sharpe['volatility']:.4f}")

    x_rvar, m_rvar, lam = optimizer.optimize_rvar_analytical(verbose=False)
    print(f"\n✓ Return-to-VaR optimization:")
    print(f"  λ*: {lam:.4f}")
    print(f"  Sharpe ratio: {m_rvar['sharpe_ratio']:.4f}")
    print(f"  R-VaR ratio: {m_rvar['r_var_ratio']:.4f}")

    print("\n" + "-" * 75)
    print("Step 5: Constrained optimization (box: ±0.5)...")
    print("-" * 75)

    x_sharpe_c, m_sharpe_c, _ = optimizer.optimize_sharpe_constrained(verbose=False)
    x_rvar_c, m_rvar_c, _ = optimizer.optimize_rvar_constrained(verbose=False)

    diff_unconstrained = np.linalg.norm(x_sharpe - x_rvar, ord=1)
    diff_constrained = np.linalg.norm(x_sharpe_c - x_rvar_c, ord=1)

    print(f"\n✓ Comparison:")
    print(f"  Unconstrained Sharpe vs R-VaR L1 distance: {diff_unconstrained:.4f}")
    print(f"  Constrained L1 distance: {diff_constrained:.4f}")

    if diff_constrained < diff_unconstrained:
        print(
            f"  → Convergence confirmed! (ratio: {diff_constrained/diff_unconstrained:.1%})"
        )


if __name__ == "__main__":
    print("\n" + "█" * 75)
    print("█" + " " * 73 + "█")
    print("█" + "  Sung & Pirvu (2026) Replication: Complete Tutorial".center(73) + "█")
    print("█" + "  Option Portfolio Optimization under Skew-t Returns".center(73) + "█")
    print("█" + " " * 73 + "█")
    print("█" * 75)

    demo_1_skew_t_fitting()
    demo_2_black_scholes_greeks()
    demo_3_delta_gamma_approximation()
    demo_4_full_portfolio_optimization()

    print("\n" + "=" * 75)
    print("  Tutorial Complete!")
    print("=" * 75)
    print("\nNext steps:")
    print("  1. Explore the code in optimizer.py")
    print("  2. Run: python -m option_portfolio")
    print("  3. Modify the optimization objectives")
    print("  4. Try with different options (puts, different strikes)")
    print("\nSee README.md for detailed documentation.")
    print("=" * 75 + "\n")
