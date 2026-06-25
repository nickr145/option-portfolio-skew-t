"""
Real data example using yfinance

Downloads real stock returns and runs the full optimization pipeline.
"""

import numpy as np
import pandas as pd
from option_portfolio import OptionPortfolioOptimizer


def main():
    """Run full optimization on real market data."""

    print("\n" + "=" * 80)
    print("REAL DATA EXAMPLE: Sung & Pirvu (2026) Implementation")
    print("=" * 80)

    # Download real data
    print("\nDownloading historical stock data from Yahoo Finance...")
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Install with: pip install yfinance")
        return

    tickers = ["DIS", "XOM", "PFE", "MO", "INTC"]
    print(f"Tickers: {tickers}")

    data = yf.download(tickers, start="2023-01-01", end="2024-12-31", progress=False)[
        "Adj Close"
    ]
    returns = data.pct_change().dropna()

    print(f"\nData downloaded:")
    print(f"  Shape: {returns.shape}")
    print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")

    spot_prices = data.iloc[-1].values
    print(f"  Current spot prices: {spot_prices}")

    # Initialize optimizer
    print("\n" + "-" * 80)
    print("Initializing portfolio optimizer...")
    print("-" * 80)

    optimizer = OptionPortfolioOptimizer(
        returns.values, spot_prices, asset_names=tickers
    )

    # Step 1: Fit skew-t
    print("\nSTEP 1: Fitting Skew-t Distribution")
    print("-" * 80)
    optimizer.fit_skew_t(verbose=True)

    # Step 2: Setup options (ATM calls on all 5 stocks)
    print("\nSTEP 2: Setting up Option Portfolio")
    print("-" * 80)
    option_specs = [
        {"type": "c", "strike": spot_prices[i], "underlying_idx": i}
        for i in range(len(tickers))
    ]
    optimizer.setup_options(
        option_specs,
        T=30 / 365,  # 30 days
        r=0.02,  # 2% risk-free rate
        sigma_surface=0.25,  # 25% implied volatility
        verbose=True,
    )

    # Step 3: Assemble portfolio parameters
    print("\nSTEP 3: Assembling Portfolio Parameters")
    print("-" * 80)
    u, Q = optimizer.assemble_portfolio_params(verbose=True)

    # Step 4: Run all optimizations
    print("\nSTEP 4: Running Optimizations")
    print("-" * 80)

    # Unconstrained Sharpe
    x_sharpe, m_sharpe = optimizer.optimize_sharpe_analytical(verbose=True)

    # Unconstrained R-VaR
    x_rvar, m_rvar, lam = optimizer.optimize_rvar_analytical(alpha=0.01, verbose=True)

    # Constrained Sharpe
    x_sharpe_c, m_sharpe_c, _ = optimizer.optimize_sharpe_constrained(
        box_bounds=(-0.5, 0.5), verbose=True
    )

    # Constrained R-VaR
    x_rvar_c, m_rvar_c, _ = optimizer.optimize_rvar_constrained(
        alpha=0.01, box_bounds=(-0.5, 0.5), verbose=True
    )

    # Step 5: Compare results
    print("\nSTEP 5: Comparison of Results")
    print("-" * 80)
    optimizer.compare_results(x_sharpe, x_rvar, x_sharpe_c, x_rvar_c)

    # Visualize
    print("\nGenerating visualizations...")
    results = {
        "sharpe_unconstrained": (x_sharpe, m_sharpe),
        "rvar_unconstrained": (x_rvar, m_rvar),
        "sharpe_constrained": (x_sharpe_c, m_sharpe_c),
        "rvar_constrained": (x_rvar_c, m_rvar_c),
    }

    fig = optimizer.plot_comparison(results)
    fig.savefig("real_data_results.png", dpi=150, bbox_inches="tight")
    print("✓ Saved visualization to real_data_results.png")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print("\nSharpe Ratio Optimization (Unconstrained):")
    print(f"  Sharpe ratio: {m_sharpe['sharpe_ratio']:.6f}")
    print(f"  Expected return: {m_sharpe['exp_return']:.6f}")
    print(f"  Volatility: {m_sharpe['volatility']:.6f}")
    print(f"  Weights: {m_sharpe['weights']}")

    print("\nReturn-to-VaR Optimization (Unconstrained):")
    print(f"  R-VaR ratio: {m_rvar['r_var_ratio']:.6f}")
    print(f"  Expected return: {m_rvar['exp_return']:.6f}")
    print(f"  Volatility: {m_rvar['volatility']:.6f}")
    print(f"  Lambda*: {lam:.6f}")
    print(f"  Weights: {m_rvar['weights']}")

    print("\nSharpe Ratio Optimization (Constrained ±0.5):")
    print(f"  Sharpe ratio: {m_sharpe_c['sharpe_ratio']:.6f}")
    print(f"  Expected return: {m_sharpe_c['exp_return']:.6f}")
    print(f"  Volatility: {m_sharpe_c['volatility']:.6f}")
    print(f"  Weights: {m_sharpe_c['weights']}")

    print("\nReturn-to-VaR Optimization (Constrained ±0.5):")
    print(f"  R-VaR ratio: {m_rvar_c['r_var_ratio']:.6f}")
    print(f"  Expected return: {m_rvar_c['exp_return']:.6f}")
    print(f"  Volatility: {m_rvar_c['volatility']:.6f}")
    print(f"  Weights: {m_rvar_c['weights']}")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
