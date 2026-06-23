"""
Complete Option Portfolio Optimization System
Full replication of Sung & Pirvu (2026)

Workflow:
1. Fit skew-t distribution to stock returns
2. Compute option Greeks via Black-Scholes
3. Assemble u (expected P&L) and Q (variance) from Appendix A
4. Optimize Sharpe ratio and Return-to-VaR ratio
5. Compare results with/without constraints
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from .distributions import SkewTFitter
from .greeks import compute_greeks_for_options, DeltaGammaPortfolio


class OptionPortfolioOptimizer:
    """Complete option portfolio optimization under skew-t returns."""

    def __init__(self, returns_data, spot_prices, asset_names=None):
        """
        Args:
            returns_data (np.ndarray): (T, N) returns data
            spot_prices (np.ndarray): (N,) current spot prices
            asset_names (list): Asset names
        """
        self.returns_data = returns_data
        self.spot_prices = spot_prices
        self.N = len(spot_prices)
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.N)]

        self.skew_t_fitter = None
        self.skew_t_params = None
        self.greeks = None
        self.delta_gamma_port = None
        self.u = None
        self.Q = None
        self.v = None

    def fit_skew_t(self, verbose=True):
        """Fit skew-t distribution to returns."""
        if verbose:
            print("\n" + "=" * 70)
            print("STEP 1: Fitting Skew-t Distribution")
            print("=" * 70)

        self.skew_t_fitter = SkewTFitter(self.returns_data)
        self.skew_t_params = self.skew_t_fitter.fit(method="mle")

        if verbose:
            self.skew_t_fitter.summary()

        return self.skew_t_params

    def setup_options(
        self, option_specs, T=30 / 365, r=0.02, sigma_surface=None, verbose=True
    ):
        """Set up option portfolio."""
        if verbose:
            print("\n" + "=" * 70)
            print("STEP 2: Computing Option Greeks")
            print("=" * 70)

        self.r = r
        self.T = T

        self.greeks = compute_greeks_for_options(
            option_specs, self.spot_prices, r, T, sigma_surface
        )

        self.v = self.greeks["prices"]
        self.M = len(self.v)

        if verbose:
            print(f"\nOption Portfolio:")
            print(f"  Number of options: {self.M}")
            print(f"  Option values: {self.v}")
            print(f"  Time to expiry: {T*365:.0f} days")
            print(f"  Risk-free rate: {r}")

        return self.greeks

    def assemble_portfolio_params(self, dt=1 / 252, verbose=True):
        """Compute u and Q from Appendix A."""
        if verbose:
            print("\n" + "=" * 70)
            print("STEP 3: Assembling Delta-Gamma Portfolio Parameters (Appendix A)")
            print("=" * 70)

        self.delta_gamma_port = DeltaGammaPortfolio(
            self.greeks, self.spot_prices, self.skew_t_params, dt=dt, r=self.r
        )

        self.u = self.delta_gamma_port.compute_u_vector()
        self.Q = self.delta_gamma_port.compute_Q_matrix()

        if verbose:
            print(f"\nExpected P&L vector (u):")
            print(self.u)
            print(f"\nVariance-Covariance matrix (Q):")
            print(self.Q)
            print(f"\nCondition number of Q: {np.linalg.cond(self.Q):.2e}")

        return self.u, self.Q

    def optimize_sharpe_analytical(self, verbose=True):
        """Theorem 3.1: Analytical Sharpe ratio maximization."""
        if verbose:
            print("\n" + "=" * 70)
            print("STEP 4a: Analytical Sharpe Ratio Optimization")
            print("=" * 70)

        try:
            Q_inv = np.linalg.inv(self.Q)
            numerator = self.u - self.r * self.v

            x_tilde = Q_inv @ numerator
            denominator = self.v @ x_tilde

            if abs(denominator) < 1e-10:
                raise ValueError("Denominator too small (numerical issues)")

            x_star = x_tilde / denominator

            metrics = self.portfolio_metrics(x_star)

            if verbose:
                print(f"\nOptimal portfolio (unconstrained):")
                print(f"  Shares: {x_star}")
                print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
                print(f"  R-VaR ratio: {metrics['r_var_ratio']:.4f}")
                print(f"  Expected return: {metrics['exp_return']:.4f}")
                print(f"  Volatility: {metrics['volatility']:.4f}")

            return x_star, metrics

        except Exception as e:
            print(f"Error in analytical optimization: {e}")
            return None, None

    def optimize_rvar_analytical(self, alpha=0.01, verbose=True):
        """Theorem 3.2: Analytical Return-to-VaR maximization."""
        if verbose:
            print("\n" + "=" * 70)
            print("STEP 4b: Analytical Return-to-VaR Optimization")
            print("=" * 70)

        z_alpha = norm.ppf(alpha)

        def objective_lambda(lam):
            """Objective to maximize (returns negative for minimization)."""
            try:
                Q_inv = np.linalg.inv(self.Q)
                numerator = (1 - lam) * self.u - self.r * self.v

                x_denom = self.v @ Q_inv @ numerator
                if abs(x_denom) < 1e-10:
                    return 1e10

                x = Q_inv @ numerator / x_denom

                exp_ret = self.u @ x - self.r
                var_denom = -self.u @ x - z_alpha * np.sqrt(0.5 * x @ self.Q @ x)

                if abs(var_denom) < 1e-10:
                    return 1e10

                r_var = exp_ret / var_denom
                return -r_var

            except:
                return 1e10

        from scipy.optimize import minimize_scalar

        result = minimize_scalar(objective_lambda, bounds=(0.0, 3.0), method="bounded")

        lambda_star = result.x

        try:
            Q_inv = np.linalg.inv(self.Q)
            numerator = (1 - lambda_star) * self.u - self.r * self.v

            x_tilde = Q_inv @ numerator
            x_star = x_tilde / (self.v @ x_tilde)

            metrics = self.portfolio_metrics(x_star)

            if verbose:
                print(f"\nOptimal portfolio (unconstrained):")
                print(f"  λ*: {lambda_star:.4f}")
                print(f"  Shares: {x_star}")
                print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
                print(f"  R-VaR ratio: {metrics['r_var_ratio']:.4f}")
                print(f"  Expected return: {metrics['exp_return']:.4f}")
                print(f"  Volatility: {metrics['volatility']:.4f}")

            return x_star, metrics, lambda_star

        except Exception as e:
            print(f"Error in analytical optimization: {e}")
            return None, None, None

    def optimize_sharpe_constrained(self, box_bounds=(-0.5, 0.5), verbose=True):
        """Numerical Sharpe ratio optimization with box constraints."""
        if verbose:
            print("\n" + "=" * 70)
            print("STEP 4c: Constrained Sharpe Ratio Optimization")
            print("=" * 70)

        def objective(x):
            exp_ret = self.u @ x - self.r
            std = np.sqrt(0.5 * x @ self.Q @ x)
            if std < 1e-10:
                return 1e10
            return -exp_ret / std

        constraints = {"type": "eq", "fun": lambda x: (self.v @ x) - 1.0}
        bounds = [(box_bounds[0], box_bounds[1])] * self.M
        x0 = np.ones(self.M) / self.M

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9},
        )

        metrics = self.portfolio_metrics(result.x)

        if verbose:
            print(
                f"\nOptimal portfolio (constrained [{box_bounds[0]}, {box_bounds[1]}]):"
            )
            print(f"  Shares: {result.x}")
            print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"  R-VaR ratio: {metrics['r_var_ratio']:.4f}")

        return result.x, metrics, result

    def optimize_rvar_constrained(
        self, alpha=0.01, box_bounds=(-0.5, 0.5), verbose=True
    ):
        """Numerical Return-to-VaR optimization with box constraints."""
        if verbose:
            print("\n" + "=" * 70)
            print("STEP 4d: Constrained Return-to-VaR Optimization")
            print("=" * 70)

        z_alpha = norm.ppf(alpha)

        def objective(x):
            exp_ret = self.u @ x - self.r
            var_denom = -self.u @ x - z_alpha * np.sqrt(0.5 * x @ self.Q @ x)
            if abs(var_denom) < 1e-10:
                return 1e10
            return -exp_ret / var_denom

        constraints = {"type": "eq", "fun": lambda x: (self.v @ x) - 1.0}
        bounds = [(box_bounds[0], box_bounds[1])] * self.M
        x0 = np.ones(self.M) / self.M

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9},
        )

        metrics = self.portfolio_metrics(result.x)

        if verbose:
            print(
                f"\nOptimal portfolio (constrained [{box_bounds[0]}, {box_bounds[1]}]):"
            )
            print(f"  Shares: {result.x}")
            print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"  R-VaR ratio: {metrics['r_var_ratio']:.4f}")

        return result.x, metrics, result

    def portfolio_metrics(self, x, alpha=0.01):
        """Compute key metrics for a given portfolio."""
        z_alpha = norm.ppf(alpha)

        exp_ret = self.u @ x - self.r
        volatility = np.sqrt(0.5 * x @ self.Q @ x)
        sharpe = exp_ret / volatility if volatility > 0 else 0

        var_denom = -self.u @ x - z_alpha * np.sqrt(0.5 * x @ self.Q @ x)
        r_var = exp_ret / var_denom if abs(var_denom) > 1e-10 else 0

        dollar_value = self.v * x
        weights = (
            dollar_value / dollar_value.sum()
            if dollar_value.sum() != 0
            else x / x.sum()
        )

        return {
            "exp_return": exp_ret,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "r_var_ratio": r_var,
            "weights": weights,
            "shares": x,
        }

    def compare_results(self, x_sharpe, x_rvar, x_sharpe_const, x_rvar_const):
        """Compare all optimization results."""
        print("\n" + "=" * 70)
        print("STEP 5: Comparison of Results")
        print("=" * 70)

        if x_sharpe is not None and x_rvar is not None:
            unconstrained_diff = np.linalg.norm(x_sharpe - x_rvar, ord=1)
            print(f"\nUnconstrained portfolios:")
            print(f"  Sharpe vs R-VaR L1 difference: {unconstrained_diff:.4f}")

        if x_sharpe_const is not None and x_rvar_const is not None:
            constrained_diff = np.linalg.norm(x_sharpe_const - x_rvar_const, ord=1)
            print(f"\nConstrained portfolios:")
            print(f"  Sharpe vs R-VaR L1 difference: {constrained_diff:.4f}")

        print("\nConclusion:")
        if unconstrained_diff > constrained_diff:
            print("  ✓ Constrained optimization leads to convergence")
            print("    (Sharpe and R-VaR portfolios become more similar)")
        else:
            print("  ! Portfolios remain distinct even with constraints")

    def plot_comparison(self, results_dict, figsize=(14, 10)):
        """Visualize portfolio comparisons."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Unconstrained weights
        ax = axes[0, 0]
        if (
            "sharpe_unconstrained" in results_dict
            and "rvar_unconstrained" in results_dict
        ):
            x_sharpe = results_dict["sharpe_unconstrained"][0]
            x_rvar = results_dict["rvar_unconstrained"][0]

            x = np.arange(self.M)
            width = 0.35
            ax.bar(x - width / 2, x_sharpe, width, label="Sharpe", alpha=0.8)
            ax.bar(x + width / 2, x_rvar, width, label="R-VaR", alpha=0.8)
            ax.set_ylabel("Portfolio Shares")
            ax.set_title("Unconstrained Optimization")
            ax.legend()
            ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

        # Plot 2: Constrained weights
        ax = axes[0, 1]
        if "sharpe_constrained" in results_dict and "rvar_constrained" in results_dict:
            x_sharpe = results_dict["sharpe_constrained"][0]
            x_rvar = results_dict["rvar_constrained"][0]

            x = np.arange(self.M)
            width = 0.35
            ax.bar(x - width / 2, x_sharpe, width, label="Sharpe", alpha=0.8)
            ax.bar(x + width / 2, x_rvar, width, label="R-VaR", alpha=0.8)
            ax.set_ylabel("Portfolio Shares")
            ax.set_title("Constrained Optimization (±0.5)")
            ax.legend()
            ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

        # Plot 3: Risk-Return Profile
        ax = axes[1, 0]
        methods = []
        returns = []
        vols = []
        colors_list = []

        for key, (x, metrics, *_) in results_dict.items():
            if x is not None:
                methods.append(key)
                returns.append(metrics["exp_return"])
                vols.append(metrics["volatility"])
                if "sharpe" in key:
                    colors_list.append("blue")
                else:
                    colors_list.append("red")

        for i, (ret, vol, label, color) in enumerate(
            zip(returns, vols, methods, colors_list)
        ):
            ax.scatter(vol, ret, s=100, label=label, color=color, alpha=0.6)

        ax.set_xlabel("Volatility")
        ax.set_ylabel("Expected Return")
        ax.set_title("Risk-Return Profile")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 4: Sharpe Ratio Comparison
        ax = axes[1, 1]
        methods = []
        sharpes = []

        for key, (x, metrics, *_) in results_dict.items():
            if x is not None:
                methods.append(key.replace("_", "\n"))
                sharpes.append(metrics["sharpe_ratio"])

        ax.bar(
            range(len(methods)),
            sharpes,
            alpha=0.7,
            color=["blue", "red", "blue", "red"],
        )
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Sharpe Ratio Comparison")
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig
