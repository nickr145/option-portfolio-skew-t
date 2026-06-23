"""
Option Greeks and Delta-Gamma Approximation

Computes:
1. Black-Scholes Greeks (delta, gamma, theta, vega)
2. u vector (expected P&L) from Appendix A
3. Q matrix (variance-covariance of P&L) from Appendix A
"""

import numpy as np
from scipy.stats import norm
from scipy.special import gamma as gamma_fn


class BlackScholesGreeks:
    """Compute Black-Scholes Greeks for European options."""

    @staticmethod
    def d1(S, K, T, r, sigma):
        """Compute d1 for Black-Scholes formula."""
        if T <= 0 or sigma <= 0:
            return np.nan
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S, K, T, r, sigma):
        """Compute d2 for Black-Scholes formula."""
        return BlackScholesGreeks.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S, K, T, r, sigma):
        """Compute call option price using Black-Scholes formula."""
        if T <= 0:
            return max(S - K, 0)
        d1 = BlackScholesGreeks.d1(S, K, T, r, sigma)
        d2 = BlackScholesGreeks.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S, K, T, r, sigma):
        """Compute put option price using Black-Scholes formula."""
        if T <= 0:
            return max(K - S, 0)
        d1 = BlackScholesGreeks.d1(S, K, T, r, sigma)
        d2 = BlackScholesGreeks.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def delta(option_type, S, K, T, r, sigma):
        """Compute delta of an option."""
        if T <= 0 or sigma <= 0:
            return np.nan

        d1 = BlackScholesGreeks.d1(S, K, T, r, sigma)
        if option_type.lower() == "c":
            return norm.cdf(d1)
        else:
            return -norm.cdf(-d1)

    @staticmethod
    def gamma(S, K, T, r, sigma):
        """Compute gamma of an option."""
        if T <= 0 or sigma <= 0:
            return np.nan

        d1 = BlackScholesGreeks.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def theta(option_type, S, K, T, r, sigma):
        """Compute theta of an option. Expressed as daily theta"""
        if T <= 0 or sigma <= 0:
            return np.nan

        d1 = BlackScholesGreeks.d1(S, K, T, r, sigma)
        d2 = BlackScholesGreeks.d2(S, K, T, r, sigma)

        term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

        if option_type.lower() == "c":
            term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
            theta_val = (term1 - term2) / 365
        else:
            term2 = -r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta_val = (term1 + term2) / 365

        return theta_val

    @staticmethod
    def vega(S, K, T, r, sigma):
        """Compute vega of an option. Expressed per 1% change in volatility."""
        if T <= 0 or sigma <= 0:
            return np.nan

        d1 = BlackScholesGreeks.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100


class DeltaGammaPortfolio:
    """
    Compute delta-gamma approximation for option portfolio P&L.

    Portfolio P&L: ΔV = θ·Δt + δ'·ΔS + (1/2)·(ΔS)'·Γ·(ΔS)
    """

    def __init__(self, greeks, spot_prices, skew_t_params, dt=1 / 252, r=0.02):
        """
        Args:
            greeks (dict): Greeks data with 'delta', 'gamma', 'theta'
            spot_prices (np.ndarray): Current spot prices (N,)
            skew_t_params (dict): Fitted skew-t parameters (mu, sigma, nu, omega)
            dt (float): Time step (default: 1/252 for daily)
            r (float): Risk-free rate (annual)
        """
        self.delta = greeks["delta"]
        self.gamma = greeks["gamma"]
        self.theta = greeks["theta"]

        self.S = spot_prices
        self.M, self.N = self.delta.shape

        self.mu = skew_t_params["mu"]
        self.sigma = skew_t_params["sigma"]
        self.nu = skew_t_params["nu"]
        self.omega = skew_t_params["omega"]

        self.dt = dt
        self.r = r

        self._precompute_skew_t_constants()

    def _precompute_skew_t_constants(self):
        """Precompute constants from skew-t parametrization."""
        self.c = (
            np.sqrt(self.nu / np.pi)
            * gamma_fn((self.nu - 1) / 2)
            / gamma_fn(self.nu / 2)
        )

        omega_sigma_omega = self.omega @ self.sigma @ self.omega
        self.h = (self.sigma @ self.omega) / np.sqrt(1 + omega_sigma_omega)

    def compute_u_vector(self):
        """
        Compute expected P&L vector u from Appendix A.

        Returns:
            u (np.ndarray): (M,) expected P&L for each option
        """
        u = np.zeros(self.M)

        time_decay = self.dt * self.theta
        delta_mu = self.delta @ self.mu

        drift_adj = (self.nu / (2 * (self.nu - 2))) * np.ones(self.M)
        if self.nu > 2:
            drift_adj *= self.c

        gamma_diag_contrib = np.array(
            [0.5 * np.trace(self.gamma[m] @ self.sigma) for m in range(self.M)]
        )

        u = time_decay + delta_mu + drift_adj + gamma_diag_contrib

        skew_mod = np.zeros(self.M)
        for m in range(self.M):
            h_gamma_h = self.h @ self.gamma[m] @ self.h
            skew_mod[m] = self.c * h_gamma_h

        u += skew_mod

        return u

    def compute_Q_matrix(self):
        """
        Compute variance-covariance matrix Q from Appendix A.

        Returns:
            Q (np.ndarray): (M, M) variance-covariance matrix
        """

        U = np.zeros((self.M, self.M))

        for m in range(self.M):
            for n in range(self.M):
                gamma_sigma_term = np.trace(
                    self.gamma[m] @ self.sigma @ self.gamma[n] @ self.sigma
                )

                if self.nu > 4:
                    factor = self.nu / (2 * (self.nu - 2) * (self.nu - 4))
                else:
                    factor = 1.0

                U[m, n] = (
                    (2 * self.nu / (self.nu - 2))
                    * np.trace(self.gamma[m] @ self.sigma)
                    * np.trace(self.gamma[n] @ self.sigma)
                )
                U[m, n] += factor * gamma_sigma_term

        H = np.zeros((self.M, self.M))
        for m in range(self.M):
            for n in range(self.M):
                H[m, n] = self.mu @ self.gamma[m] @ self.sigma @ self.gamma[n] @ self.h

        E = np.zeros((self.M, self.M))
        for m in range(self.M):
            for n in range(self.M):
                E[m, n] = self.delta[m] @ self.sigma @ self.gamma[n] @ self.h

        R = np.zeros((self.M, self.M))
        for m in range(self.M):
            for n in range(self.M):
                R[m, n] = np.trace(
                    self.gamma[m] @ self.sigma @ self.gamma[n] @ self.sigma
                )

        Q_tilde = U.copy()
        Q_tilde += (4 * self.c * self.nu / (self.nu - 3)) * (H + E)

        if self.nu > 4:
            Q_tilde += (self.nu / (2 * (self.nu - 2) * (self.nu - 4))) * R

        Q = 0.5 * (Q_tilde + Q_tilde.T)

        return Q


def compute_greeks_for_options(option_specs, spot_prices, r, T, sigma_surface=None):
    """
    Compute Greeks for a portfolio of options.

    Args:
        option_specs (list): List of dicts with 'type', 'strike', 'underlying_idx'
        spot_prices (np.ndarray): (N,) current spot prices
        r (float): Risk-free rate (annual)
        T (float): Time to expiry (years)
        sigma_surface (callable or float): Implied vol

    Returns:
        dict: Greeks data with 'delta', 'gamma', 'theta', 'vega', 'prices'
    """

    M = len(option_specs)
    N = len(spot_prices)

    delta = np.zeros((M, N))
    gamma = np.zeros((M, N, N))
    theta = np.zeros(M)
    vega = np.zeros(M)
    prices = np.zeros(M)

    for m, spec in enumerate(option_specs):
        opt_type = spec["type"]
        K = spec["strike"]
        idx = spec["underlying_idx"]
        S = spot_prices[idx]

        if callable(sigma_surface):
            sigma = sigma_surface(S, K, T)
        else:
            sigma = sigma_surface if sigma_surface else 0.2

        delta[m, idx] = BlackScholesGreeks.delta(opt_type, S, K, T, r, sigma)
        gamma[m, idx, idx] = BlackScholesGreeks.gamma(S, K, T, r, sigma)
        theta[m] = BlackScholesGreeks.theta(opt_type, S, K, T, r, sigma)
        vega[m] = BlackScholesGreeks.vega(S, K, T, r, sigma)

        if opt_type.lower() == "c":
            prices[m] = BlackScholesGreeks.call_price(S, K, T, r, sigma)
        else:
            prices[m] = BlackScholesGreeks.put_price(S, K, T, r, sigma)

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "prices": prices,
    }
