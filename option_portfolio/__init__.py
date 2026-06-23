"""
Option Portfolio Optimization under Skew-Elliptical t Returns

Implementation of Sung & Pirvu (2026) "Sharpe Ratio and Return-VaR Ratio
Maximization for Option Portfolios with Skew-Elliptical t Underlying Returns"

Main classes:
    - OptionPortfolioOptimizer: End-to-end optimization pipeline
    - SkewTFitter: Fit multivariate skew-t distribution
    - BlackScholesGreeks: Compute option Greeks
    - DeltaGammaPortfolio: Assemble delta-gamma parameters
"""

from .optimizer import OptionPortfolioOptimizer
from .distributions import SkewTFitter
from .greeks import BlackScholesGreeks, DeltaGammaPortfolio

__version__ = "1.0.0"
__author__ = "Nicholas Rebello"
__all__ = [
    "OptionPortfolioOptimizer",
    "SkewTFitter",
    "BlackScholesGreeks",
    "DeltaGammaPortfolio",
]
