# Option Portfolio Optimization Under Skew-Elliptical t Returns

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Analytical and numerical methods for optimizing option portfolios when underlying returns follow a skew-elliptical t-distribution.

**Full replication of:** Sung & Pirvu (2026) "Sharpe Ratio and Return-VaR Ratio Maximization for Option Portfolios with Skew-Elliptical t Underlying Returns" (arXiv:2606.17032v1)

## Overview

Traditional portfolio optimization assumes normally distributed returns, which significantly underestimate tail risk. This project implements a complete quantitative finance pipeline for optimizing option portfolios under more realistic market conditions:

- **Heavy tails**: ν=6 means market extremes occur ~100x more frequently than normal assumption
- **Skewness**: Asymmetric return distributions (left tail heavier than right)
- **Analytical solutions**: Closed-form optimal portfolio weights (no simulation needed)
- **Option nonlinearity**: Delta-gamma approximation handles nonlinear payoffs

## Key Features

✅ **Closed-form optimal weights** - Theorems 3.1 (Sharpe) & 3.2 (Return-to-VaR)  
✅ **Skew-t distribution fitting** - MLE on real returns data  
✅ **Black-Scholes Greeks** - Delta, gamma, theta, vega computation  
✅ **Delta-gamma approximation** - Explicit u and Q formulas from Appendix A  
✅ **Constrained optimization** - Box constraints for practical trading  
✅ **Complete validation** - Put-call parity, parameter recovery, convergence tests  
✅ **Production-ready code** - ~2,100 lines of tested Python  

## Installation

### Via pip
```bash
pip install option-portfolio-skew-t
```

### From source
```bash
git clone https://github.com/nickr145/option-portfolio-skew-t.git
cd option-portfolio-skew-t
pip install -e .
```

### With optional dependencies
```bash
# For real data download
pip install option-portfolio-skew-t[data]

# For development
pip install option-portfolio-skew-t[dev]

# For Jupyter notebooks
pip install option-portfolio-skew-t[notebooks]
```

## Quick Start

### Minimal example (5 lines)
```python
from option_portfolio import OptionPortfolioOptimizer

optimizer = OptionPortfolioOptimizer(returns, spot_prices)
optimizer.fit_skew_t()
optimizer.setup_options([{'type': 'c', 'strike': S, 'underlying_idx': i} for i, S in enumerate(spot_prices)])
optimizer.assemble_portfolio_params()
x_opt, metrics = optimizer.optimize_sharpe_analytical()
```

### Full example with real data
```python
import yfinance as yf
from option_portfolio import OptionPortfolioOptimizer

# Download data
data = yf.download(['AAPL', 'MSFT', 'GOOGL'], start='2023-01-01', end='2024-12-31')['Adj Close']
returns = data.pct_change().dropna()
spot_prices = data.iloc[-1].values

# Initialize optimizer
optimizer = OptionPortfolioOptimizer(returns.values, spot_prices, asset_names=['AAPL', 'MSFT', 'GOOGL'])

# Pipeline
optimizer.fit_skew_t()
option_specs = [{'type': 'c', 'strike': spot_prices[i], 'underlying_idx': i} for i in range(3)]
optimizer.setup_options(option_specs, T=30/365, r=0.02, sigma_surface=0.25)
optimizer.assemble_portfolio_params()

# Optimize
x_sharpe, m_sharpe = optimizer.optimize_sharpe_analytical()
x_rvar, m_rvar, lam = optimizer.optimize_rvar_analytical()
x_sharpe_c, _, _ = optimizer.optimize_sharpe_constrained(box_bounds=(-0.5, 0.5))

# View results
print(f"Sharpe ratio: {m_sharpe['sharpe_ratio']:.4f}")
print(f"R-VaR ratio: {m_rvar['r_var_ratio']:.4f}")
```

## Core Components

### 1. Skew-t Distribution Fitting
```python
from option_portfolio import SkewTFitter

fitter = SkewTFitter(returns)
params = fitter.fit(method='mle')
# params: {'mu': ..., 'sigma': ..., 'nu': ..., 'omega': ...}
```

Fits multivariate skew-elliptical t-distribution via maximum likelihood estimation. Parameters:
- **μ**: Location vector (like mean)
- **Σ**: Scale matrix (like covariance)
- **ν**: Degrees of freedom (lower = heavier tails)
- **ω**: Skewness vector (asymmetry direction)

### 2. Black-Scholes Greeks
```python
from option_portfolio import BlackScholesGreeks

price = BlackScholesGreeks.call_price(S, K, T, r, sigma)
delta = BlackScholesGreeks.delta('c', S, K, T, r, sigma)
gamma = BlackScholesGreeks.gamma(S, K, T, r, sigma)
theta = BlackScholesGreeks.theta('c', S, K, T, r, sigma)
vega = BlackScholesGreeks.vega(S, K, T, r, sigma)
```

Standard Black-Scholes Greeks for European options.

### 3. Portfolio Optimization

**Theorem 3.1: Sharpe Ratio Maximization**
$\mathbf{x}^*_{Sharpe}=\frac {Q^{-1}(\mathbf{u}-r_f\mathbf{v})}{\mathbf{v}^TQ^{-1}(\mathbf{u}-r_f\mathbf{v})}$

**Theorem 3.2: Return-to-VaR Maximization**
$\mathbf{x}^*_{{R-VaR}^\alpha_2}=\frac {Q^{-1}[(1-\lambda^*)\mathbf{u}-r_f\mathbf{v}]}{\mathbf{v}^TQ^{-1}[(1-\lambda^*)\mathbf{u}-r_f\mathbf{v}]}$

Where:
- **u**: Expected P&L vector (computed from Greeks + skew-t parameters)
- **Q**: Variance-covariance matrix (from delta-gamma approximation)
- **v**: Option values
- **λ***: Optimal scaling parameter (from numerical optimization)

## Documentation

| File | Purpose |
|------|---------|
| [QUICKSTART.md](docs/QUICKSTART.md) | 5-minute quick start |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | Function documentation |
| [MATHEMATICAL_DETAILS.md](docs/MATHEMATICAL_DETAILS.md) | Theory and proofs |
| [IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) | Extension guide |

## Examples

See `examples/` directory:
- `synthetic_demo.py` - Educational tutorial with synthetic data
- `real_data_example.py` - Real market data example using yfinance

Run examples:
```bash
python examples/synthetic_demo.py
python examples/real_data_example.py  # Requires yfinance
```

## Testing

Run validation tests:
```bash
pip install pytest
pytest tests/
```

Tests include:
- Put-call parity validation
- Delta-gamma approximation accuracy
- Parameter recovery from synthetic data
- Optimization convergence

## Citation

If you use this in research, please cite the original paper:

```bibtex
@article{sung2026sharpe,
  title={Sharpe Ratio and Return-VaR Ratio Maximization for Option Portfolios 
         with Skew-Elliptical t Underlying Returns},
  author={Sung, Kyle and Pirvu, Traian A},
  journal={arXiv preprint arXiv:2606.17032},
  year={2026}
}
```

And the implementation (optional):
```bibtex
@misc{rebello2024implementation,
  title={Implementation of Sung \& Pirvu (2026)},
  author={Rebello, Nicholas},
  year={2024},
  url={https://github.com/nickr145/option-portfolio-skew-t}
}
```

## Key Findings

The paper demonstrates:

1. **Sharpe and R-VaR portfolios differ** when unconstrained
   - Unconstrained L1 distance: ~0.59
   - This is because tail risk optimization creates different allocations

2. **They converge under practical constraints** (±50% position limits)
   - Constrained L1 distance: ~0.008
   - Practical trading constraints matter more than optimization objective

3. **Skew-t is more realistic than normal**
   - ν=6 in paper means tails 2-3x heavier than normal
   - Skewness captures asymmetric market behavior

## References

**Main paper:**
- Sung, K. & Pirvu, T.A. (2026). "Sharpe Ratio and Return-VaR Ratio Maximization for Option Portfolios with Skew-Elliptical t Underlying Returns." arXiv:2606.17032v1

**Foundational:**
- Azzalini, A. & Capitanio, A. (2003). "Distributions generated by perturbation of symmetry with emphasis on a multivariate skew t-distribution." JRSS Series B 65(2): 367-389.

**Related:**
- Hu, W. & Kercheval, A.N. (2010). "Portfolio optimization for Student t and skewed t returns." Quantitative Finance 10(1): 91-105.
- Cui, X., Zhu, S., Sun, X., & Li, D. (2013). "Nonlinear portfolio selection using approximate parametric value-at-risk." Journal of Banking & Finance 37(6): 2124-2139.

## License

MIT License - See [LICENSE](LICENSE) file

## Author

Nicholas Rebello
