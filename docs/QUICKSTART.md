# Quick Start Guide

Get up and running in 5 minutes.

## Installation

```bash
pip install option-portfolio-skew-t
```

## Your First Optimization

### Step 1: Prepare Data
```python
import numpy as np
from option_portfolio import OptionPortfolioOptimizer

# Your returns data: (T, N) - T time periods, N assets
returns = np.random.randn(500, 5) * 0.01 + 0.0005

# Current spot prices for 5 assets
spot_prices = np.array([100, 150, 80, 120, 110])
```

### Step 2: Initialize Optimizer
```python
optimizer = OptionPortfolioOptimizer(
    returns, 
    spot_prices,
    asset_names=['Stock_1', 'Stock_2', 'Stock_3', 'Stock_4', 'Stock_5']
)
```

### Step 3: Fit Skew-t Distribution
```python
optimizer.fit_skew_t(verbose=True)
# Outputs fitted parameters: μ, Σ, ν, ω
```

### Step 4: Define Options
```python
# ATM (at-the-money) calls on all 5 stocks
option_specs = [
    {'type': 'c', 'strike': spot_prices[i], 'underlying_idx': i}
    for i in range(5)
]

optimizer.setup_options(
    option_specs,
    T=30/365,  # 30 days to expiry
    r=0.02,    # 2% risk-free rate
    sigma_surface=0.25  # 25% implied volatility
)
```

### Step 5: Assemble Portfolio Parameters
```python
u, Q = optimizer.assemble_portfolio_params()
# u = expected P&L vector
# Q = variance-covariance matrix
```

### Step 6: Optimize (Unconstrained)
```python
# Sharpe ratio maximization (Theorem 3.1)
x_sharpe, metrics_sharpe = optimizer.optimize_sharpe_analytical()

print(f"Sharpe ratio: {metrics_sharpe['sharpe_ratio']:.4f}")
print(f"Expected return: {metrics_sharpe['exp_return']:.4f}")
print(f"Volatility: {metrics_sharpe['volatility']:.4f}")
print(f"Portfolio shares: {x_sharpe}")
```

```python
# Return-to-VaR maximization (Theorem 3.2)
x_rvar, metrics_rvar, lambda_star = optimizer.optimize_rvar_analytical(alpha=0.01)

print(f"R-VaR ratio: {metrics_rvar['r_var_ratio']:.4f}")
print(f"λ*: {lambda_star:.4f}")
```

### Step 7: Optimize (Constrained)
```python
# Add practical box constraints: ±50% position limits
x_sharpe_c, metrics_c, _ = optimizer.optimize_sharpe_constrained(
    box_bounds=(-0.5, 0.5)
)

print(f"Constrained Sharpe: {metrics_c['sharpe_ratio']:.4f}")
```

## With Real Data

```python
import yfinance as yf
from option_portfolio import OptionPortfolioOptimizer

# Step 1: Download data
tickers = ['AAPL', 'MSFT', 'GOOGL']
data = yf.download(tickers, start='2023-01-01', end='2024-12-31')['Adj Close']
returns = data.pct_change().dropna()
spot_prices = data.iloc[-1].values

# Step 2-7: Same as above
optimizer = OptionPortfolioOptimizer(returns.values, spot_prices, asset_names=tickers)
optimizer.fit_skew_t()

option_specs = [
    {'type': 'c', 'strike': spot_prices[i], 'underlying_idx': i}
    for i in range(len(tickers))
]
optimizer.setup_options(option_specs, T=30/365, r=0.02, sigma_surface=0.25)
optimizer.assemble_portfolio_params()

x_sharpe, m_sharpe = optimizer.optimize_sharpe_analytical()
```

## Common Patterns

### Change Option Type (Puts Instead of Calls)
```python
option_specs = [
    {'type': 'p', 'strike': spot_prices[i] * 0.95, 'underlying_idx': i}  # 5% OTM puts
    for i in range(len(spot_prices))
]
```

### Different Time Horizons
```python
optimizer.setup_options(option_specs, T=60/365)  # 60 days instead of 30
optimizer.setup_options(option_specs, T=1/12)     # 1 month
optimizer.setup_options(option_specs, T=1)        # 1 year
```

### Change VaR Confidence Level
```python
x_rvar, m_rvar, lam = optimizer.optimize_rvar_analytical(alpha=0.05)  # 5% VaR instead of 1%
```

### Extract Metrics for Any Portfolio
```python
x = np.array([0.2, -0.1, 0.15, 0.05, 0.1])  # Your allocation
metrics = optimizer.portfolio_metrics(x)

print(f"Sharpe: {metrics['sharpe_ratio']:.4f}")
print(f"R-VaR: {metrics['r_var_ratio']:.4f}")
print(f"Return: {metrics['exp_return']:.4f}")
print(f"Volatility: {metrics['volatility']:.4f}")
print(f"Weights: {metrics['weights']}")
```

## Understanding the Output

### metrics dictionary
```python
{
    'exp_return': 0.0045,           # Expected return
    'volatility': 0.0832,           # Standard deviation
    'sharpe_ratio': 0.0541,         # Return per unit volatility
    'r_var_ratio': 0.0234,          # Return per unit VaR
    'weights': array([...]),        # Dollar weights (sums to 1)
    'shares': array([...])          # Number of shares (unconstrained)
}
```

### skew_t_params dictionary
```python
{
    'mu': array([...]),             # Location vector (N,)
    'sigma': array([...]),          # Scale matrix (N, N)
    'nu': 6.041,                    # Degrees of freedom (lower = heavier tails)
    'omega': array([...])           # Skewness vector (N,)
}
```

## Troubleshooting

**Q: "Singular matrix Q"**  
A: Q is ill-conditioned. This can happen with certain data. Try regularization:
```python
Q_reg = optimizer.Q + 1e-4 * np.eye(optimizer.Q.shape[0])
```

**Q: "No feasible solution"**  
A: Constraints too tight. Relax them:
```python
x_opt, _, _ = optimizer.optimize_sharpe_constrained(box_bounds=(-1.0, 1.0))
```

**Q: "Optimization didn't converge"**  
A: Try different initial point or optimization method. Usually convergence is quick (~10 iterations).

**Q: "ImportError: No module named yfinance"**  
A: Install yfinance:
```bash
pip install yfinance
```

## Next Steps

- Read [API_REFERENCE.md](API_REFERENCE.md) for detailed function documentation
- Check [MATHEMATICAL_DETAILS.md](MATHEMATICAL_DETAILS.md) for theory
- See [examples/](../examples/) for complete working scripts
- Review [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for extensions

## Key Paper Results

The paper (Sung & Pirvu 2026) shows:

1. **Theorem 3.1**: Closed-form optimal weights for Sharpe ratio under skew-t returns
2. **Theorem 3.2**: Closed-form optimal weights for Return-to-VaR under skew-t returns
3. **Finding**: Constrained and unconstrained solutions converge with practical box constraints

This implementation reproduces all these results with validated, tested Python code.