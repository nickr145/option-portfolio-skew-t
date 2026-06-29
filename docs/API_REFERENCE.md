# API Reference

Complete function and class documentation.

## OptionPortfolioOptimizer

Main class for end-to-end portfolio optimization.

### Constructor
```python
OptionPortfolioOptimizer(returns_data, spot_prices, asset_names=None)
```

**Parameters:**
- `returns_data` (np.ndarray): (T, N) matrix of T returns observations on N assets
- `spot_prices` (np.ndarray): (N,) vector of current spot prices
- `asset_names` (list, optional): List of N asset names for labeling

**Attributes:**
- `skew_t_params` (dict): Fitted skew-t parameters {mu, sigma, nu, omega}
- `u` (np.ndarray): Expected P&L vector (M,)
- `Q` (np.ndarray): Variance-covariance matrix (M, M)
- `v` (np.ndarray): Option values (M,)

### Methods

#### fit_skew_t()
```python
fit_skew_t(verbose=True) -> dict
```

Fit multivariate skew-t distribution to returns data via MLE.

**Parameters:**
- `verbose` (bool): Print progress and results

**Returns:**
- Dictionary with keys: 'mu', 'sigma', 'nu', 'omega'

**Example:**
```python
params = optimizer.fit_skew_t(verbose=True)
print(f"ν = {params['nu']:.3f}")  # Degrees of freedom
```

---

#### setup_options()
```python
setup_options(option_specs, T=30/365, r=0.02, sigma_surface=None, verbose=True) -> dict
```

Set up option portfolio.

**Parameters:**
- `option_specs` (list): List of option specifications. Each element is a dict:
  - `'type'` (str): 'c' for call, 'p' for put
  - `'strike'` (float): Strike price
  - `'underlying_idx'` (int): Index of underlying asset (0 to N-1)
- `T` (float): Time to expiry in years (default: 30/365)
- `r` (float): Risk-free rate (annual, default: 0.02)
- `sigma_surface` (float or callable): Implied volatility
  - If float: constant volatility
  - If callable: function of (spot, strike) -> volatility
  - If None: defaults to 0.20
- `verbose` (bool): Print progress

**Returns:**
- Dictionary with Greeks: 'delta', 'gamma', 'theta', 'vega', 'prices'

**Example:**
```python
option_specs = [
    {'type': 'c', 'strike': 100, 'underlying_idx': 0},
    {'type': 'p', 'strike': 95, 'underlying_idx': 0},
]
optimizer.setup_options(option_specs, T=30/365, r=0.02, sigma_surface=0.25)
```

---

#### assemble_portfolio_params()
```python
assemble_portfolio_params(dt=1/252, verbose=True) -> (np.ndarray, np.ndarray)
```

Compute u and Q matrices from Appendix A (delta-gamma approximation with skew-t).

**Parameters:**
- `dt` (float): Time step (default: 1/252 for daily)
- `verbose` (bool): Print results

**Returns:**
- Tuple (u, Q):
  - `u` (np.ndarray): Expected P&L vector (M,)
  - `Q` (np.ndarray): Variance-covariance matrix (M, M)

**Details:**
- `u` = expected P&L from time decay, drift, and skew-t adjustments
- `Q` = variance structure incorporating gamma and distributional parameters

---

#### optimize_sharpe_analytical()
```python
optimize_sharpe_analytical(verbose=True) -> (np.ndarray, dict)
```

Theorem 3.1: Analytical Sharpe ratio maximization (unconstrained).

$$x^*_{\text{Sharpe}} = \frac{Q^{-1}(u - r_f v)}{v^T Q^{-1}(u - r_f v)}$$

**Parameters:**
- `verbose` (bool): Print results

**Returns:**
- Tuple (x_optimal, metrics):
  - `x_optimal` (np.ndarray): Optimal shares (M,)
  - `metrics` (dict): Portfolio metrics

**Example:**
```python
x, m = optimizer.optimize_sharpe_analytical()
print(f"Sharpe ratio: {m['sharpe_ratio']:.4f}")
print(f"Expected return: {m['exp_return']:.4f}")
print(f"Volatility: {m['volatility']:.4f}")
```

---

#### optimize_rvar_analytical()
```python
optimize_rvar_analytical(alpha=0.01, verbose=True) -> (np.ndarray, dict, float)
```

Theorem 3.2: Analytical Return-to-VaR maximization (unconstrained).

$$x^*_{\text{R-VaR}} = \frac{Q^{-1}[(1 - \lambda^*) u - r_f v]}{v^T Q^{-1}[(1 - \lambda^*) u - r_f v]}$$

**Parameters:**
- `alpha` (float): VaR confidence level (default: 0.01 = 1% tail)
- `verbose` (bool): Print results

**Returns:**
- Tuple (x_optimal, metrics, lambda_star):
  - `x_optimal` (np.ndarray): Optimal shares (M,)
  - `metrics` (dict): Portfolio metrics
  - `lambda_star` (float): Optimal λ parameter

**Example:**
```python
x, m, lam = optimizer.optimize_rvar_analytical(alpha=0.05)  # 5% VaR
print(f"R-VaR ratio: {m['r_var_ratio']:.4f}")
print(f"λ*: {lam:.4f}")
```

---

#### optimize_sharpe_constrained()
```python
optimize_sharpe_constrained(box_bounds=(-0.5, 0.5), verbose=True) -> (np.ndarray, dict, OptimizeResult)
```

Numerical Sharpe ratio maximization with box constraints.

**Parameters:**
- `box_bounds` (tuple): (lower, upper) bounds for each position
  - Example: (-0.5, 0.5) means -50% to +50% positions
- `verbose` (bool): Print results

**Returns:**
- Tuple (x_optimal, metrics, scipy_result):
  - `x_optimal` (np.ndarray): Optimal shares (M,)
  - `metrics` (dict): Portfolio metrics
  - `scipy_result` (OptimizeResult): Scipy optimization result

**Example:**
```python
x, m, res = optimizer.optimize_sharpe_constrained(box_bounds=(-0.3, 0.3))
print(f"Optimization success: {res.success}")
print(f"Constrained Sharpe: {m['sharpe_ratio']:.4f}")
```

---

#### optimize_rvar_constrained()
```python
optimize_rvar_constrained(alpha=0.01, box_bounds=(-0.5, 0.5), verbose=True) -> (np.ndarray, dict, OptimizeResult)
```

Numerical Return-to-VaR maximization with box constraints.

**Parameters:**
- `alpha` (float): VaR confidence level (default: 0.01)
- `box_bounds` (tuple): Position bounds
- `verbose` (bool): Print results

**Returns:**
- Tuple (x_optimal, metrics, scipy_result)

---

#### portfolio_metrics()
```python
portfolio_metrics(x, alpha=0.01) -> dict
```

Compute metrics for any portfolio allocation.

**Parameters:**
- `x` (np.ndarray): Portfolio shares (M,)
- `alpha` (float): VaR confidence level for R-VaR calculation

**Returns:**
- Dictionary with keys:
  - `'exp_return'`: Expected return
  - `'volatility'`: Standard deviation
  - `'sharpe_ratio'`: Sharpe ratio
  - `'r_var_ratio'`: Return-to-VaR ratio
  - `'weights'`: Dollar weights (sum to 1)
  - `'shares'`: Your input x

**Example:**
```python
x = np.array([0.2, -0.1, 0.15])
m = optimizer.portfolio_metrics(x)
print(m['sharpe_ratio'])
```

---

## SkewTFitter

Fit multivariate skew-t distribution.

### Constructor
```python
SkewTFitter(returns_data)
```

**Parameters:**
- `returns_data` (np.ndarray): (T, N) returns matrix

### Methods

#### fit()
```python
fit(method='mle', initial_nu=5.0) -> dict
```

Fit skew-t distribution.

**Parameters:**
- `method` (str): 'mle' (recommended) or 'moment'
- `initial_nu` (float): Starting point for degrees of freedom

**Returns:**
- Dictionary with keys: 'mu', 'sigma', 'nu', 'omega'

---

#### summary()
```python
summary() -> None
```

Print summary of fitted parameters.

---

## BlackScholesGreeks

Static methods for Black-Scholes Greeks.

### Methods

#### call_price()
```python
BlackScholesGreeks.call_price(S, K, T, r, sigma) -> float
```

Black-Scholes call price.

#### put_price()
```python
BlackScholesGreeks.put_price(S, K, T, r, sigma) -> float
```

Black-Scholes put price.

#### delta()
```python
BlackScholesGreeks.delta(option_type, S, K, T, r, sigma) -> float
```

Option delta (∂V/∂S).

**Parameters:**
- `option_type` (str): 'c' for call, 'p' for put

**Returns:**
- Delta value (0-1 for calls, -1-0 for puts)

#### gamma()
```python
BlackScholesGreeks.gamma(S, K, T, r, sigma) -> float
```

Option gamma (∂²V/∂S²). Same for calls and puts.

#### theta()
```python
BlackScholesGreeks.theta(option_type, S, K, T, r, sigma) -> float
```

Option theta (time decay per day).

#### vega()
```python
BlackScholesGreeks.vega(S, K, T, r, sigma) -> float
```

Option vega (per 1% volatility change).

---

## DeltaGammaPortfolio

Assemble delta-gamma approximation parameters.

### Constructor
```python
DeltaGammaPortfolio(greeks, spot_prices, skew_t_params, dt=1/252, r=0.02)
```

**Parameters:**
- `greeks` (dict): Greeks data from compute_greeks_for_options()
- `spot_prices` (np.ndarray): Current spot prices
- `skew_t_params` (dict): Fitted skew-t parameters
- `dt` (float): Time step (default: daily)
- `r` (float): Risk-free rate

### Methods

#### compute_u_vector()
```python
compute_u_vector() -> np.ndarray
```

Compute expected P&L vector from Appendix A formulas.

**Returns:**
- `u` (np.ndarray): Expected P&L vector (M,)

---

#### compute_Q_matrix()
```python
compute_Q_matrix() -> np.ndarray
```

Compute variance-covariance matrix from Appendix A formulas.

**Returns:**
- `Q` (np.ndarray): Variance-covariance matrix (M, M)

---

## Utility Functions

### compute_greeks_for_options()
```python
from option_portfolio.greeks import compute_greeks_for_options

compute_greeks_for_options(option_specs, spot_prices, r, T, sigma_surface=None) -> dict
```

Compute Greeks for a portfolio of options.

**Parameters:**
- `option_specs` (list): Option specifications
- `spot_prices` (np.ndarray): Current prices
- `r` (float): Risk-free rate
- `T` (float): Time to expiry
- `sigma_surface` (float or callable): Implied volatility

**Returns:**
- Dictionary with 'delta', 'gamma', 'theta', 'vega', 'prices'

---

## Common Patterns

### Get Greeks for Multiple Options
```python
from option_portfolio.greeks import compute_greeks_for_options

option_specs = [
    {'type': 'c', 'strike': 100, 'underlying_idx': 0},
    {'type': 'p', 'strike': 95, 'underlying_idx': 0},
]
greeks = compute_greeks_for_options(option_specs, spot_prices, r=0.02, T=30/365)
```

### Sensitivity Analysis
```python
for alpha in [0.01, 0.05, 0.10]:
    x, m, lam = optimizer.optimize_rvar_analytical(alpha=alpha)
    print(f"α={alpha}: R-VaR={m['r_var_ratio']:.4f}")
```

### Compare Multiple Portfolios
```python
portfolios = {
    'sharpe': optimizer.optimize_sharpe_analytical()[0],
    'rvar': optimizer.optimize_rvar_analytical()[0],
    'equal_weight': np.ones(optimizer.M) / optimizer.M,
}

for name, x in portfolios.items():
    m = optimizer.portfolio_metrics(x)
    print(f"{name}: Sharpe={m['sharpe_ratio']:.4f}, Return={m['exp_return']:.4f}")
```