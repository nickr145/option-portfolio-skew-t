# Implementation Guide: How to Extend

This guide shows you how to extend the implementation with new features.

## Level 1: Easy Modifications (30 minutes each)

### 1.1 Add Transaction Costs

**Motivation:** Real trading has costs (bid-ask spreads, commissions).

**Implementation:**

```python
def optimize_sharpe_with_costs(optimizer, x_prev, cost_rate=0.001):
    """
    Optimize Sharpe ratio with transaction costs.
    
    Args:
        cost_rate (float): Cost as percentage of trade size
    """
    def objective(x):
        # Original objective
        exp_ret = optimizer.u @ x - optimizer.r
        std = np.sqrt(0.5 * x @ optimizer.Q @ x)
        sharpe = -exp_ret / std if std > 0 else -1e10
        
        # Add transaction cost
        trading_cost = cost_rate * np.sum(np.abs(x - x_prev))
        
        return sharpe + trading_cost / (std + 1e-10)
    
    # Optimize with constraints
    constraints = {'type': 'eq', 'fun': lambda x: (optimizer.v @ x) - 1.0}
    bounds = [(-0.5, 0.5)] * optimizer.M
    
    from scipy.optimize import minimize
    result = minimize(objective, x_prev, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    return result.x
```

### 1.2 Different Option Types

**Motivation:** Test puts, OTM options, straddles.

**Implementation:**

```python
# ATM puts instead of calls
option_specs = [
    {'type': 'p', 'strike': spot_prices[i], 'underlying_idx': i}
    for i in range(len(spot_prices))
]

# OTM calls (5% above spot)
option_specs = [
    {'type': 'c', 'strike': spot_prices[i] * 1.05, 'underlying_idx': i}
    for i in range(len(spot_prices))
]

# Straddle (call + put at same strike)
option_specs = [
    {'type': 'c', 'strike': spot_prices[0], 'underlying_idx': 0},
    {'type': 'p', 'strike': spot_prices[0], 'underlying_idx': 0},
]

optimizer.setup_options(option_specs, T=30/365, r=0.02, sigma_surface=0.25)
```

### 1.3 Different Risk Metrics

**Motivation:** Try Sortino ratio (downside deviation only).

**Implementation:**

```python
def optimize_sortino_ratio(optimizer):
    """
    Maximize Sortino ratio = E[R] / downside_deviation
    """
    def objective(x):
        exp_ret = optimizer.u @ x - optimizer.r
        
        # Downside deviation (only negative P&L)
        var = 0.5 * x @ optimizer.Q @ x
        downside = np.sqrt(np.maximum(var, 0))
        
        sortino = -exp_ret / (downside + 1e-10)
        return sortino
    
    constraints = {'type': 'eq', 'fun': lambda x: (optimizer.v @ x) - 1.0}
    bounds = [(-0.5, 0.5)] * optimizer.M
    x0 = np.ones(optimizer.M) / optimizer.M
    
    from scipy.optimize import minimize
    result = minimize(objective, x0, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    return result.x, optimizer.portfolio_metrics(result.x)
```

### 1.4 Time Horizon Sensitivity

**Motivation:** See how optimal portfolio changes with time to expiry.

**Implementation:**

```python
T_values = [7/365, 14/365, 30/365, 60/365, 90/365]  # 1-3 months

results = {}
for T in T_values:
    optimizer.setup_options(option_specs, T=T, r=0.02, sigma_surface=0.25)
    optimizer.assemble_portfolio_params()
    
    x_opt, metrics = optimizer.optimize_sharpe_analytical(verbose=False)
    results[T*365] = {'x': x_opt, 'metrics': metrics}

# Plot how Sharpe ratio changes with T
import matplotlib.pyplot as plt
days = list(results.keys())
sharpes = [results[d]['metrics']['sharpe_ratio'] for d in days]
plt.plot(days, sharpes)
plt.xlabel('Days to Expiry')
plt.ylabel('Sharpe Ratio')
plt.show()
```

---

## Level 2: Medium Extensions (1-2 hours each)

### 2.1 Implied Volatility Surface

**Motivation:** Use realistic vol smile instead of constant vol.

**Implementation:**

```python
def iv_surface(spot, strike):
    """
    Quadratic volatility smile.
    Lower vol at ATM, higher at OTM.
    """
    moneyness = strike / spot
    base_vol = 0.20
    smile_curvature = 0.05
    
    vol = base_vol + smile_curvature * (moneyness - 1)**2
    return np.clip(vol, 0.10, 0.50)  # Bounds

# Use in optimization
optimizer.setup_options(option_specs, T=30/365, r=0.02, sigma_surface=iv_surface)
```

### 2.2 Multi-Period Optimization

**Motivation:** Rebalance portfolio over time.

**Implementation:**

```python
class MultiPeriodOptimizer:
    """Optimize portfolio over multiple periods with rebalancing."""
    
    def __init__(self, returns_data, spot_prices):
        self.returns = returns_data
        self.spots = spot_prices
        
    def optimize_rolling(self, window=250, rebalance_freq=20):
        """
        Rolling window optimization with rebalancing.
        
        Args:
            window (int): Rolling window size (days)
            rebalance_freq (int): Rebalance every N days
        """
        T = len(self.returns)
        portfolios = []
        
        for t in range(window, T, rebalance_freq):
            # Use returns from [t-window : t]
            returns_window = self.returns[t-window:t]
            
            # Optimize using this window
            optimizer = OptionPortfolioOptimizer(returns_window, self.spots[t])
            optimizer.fit_skew_t(verbose=False)
            optimizer.setup_options(...)
            optimizer.assemble_portfolio_params()
            
            x_opt, _ = optimizer.optimize_sharpe_analytical(verbose=False)
            portfolios.append({'date': t, 'allocation': x_opt})
        
        return portfolios
```

### 2.3 Liquidity-Adjusted Pricing

**Motivation:** Account for bid-ask spreads in option prices.

**Implementation:**

```python
def price_with_liquidity(S, K, T, r, sigma, bid_ask_spread=0.01):
    """
    Adjust option price for liquidity.
    
    Args:
        bid_ask_spread (float): Spread as % of price
    """
    from option_portfolio import BlackScholesGreeks
    
    mid_price = BlackScholesGreeks.call_price(S, K, T, r, sigma)
    
    # When buying (ask): pay more
    # When selling (bid): receive less
    spread_adjustment = bid_ask_spread / 2
    
    return mid_price * (1 + spread_adjustment)  # For long positions

# Use in setup_options by modifying Greeks
# (More involved - would need to wrap compute_greeks_for_options)
```

### 2.4 Regime-Switching Model

**Motivation:** Market regimes (bull/bear) have different ν.

**Implementation:**

```python
class RegimeSwitchingOptimizer:
    """
    Optimize under multiple market regimes.
    """
    
    def __init__(self, returns_data, spot_prices, n_regimes=2):
        self.returns = returns_data
        self.spots = spot_prices
        self.n_regimes = n_regimes
        
    def fit_regimes(self):
        """Fit hidden Markov model with skew-t emissions."""
        from hmmlearn import hmm
        
        # Fit HMM with skew-t emissions
        # (Implementation details omitted - complex)
        pass
    
    def optimize_with_regime_probs(self, regime_probs):
        """
        Optimize portfolio as weighted combination over regimes.
        
        Args:
            regime_probs (np.ndarray): Probability of each regime
        """
        x_regimes = []
        
        for regime in range(self.n_regimes):
            # Fit skew-t for this regime
            regime_returns = self.returns[self.regime_labels == regime]
            
            optimizer = OptionPortfolioOptimizer(regime_returns, self.spots)
            optimizer.fit_skew_t(verbose=False)
            optimizer.setup_options(...)
            optimizer.assemble_portfolio_params()
            
            x_opt, _ = optimizer.optimize_sharpe_analytical(verbose=False)
            x_regimes.append(x_opt)
        
        # Blend portfolios by regime probability
        x_blended = np.average(x_regimes, axis=0, weights=regime_probs)
        return x_blended
```

---

## Level 3: Advanced Extensions (2-4 hours each)

### 3.1 Higher-Order Approximations

**Motivation:** Delta-gamma-vanna-volga for better accuracy.

**Theory:**
- Delta-gamma is 2nd order: O(ΔS²)
- Delta-gamma-vanna-volga is 4th order: O(ΔS⁴)
- Vanna = ∂²V/∂S∂σ (delta-vol cross-gamma)
- Volga = ∂²V/∂σ² (vol-vol gamma)

**Implementation sketch:**
```python
def fourth_order_pnl(x, delta, gamma, vanna, volga, delta_s, delta_sigma):
    """
    4th order P&L approximation.
    """
    pnl_2nd = delta @ delta_s + 0.5 * (delta_s**2) @ gamma
    pnl_cross = vanna @ (delta_s * delta_sigma)
    pnl_4th = 0.5 * volga @ (delta_sigma**2)
    
    return pnl_2nd + pnl_cross + pnl_4th
```

### 3.2 Monte Carlo Validation

**Motivation:** Compare analytical delta-gamma to simulation.

**Implementation:**

```python
def validate_with_monte_carlo(optimizer, n_paths=10000):
    """
    Validate delta-gamma approximation via Monte Carlo.
    """
    # Simulate skew-t returns
    returns_sim = optimizer.skew_t_fitter.simulate(n_paths)
    
    # Compute portfolio P&Ls
    pnl_actual = np.zeros(n_paths)
    pnl_approx = np.zeros(n_paths)
    
    x_opt, _ = optimizer.optimize_sharpe_analytical()
    
    for i in range(n_paths):
        # Actual P&L (would need option repricing)
        # pnl_actual[i] = recompute_option_portfolio(returns_sim[i])
        
        # Approximate P&L (delta-gamma)
        delta_s = returns_sim[i]
        pnl_approx[i] = (x_opt * optimizer.delta) @ delta_s
        pnl_approx[i] += 0.5 * delta_s @ (x_opt @ optimizer.gamma) @ delta_s
    
    # Compare distributions
    print(f"E[PnL actual]: {pnl_actual.mean():.6f}")
    print(f"E[PnL approx]: {pnl_approx.mean():.6f}")
    print(f"Std[PnL actual]: {pnl_actual.std():.6f}")
    print(f"Std[PnL approx]: {pnl_approx.std():.6f}")
```

### 3.3 Stochastic Volatility

**Motivation:** Vol is not constant - use Heston model.

**Theory:**
- Heston: dS = μS dt + √v S dW₁, dv = κ(θ - v) dt + σ_v √v dW₂
- ρ: correlation between asset and vol shocks
- Need to include vanna in P&L

**Implementation:** Complex - would involve:
1. Fitting Heston parameters
2. Computing Greeks under Heston
3. Extending delta-gamma to include vol dynamics

---

## Level 4: Research Directions (4+ hours)

### 4.1 Backtesting Engine

**What it does:**
- Walk-forward optimization (rolling window)
- Transaction costs and slippage
- Compare vs benchmarks
- Compute returns, Sharpe, max drawdown

**Key metrics:**
```python
metrics = {
    'returns': portfolio_returns,
    'sharpe': annual_return / annual_std,
    'sortino': annual_return / downside_std,
    'max_drawdown': compute_max_drawdown(cumulative_returns),
    'win_rate': percent_positive_days,
    'var_95': np.percentile(returns, 5),  # Value-at-Risk
    'cvar_95': returns[returns < np.percentile(returns, 5)].mean(),  # CVaR
}
```

### 4.2 Diversification Constraints

**Motivation:** Avoid over-concentration.

**Implementation:**
```python
# Constraint: no position > 20% of total portfolio
constraints.append({'type': 'ineq', 'fun': lambda x: 0.20 - np.abs(x)})

# Constraint: correlation with market < 0.8
constraints.append({'type': 'ineq', 'fun': lambda x: 0.80 - np.abs(np.corrcoef(x, market)[0,1])})
```

### 4.3 Machine Learning Integration

**Directions:**
- Use LSTM to predict ν (regime estimation)
- Neural network for implied vol surface
- Reinforcement learning for dynamic rebalancing
- Generative models for synthetic scenarios

---

## Common Debugging Issues

### Issue: "Q matrix is singular"

**Cause:** Ill-conditioned covariance structure

**Solution:**
```python
# Regularize Q
epsilon = 1e-4
Q_reg = optimizer.Q + epsilon * np.eye(optimizer.Q.shape[0])

# Use regularized Q for optimization
# (Would need to modify optimizer class)
```

### Issue: "Optimization doesn't converge"

**Cause:** Bad initial point or tight constraints

**Solution:**
```python
# Try different initial points
x0_candidates = [
    np.ones(optimizer.M) / optimizer.M,  # Equal weight
    optimizer.optimize_sharpe_analytical(verbose=False)[0],  # Start from analytical
    np.random.randn(optimizer.M) / optimizer.M,  # Random
]

best_result = None
for x0 in x0_candidates:
    result = minimize(objective, x0, method='SLSQP', ...)
    if result.fun < best_result.fun:
        best_result = result
```

### Issue: "Parameters don't converge"

**Cause:** Bad initial values or data issues

**Solution:**
```python
# Try multiple initial nu values
for nu_init in [3, 5, 7, 10, 15]:
    fitter = SkewTFitter(returns)
    try:
        params = fitter.fit(initial_nu=nu_init)
        if params['nu'] > 2:  # Valid
            break
    except:
        continue
```

---

## Testing Your Extensions

```python
def test_extension():
    """Template for testing new feature."""
    # 1. Create synthetic data
    np.random.seed(42)
    returns = np.random.randn(200, 3) * 0.01
    spots = np.array([100, 120, 90])
    
    # 2. Run extension
    optimizer = OptionPortfolioOptimizer(returns, spots)
    optimizer.fit_skew_t(verbose=False)
    optimizer.setup_options(...)
    optimizer.assemble_portfolio_params()
    
    x_new = your_new_extension(optimizer)
    
    # 3. Validate
    assert np.all(np.isfinite(x_new)), "Result contains NaN/inf"
    assert np.isclose(optimizer.v @ x_new, 1.0), "Dollar constraint violated"
    metrics = optimizer.portfolio_metrics(x_new)
    assert metrics['sharpe_ratio'] >= 0, "Sharpe should be non-negative"
    
    print("✓ Extension works!")
```

---

## Performance Tips

1. **Avoid loops:** Use NumPy vectorization
```python
   # Slow
   for i in range(M):
       u[i] = np.dot(delta[i], mu)
   
   # Fast
   u = delta @ mu
```

2. **Cache matrix inversions:** Reuse Q⁻¹
```python
   Q_inv = np.linalg.inv(Q)
   x1 = Q_inv @ vector1
   x2 = Q_inv @ vector2
```

3. **Use sparse matrices** for large portfolios

---

Use this guide to extend the implementation with your own ideas.