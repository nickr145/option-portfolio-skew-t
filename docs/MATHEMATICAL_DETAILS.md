# Mathematical Details

Complete and rigorous mathematical exposition with full formulas from Sung & Pirvu (2026).

All formulas are taken directly from the paper with exact page citations.

---

## 1. Skew-Elliptical $t$-Distribution

### Parametrization (Azzalini & Capitanio, 2003)

A random vector $\mathbf{X} = (X_1, \ldots, X_N)$ follows a multivariate skew-elliptical $t$-distribution with parameters:
- Location parameter $\boldsymbol{\mu} \in \mathbb{R}^N$
- Scale parameter $\boldsymbol{\Sigma} \in \mathbb{R}^{N \times N}$ (positive definite)
- Degrees of freedom $\nu \in \mathbb{R}$, $\nu > 0$
- Skewness parameter $\boldsymbol{\omega} \in \mathbb{R}^N$

### Why Skew-*t*?

**Normal distribution assumption underestimates tail risk:**

| Distribution | $P(X > 5\sigma)$ |
|---|---|
| Normal | $\approx 10^{-7}$ (1 in 3.5 million) |
| $t_6$ (skew-*t* in paper) | $\approx 10^{-4}$ (1 in 10,000) |
| Real markets | $\approx 10^{-4}$ (empirical) |

**Skewness matters:** Left tail heavier than right (negative skew in equities). Options have different values depending on tail direction.

### Distribution Characterization

The skew-*t* distribution is characterized by two key constants:

$$c = \sqrt{\frac{\nu}{\pi}} \frac{\Gamma\left(\frac{\nu-1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)}$$

$$\mathbf{h} = \frac{\boldsymbol{\Sigma} \boldsymbol{\omega}}{\sqrt{1 + \boldsymbol{\omega}^T \boldsymbol{\Sigma} \boldsymbol{\omega}}}$$

where:
- $c$: Normalization constant (depends on $\nu$)
- $\mathbf{h}$: Skewness direction (adjusted for scale)

---

## 2. Delta-Gamma Approximation

### Portfolio P&L

For a portfolio of $M$ options on $N$ underlying stocks:

$$V(\mathbf{x}; \mathbf{S}, t) = \mathbf{x}^T \mathbf{v}(\mathbf{S}, t)$$

where:
- $\mathbf{x} = (x_1, \ldots, x_M)$: shares of each option
- $\mathbf{v}(\mathbf{S}, t) = (V_1, \ldots, V_M)$: option values

Over time interval $[t, t + \Delta t]$:

$$\Delta V(\mathbf{x}) = V(\mathbf{x}; \mathbf{S} + \Delta \mathbf{S}, t + \Delta t) - V(\mathbf{x}; \mathbf{S}, t)$$

### Second-Order Taylor Expansion

$$\Delta V(\mathbf{x}) = (\Delta t)\boldsymbol{\theta} + \boldsymbol{\delta}^T (\Delta \mathbf{S}) + \frac{1}{2} (\Delta \mathbf{S})^T \boldsymbol{\Gamma} (\Delta \mathbf{S})$$

where:
- $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_M)$: theta vector (time decay, per option)
- $\boldsymbol{\delta} \in \mathbb{R}^{M \times N}$: delta matrix where $\delta_{mn} = \frac{\partial V_m}{\partial S_n}$
- $\boldsymbol{\Gamma} \in \mathbb{R}^{M \times N \times N}$: gamma tensor where $\Gamma_{mij} = \frac{\partial^2 V_m}{\partial S_i \partial S_j}$

The approximation error is $O(\|\Delta \mathbf{S}\|^3)$. For typical daily moves ($\pm \$2$), error is 3-8%.

---

## 3. Expected P&L and Variance Under Skew-*t* Returns

### Assumption (Page 3)

**Assumption 2.2:** Underlying returns follow skew-*t*:

$$\Delta \mathbf{S} \sim t_{\text{skew}}^N(\boldsymbol{\mu}, \boldsymbol{\Sigma}, \nu, \boldsymbol{\omega})$$

### Main Results

Under this assumption and delta-gamma approximation:

$$\mathbb{E}[\Delta V(\mathbf{x})] = \mathbf{u}^T \mathbf{x}$$

$$\text{Var}[\Delta V(\mathbf{x})] = \frac{1}{2} \mathbf{x}^T \mathbf{Q} \mathbf{x}$$

where $\mathbf{u}$ and $\mathbf{Q}$ are computed from explicit formulas in **Appendix A** (pages 10-11).

---

## 4. Theorem 3.1: Sharpe Ratio Maximization

### Problem Formulation (Page 4)

$$\begin{aligned}
\max_{\mathbf{x} \in \mathbb{R}^M} & \quad \text{Sharpe}[\Delta V(\mathbf{x})] \\
\text{subject to} & \quad \mathbf{x}^T \mathbf{v} = 1
\end{aligned}$$

where the Sharpe ratio is:

$$\text{Sharpe}[\Delta V(\mathbf{x})] = \frac{\mathbf{u}^T \mathbf{x} - r_f}{\sqrt{\frac{1}{2} \mathbf{x}^T \mathbf{Q} \mathbf{x}}}$$

### Solution: Theorem 3.1 (Page 5, Equation 7)

$$\mathbf{x}^{\ast}_{\text{Sharpe}} = \frac{\mathbf{Q}^{-1}(\mathbf{u} - r_f \mathbf{v})}{\mathbf{v}^T \mathbf{Q}^{-1}(\mathbf{u} - r_f \mathbf{v})}$$

### Detailed Derivation (Appendix B.1, Pages 11-12)

The paper uses the classical **fractional programming** technique by parametrizing with $\varepsilon$.

**Step 1: Parametrize by standard deviation**

Set $\varepsilon = \sqrt{\frac{1}{2} \mathbf{x}^T \mathbf{Q} \mathbf{x}}$ (the volatility of the portfolio P&L).

**Step 2: Reformulate as constrained linear program**

Convert the fractional program into:

$$\begin{aligned}
\max_{\mathbf{x}} & \quad \frac{\mathbf{u}^T \mathbf{x} - r_f}{\varepsilon} \\
\text{subject to} & \quad \mathbf{x}^T \mathbf{v} = 1 \\
& \quad \frac{1}{2} \mathbf{x}^T \mathbf{Q} \mathbf{x} = \varepsilon^2
\end{aligned}$$

**Step 3: Construct Lagrangian**

For fixed $\varepsilon > 0$, form:

$$\mathcal{L}(\mathbf{x}; \lambda_1, \lambda_2) = \mathbf{u}^T \mathbf{x} + \lambda_1(\mathbf{x}^T \mathbf{v} - 1) + \lambda_2\left(\frac{1}{2} \mathbf{x}^T \mathbf{Q} \mathbf{x} - \varepsilon^2\right)$$

**Step 4: KKT stationarity condition**

$$\nabla_{\mathbf{x}} \mathcal{L} = \mathbf{u} + \lambda_1 \mathbf{v} + \lambda_2 \mathbf{Q} \mathbf{x} = \mathbf{0}$$

Solving for $\mathbf{x}$:

$$\mathbf{x} = -\frac{1}{\lambda_2} \mathbf{Q}^{-1}(\mathbf{u} + \lambda_1 \mathbf{v})$$

**Step 5: Apply constraint** $\mathbf{x}^T \mathbf{v} = 1$

Substituting the expression for $\mathbf{x}$ into the constraint:

$$-\frac{1}{\lambda_2} (\mathbf{u} + \lambda_1 \mathbf{v})^T \mathbf{Q}^{-1} \mathbf{v} = 1$$

Solving for $\lambda_2$:

$$\lambda_2 = -\left[\mathbf{u}^T \mathbf{Q}^{-1} \mathbf{v} + \lambda_1 \mathbf{v}^T \mathbf{Q}^{-1} \mathbf{v}\right]$$

**Step 6: Optimize over Lagrange multiplier** $\lambda_1$

For any given $\lambda_1$, we have:

$$\mathbf{x}(\lambda_1) = \frac{\mathbf{Q}^{-1}(\mathbf{u} + \lambda_1 \mathbf{v})}{\mathbf{u}^T \mathbf{Q}^{-1} \mathbf{v} + \lambda_1 \mathbf{v}^T \mathbf{Q}^{-1} \mathbf{v}}$$

Taking the derivative with respect to $\lambda_1$ and setting to zero:

$$\frac{d}{d\lambda_1}\left[\frac{\mathbf{u}^T \mathbf{x}(\lambda_1) - r_f}{\sqrt{\frac{1}{2}\mathbf{x}(\lambda_1)^T \mathbf{Q} \mathbf{x}(\lambda_1)}}\right] = 0$$

**Step 7: Closed-form solution**

The optimal $\lambda_1^{\ast}$ satisfies:

$$\lambda_1^{\ast} = -r_f$$

Substituting $\lambda_1^{\ast} = -r_f$ into the expression for $\mathbf{x}(\lambda_1)$:

$$\mathbf{x}^{\ast}_{\text{Sharpe}} = \frac{\mathbf{Q}^{-1}(\mathbf{u} - r_f \mathbf{v})}{\mathbf{v}^T \mathbf{Q}^{-1}(\mathbf{u} - r_f \mathbf{v})}$$

This is **Theorem 3.1** from page 5, Equation (7).

---

## 5. Theorem 3.2: Return-to-VaR Ratio Maximization

### Problem Formulation (Page 4-5)

$$\begin{aligned}
\max_{\mathbf{x} \in \mathbb{R}^M} & \quad \frac{\mathbf{u}^T \mathbf{x} - r_f}{-\mathbf{u}^T \mathbf{x} - N^{-1}(\alpha) \sqrt{\frac{1}{2} \mathbf{x}^T \mathbf{Q} \mathbf{x}}} \\
\text{subject to} & \quad \mathbf{x}^T \mathbf{v} = 1
\end{aligned}$$

where $N^{-1}(\alpha)$ is the inverse CDF of the standard normal at confidence level $\alpha$. 

For $\alpha = 0.01$: $N^{-1}(0.01) \approx -2.326$

The denominator is the Cornish-Fisher VaR:

$$\text{CFVaR}_\alpha[\Delta V(\mathbf{x})] = -\mathbf{u}^T \mathbf{x} - N^{-1}(\alpha) \sqrt{\frac{1}{2} \mathbf{x}^T \mathbf{Q} \mathbf{x}}$$

### Solution: Theorem 3.2 (Page 5, Equation 8)

$$\mathbf{x}^{\ast}_{\text{R-VaR}^{\alpha}_{2}} = \frac{\mathbf{Q}^{-1}[(1 - \lambda^{\ast}) \mathbf{u} - r_f \mathbf{v}]}{\mathbf{v}^T \mathbf{Q}^{-1}[(1 - \lambda^{\ast}) \mathbf{u} - r_f \mathbf{v}]}$$

where $\lambda^{\ast}$ is the optimal scaling parameter determined in **Appendix B.2**.

### Detailed Derivation (Appendix B.2, Pages 12-13)

**Step 1: Parametrize by expected return**

Unlike Theorem 3.1, here we parametrize by the expected return:

$$\varepsilon := \mathbf{u}^T \mathbf{x}$$

**Step 2: Reformulate optimization**

For fixed $\varepsilon$, the objective becomes:

$$\max_{\mathbf{x}} \quad \frac{\varepsilon - r_f}{\varepsilon - N^{-1}(\alpha) \sqrt{\frac{1}{2} \mathbf{x}^T \mathbf{Q} \mathbf{x}}}$$

subject to:

$$\mathbf{x}^T \mathbf{v} = 1, \quad \mathbf{u}^T \mathbf{x} = \varepsilon$$

**Step 3: Key observation for optimization**

Since $\alpha < 1/2$, we have $N^{-1}(\alpha) < 0$, so:

$$\text{Denominator} = \varepsilon + |N^{-1}(\alpha)| \sqrt{\frac{1}{2} \mathbf{x}^T \mathbf{Q} \mathbf{x}} > 0$$

For fixed $\varepsilon$, maximizing the ratio requires **minimizing** $\sqrt{\frac{1}{2} \mathbf{x}^T \mathbf{Q} \mathbf{x}}$.

**Step 4: Convert to quadratic minimization**

$$\min_{\mathbf{x}} \quad \frac{1}{2} \mathbf{x}^T \mathbf{Q} \mathbf{x}$$

subject to:

$$\mathbf{u}^T \mathbf{x} = \varepsilon, \quad \mathbf{v}^T \mathbf{x} = 1$$

This is a **constrained quadratic program** with two linear equality constraints.

**Step 5: Solve constrained QP**

Using standard QP theory, construct the constraint matrix and vector:

$$\mathbf{J} = \begin{pmatrix} \mathbf{u}^T \\ \mathbf{v}^T \end{pmatrix}, \quad \boldsymbol{\psi}(\varepsilon) = \begin{pmatrix} \varepsilon \\ 1 \end{pmatrix}$$

The solution is:

$$\mathbf{x}(\varepsilon) = \mathbf{Q}^{-1} \mathbf{J}^T (\mathbf{J} \mathbf{Q}^{-1} \mathbf{J}^T)^{-1} \boldsymbol{\psi}(\varepsilon)$$

**Step 6: Express variance as quadratic in** $\varepsilon$

Substituting $\mathbf{x}(\varepsilon)$ back:

$$\frac{1}{2} \mathbf{x}(\varepsilon)^T \mathbf{Q} \mathbf{x}(\varepsilon) = A \varepsilon^2 + B \varepsilon + C$$

where:

$$A, B, C = \text{functions of } \mathbf{Q}^{-1}, \mathbf{u}, \mathbf{v}$$

(Explicit formulas in Appendix B.2)

**Step 7: Univariate optimization over** $\varepsilon$

Maximize:

$$f(\varepsilon) := \frac{\varepsilon - r_f}{\varepsilon - N^{-1}(\alpha) \sqrt{A \varepsilon^2 + B \varepsilon + C}}$$

**Step 8: Critical point equation**

Taking $\frac{df}{d\varepsilon} = 0$ yields a polynomial equation. Define intermediate coefficients (page 13):

$$A_{\text{crit}} := 4r_f^2 A - [N^{-1}(\alpha)]^2 [B + 2r_f A]^2$$

$$B_{\text{crit}} := 4r_f^2 B - 2[N^{-1}(\alpha)]^2 [B + 2r_f A][2C + r_f B]$$

$$C_{\text{crit}} := 4r_f^2 C - [N^{-1}(\alpha)]^2 [2C + r_f B]^2$$

The critical points are:

$$\varepsilon_{\pm} = \frac{-B_{\text{crit}} \pm \sqrt{B_{\text{crit}}^2 - 4 A_{\text{crit}} C_{\text{crit}}}}{2 A_{\text{crit}}}$$

**Step 9: Select optimal** $\varepsilon^{\ast}$

Choose the root that maximizes $f(\varepsilon)$. (Paper discusses conditions on page 13.)

**Step 10: Retrieve optimal portfolio**

$$\mathbf{x}^{\ast} = \mathbf{x}(\varepsilon^{\ast})$$

Through algebraic simplification, this yields:

$$\mathbf{x}^{\ast}_{\text{R-VaR}_\alpha} = \frac{\mathbf{Q}^{-1}[(1 - \lambda^{\ast}) \mathbf{u} - r_f \mathbf{v}]}{\mathbf{v}^T \mathbf{Q}^{-1}[(1 - \lambda^{\ast}) \mathbf{u} - r_f \mathbf{v}]}$$

where the parameter $\lambda^{\ast}$ relates $\varepsilon^{\ast}$ to the structural form. (Definition in Appendix B.2.)

---

## 6. Appendix A: Explicit Formulas for $\mathbf{u}$ and $\mathbf{Q}$

### Expected P&L Vector $\mathbf{u}$ (Page 10)

$$\mathbf{u} := \boldsymbol{\zeta} + c \mathbf{B} \mathbf{h} + c \mathbf{D} \mathbf{h}$$

where:

**Location and time decay component:**

$$\boldsymbol{\zeta}_m := (\Delta t)\theta_m + \sum_{n=1}^{N} \delta_{mn} \mu_n + \frac{\nu}{2(\nu-2)} p_m + \xi_m$$

**Gamma-delta-mu interaction term:**

$$p_m := \sum_{i=1}^{N} \sum_{j=1}^{N} \gamma^{[i,j]}_m$$

(trace of $m$-th gamma matrix w.r.t. $\boldsymbol{\Sigma}$)

**Second-order term:**

$$\xi_m := \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \mu_i \mu_j \gamma^{[i,j]}_m$$

**Intermediate vectors and matrices:**

$$\mathbf{B}m := -\sum_{i=1}^{N} \sum_{j=1}^{N} \mu_i \mu_j \gamma^{[i,j]}_m$$

$$\mathbf{D}_{m} := \sum_{i=1}^{N} \sum_{j=1}^{N} \sigma_{ij} \gamma^{[i,j]}_m$$

(More details in Appendix A of paper)

**Skewness constants:**

$$c := \sqrt{\frac{\nu}{\pi}} \frac{\Gamma\left(\frac{\nu-1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)}$$

$$\mathbf{h} := \frac{\boldsymbol{\Sigma} \boldsymbol{\omega}}{\sqrt{1 + \boldsymbol{\omega}^T \boldsymbol{\Sigma} \boldsymbol{\omega}}}$$

---

### Variance-Covariance Matrix $\mathbf{Q}$ (Pages 10-11)

$$\mathbf{Q} := \frac{1}{2}(\widetilde{\mathbf{Q}} + \widetilde{\mathbf{Q}}^T)$$

**Full decomposition:**

$$\widetilde{\mathbf{Q}} := \mathbf{U} + \frac{4c\nu}{\nu-3}(\mathbf{H} + \mathbf{E}) + \frac{2c\nu}{(\nu-2)(\nu-3)}(\mathbf{B} + \mathbf{D}) \mathbf{h} \mathbf{p}^T$$

$$- \frac{2c\nu}{\nu-3}(\mathbf{B} + \mathbf{D}) \mathbf{h} \mathbf{q}^T - 2c^2 (\mathbf{B} + \mathbf{D}) \mathbf{h} \mathbf{h}^T (\mathbf{B} + \mathbf{D})^T$$

**Variance component** $\mathbf{U}$**:**

$$U_{mn} := \frac{2\nu}{\nu-2} \sum_{i,j=1}^{N} (B_i + D_i)_m (B_i + D_i)_n \sigma_{ij}$$

$$+ \frac{\nu}{2(\nu-2)(\nu-4)} \text{tr}[\boldsymbol{\Gamma}_{[m]} \boldsymbol{\Sigma} \boldsymbol{\Gamma}_{[n]} \boldsymbol{\Sigma}]$$

**Gamma-mu interaction matrix** $\mathbf{H}$**:**

$$H_{mn} := \sum_{i,j=1}^{N} \mu_i \gamma^{[i,\cdot]}_m \sigma_{ij} \gamma^{[\cdot,j]}_n h$$

Or equivalently:

$$H_{mn} := \mathbf{p}_m \mathbf{p}_n$$

where $\mathbf{p}_m := \text{tr}[\boldsymbol{\Gamma}_{[m]} \boldsymbol{\Sigma}]$

**Delta-gamma-skewness matrix** $\mathbf{E}$**:**

$$E_{mn} := \sum_{i,j=1}^{N} \delta_{mi} \sigma_{ij} \gamma^{[\cdot,j]}_n h_{\cdot}$$

**Gamma interaction matrix** $\mathbf{R}$**:**

$$R_{mn} := \text{tr}[\boldsymbol{\Gamma}_{[m]} \boldsymbol{\Sigma} \boldsymbol{\Gamma}_{[n]} \boldsymbol{\Sigma}]$$

**Auxiliary vector for skewness:**

$$\mathbf{q}_j := \sum_{i=1}^{N} h_i \gamma^{[i,j]}_m h_{\cdot}$$

---

## 7. Special Cases

### Case 1: Normal Returns ($\nu \to \infty$, $\boldsymbol{\omega} = \mathbf{0}$)

When $\nu \to \infty$ and $\boldsymbol{\omega} = \mathbf{0}$:

$$\lim_{\nu \to \infty, \boldsymbol{\omega} \to \mathbf{0}} t_{\text{skew}}^N(\boldsymbol{\mu}, \boldsymbol{\Sigma}, \nu, \boldsymbol{\omega}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

**Result:** 
- $c \to 0$
- $\mathbf{h} \to \mathbf{0}$
- All skewness terms in $\mathbf{u}$ and $\mathbf{Q}$ vanish
- Reduces to standard Markowitz theory

### Case 2: Student-*t* ($\boldsymbol{\omega} = \mathbf{0}$, $\nu$ fixed)

Pure heavy tails without skewness:

$$u_m := \zeta_m + 0 = \text{(simplified)}$$

$$Q_{mn} := U_{mn} + 0 = \text{(simplified)}$$

Optimal weights still depend on degree of tail heaviness $\nu$.

---

## 8. Notation Reference Table

| Symbol | Meaning | Dimension | Source |
|--------|---------|-----------|--------|
| $\mathbf{S}$ | Stock prices | $(N,)$ | Given |
| $\mathbf{v}$ | Option values | $(M,)$ | Black-Scholes |
| $\mathbf{x}$ | Option shares (optimization variable) | $(M,)$ | Solution |
| $\mathbf{u}$ | Expected P&L vector | $(M,)$ | Appendix A |
| $\mathbf{Q}$ | Variance-covariance matrix | $(M, M)$ | Appendix A |
| $\boldsymbol{\delta}$ | Delta matrix | $(M, N)$ | Black-Scholes |
| $\boldsymbol{\Gamma}$ | Gamma tensor | $(M, N, N)$ | Black-Scholes |
| $\boldsymbol{\theta}$ | Theta vector | $(M,)$ | Black-Scholes |
| $\boldsymbol{\mu}$ | Location vector (skew-*t*) | $(N,)$ | Fit |
| $\boldsymbol{\Sigma}$ | Scale matrix (skew-*t*) | $(N, N)$ | Fit |
| $\nu$ | Degrees of freedom | scalar | Fit |
| $\boldsymbol{\omega}$ | Skewness vector (skew-*t*) | $(N,)$ | Fit |
| $r_f$ | Risk-free rate | scalar | Given |
| $T$ | Time to expiry | scalar | Given |
| $\Delta t$ | Time step | scalar | Given |
| $\varepsilon$ | Standard deviation (Sharpe) or expected return (R-VaR) | scalar | Parametrization |
| $\lambda^{\ast}$ | Optimal scaling parameter (R-VaR) | scalar | Appendix B.2 |
| $N^{-1}(\alpha)$ | Inverse standard normal CDF at level $\alpha$ | scalar | VaR |
| $\alpha$ | VaR confidence level (e.g., 0.01) | scalar | Given |
| $c$ | Skewness normalization constant | scalar | Definition |
| $\mathbf{h}$ | Adjusted skewness direction | $(N,)$ | Definition |

---

## 9. Critical Page References

| Result | Page | Equation |
|--------|------|----------|
| Problem statement (Sharpe & R-VaR) | 4-5 | --- |
| Theorem 3.1 solution | 5 | (7) |
| Theorem 3.2 solution | 5 | (8) |
| Appendix A: Explicit $\mathbf{u}$ and $\mathbf{Q}$ formulas | 10-11 | A.1, A.2 |
| Appendix B.1: Proof of Theorem 3.1 | 11-12 | --- |
| Appendix B.2: Proof of Theorem 3.2 | 12-13 | --- |

---

## 10. Key Implementation Notes

### Computing $\mathbf{Q}$ Complexity

The full formula for $\widetilde{\mathbf{Q}}$ involves:
- Matrix inversions: $\mathbf{Q}^{-1}$ computation $O(M^3)$
- Multiple gamma-sigma contractions: $O(M^2 N^2)$
- Outer products and tensor contractions: $O(M^2)$

**Total: $O(M^3 + M^2 N^2)$**

### Numerical Stability

1. **Check $\boldsymbol{\Sigma}$ positive definiteness:**
$$\lambda_{\min}(\boldsymbol{\Sigma}) > 10^{-6}$$

2. **Use Cholesky decomposition:**
$$\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^T$$

3. **Regularize $\mathbf{Q}$ if ill-conditioned:**
$$\mathbf{Q}_{\text{reg}} := \mathbf{Q} + \epsilon \mathbf{I}, \quad \epsilon \approx 10^{-4}$$

4. **Monitor condition number:**
$$\kappa(\mathbf{Q}) := \frac{\lambda_{\max}(\mathbf{Q})}{\lambda_{\min}(\mathbf{Q})}$$

If $\kappa > 100$, consider regularization.

---

## 11. References

### Original Paper

**Sung, K. & Pirvu, T.A.** (2026). Sharpe Ratio and Return-VaR Ratio Maximization for Option Portfolios with Skew-Elliptical $t$ Underlying Returns. *arXiv preprint arXiv:2606.17032v1*.

Key sections:
- **Theorem 3.1** (Sharpe): Page 5, Equation (7)
- **Theorem 3.2** (R-VaR): Page 5, Equation (8)  
- **Appendix A** (Explicit forms for $\mathbf{u}$, $\mathbf{Q}$): Pages 10-11
- **Appendix B.1** (Proof of Theorem 3.1): Pages 11-12
- **Appendix B.2** (Proof of Theorem 3.2): Pages 12-13

### Foundational References

**Azzalini, A. & Capitanio, A.** (2003). Distributions generated by perturbation of symmetry with emphasis on a multivariate skew $t$-distribution. *Journal of the Royal Statistical Society. Series B (Statistical Methodology)*, 65(2), 367-389.

**Markowitz, H.** (1952). Portfolio selection. *Journal of Finance*, 7(1), 77-91.

**Jorion, P.** (2007). *Value at Risk: The New Benchmark for Managing Financial Risk* (3rd ed.). McGraw-Hill.

### Related Work

**Hu, W. & Kercheval, A.N.** (2010). Portfolio optimization for Student $t$ and skewed $t$ returns. *Quantitative Finance*, 10(1), 91-105.

**Cui, X., Zhu, S., Sun, X., & Li, D.** (2013). Nonlinear portfolio selection using approximate parametric value-at-risk. *Journal of Banking & Finance*, 37(6), 2124-2139.
